from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer

from vlm_micro.cauldron import iter_qa_pairs, yes_no_answer


DEFAULT_SUBSETS = ("vqav2", "clevr", "vsr")
DEFAULT_TOKENIZER = "HuggingFaceTB/SmolVLM-256M-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a compact prompt-level yes/no dataset using a fixed token vocabulary budget."
        )
    )
    parser.add_argument("--source-root", type=Path, default=Path("data/the_cauldron"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/the_cauldron_yes_no_vsr_token1000"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/reports/cauldron_yes_no_vsr_token1000_summary.json"),
    )
    parser.add_argument("--subsets", nargs="+", default=list(DEFAULT_SUBSETS))
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--token-budget", type=int, default=1000)
    parser.add_argument("--max-examples-per-subset", type=int)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strip-last-line", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def iter_subset_rows(source_root: Path, subset: str, max_examples: int | None) -> Any:
    hf_path = source_root / subset
    if hf_path.exists():
        dataset = load_from_disk(hf_path)
        if "texts" in dataset.column_names:
            dataset = dataset.select_columns(["texts"])
        limit = len(dataset) if max_examples is None else min(len(dataset), max_examples)
        for index in range(limit):
            yield index, dataset[index]
        return

    parquet_path = source_root / "_parquet" / subset
    if parquet_path.exists():
        seen = 0
        for parquet_file in sorted(parquet_path.glob("*.parquet")):
            table = pq.read_table(parquet_file, columns=["texts"])
            for row in table.to_pylist():
                if max_examples is not None and seen >= max_examples:
                    return
                yield seen, row
                seen += 1
        return

    raise FileNotFoundError(f"Missing subset {subset} under {source_root}")


def strip_last_prompt_line(prompt: str) -> tuple[str, str | None]:
    lines = prompt.splitlines()
    nonempty_indices = [index for index, line in enumerate(lines) if line.strip()]
    if not nonempty_indices:
        return prompt.strip(), None

    last_index = nonempty_indices[-1]
    removed = lines[last_index].strip()
    compact_lines = lines[:last_index] + lines[last_index + 1 :]
    compact = "\n".join(line.rstrip() for line in compact_lines).strip()
    return compact, removed


def collect_yes_no_prompts(
    source_root: Path,
    subsets: list[str],
    tokenizer: Any,
    max_examples_per_subset: int | None,
    strip_last_line: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for subset in subsets:
        for original_index, sample in iter_subset_rows(source_root, subset, max_examples_per_subset):
            for qa_index, (prompt, answer) in enumerate(iter_qa_pairs(sample)):
                label = yes_no_answer(answer)
                if label is None:
                    continue
                student_prompt, removed_last_line = (
                    strip_last_prompt_line(prompt) if strip_last_line else (prompt.strip(), None)
                )
                token_ids = tokenizer(student_prompt, add_special_tokens=False)["input_ids"]
                source = None
                texts = sample.get("texts")
                if isinstance(texts, list) and qa_index < len(texts) and isinstance(texts[qa_index], dict):
                    raw_source = texts[qa_index].get("source")
                    if isinstance(raw_source, str):
                        source = raw_source
                records.append(
                    {
                        "source_subset": subset,
                        "original_index": original_index,
                        "qa_index": qa_index,
                        "source": source or subset,
                        "teacher_prompt": prompt,
                        "student_prompt": student_prompt,
                        "removed_last_line": removed_last_line,
                        "answer": label,
                        "student_token_ids": token_ids,
                        "student_token_count": len(token_ids),
                        "student_distinct_token_count": len(set(token_ids)),
                    }
                )
    return records


def select_token_vocabulary(records: list[dict[str, Any]], token_budget: int) -> list[int]:
    counter: Counter[int] = Counter()
    for record in records:
        counter.update(record["student_token_ids"])
    return [token_id for token_id, _ in counter.most_common(token_budget)]


def summarize(records: list[dict[str, Any]], selected_token_ids: set[int]) -> dict[str, Any]:
    by_subset: dict[str, dict[str, Any]] = {}
    for record in records:
        subset = record["source_subset"]
        row = by_subset.setdefault(
            subset,
            {
                "candidate_prompts": 0,
                "kept_prompts": 0,
                "answers": Counter(),
                "candidate_token_ids": set(),
                "kept_token_ids": set(),
                "removed_last_lines": Counter(),
            },
        )
        token_set = set(record["student_token_ids"])
        row["candidate_prompts"] += 1
        row["answers"].update([record["answer"]])
        row["candidate_token_ids"].update(token_set)
        if record["removed_last_line"] is not None:
            row["removed_last_lines"].update([record["removed_last_line"]])
        if token_set <= selected_token_ids:
            row["kept_prompts"] += 1
            row["kept_token_ids"].update(token_set)

    serializable: dict[str, Any] = {}
    for subset, row in by_subset.items():
        candidate_prompts = int(row["candidate_prompts"])
        kept_prompts = int(row["kept_prompts"])
        serializable[subset] = {
            "candidate_prompts": candidate_prompts,
            "kept_prompts": kept_prompts,
            "kept_prompt_fraction": kept_prompts / candidate_prompts if candidate_prompts else 0.0,
            "answers": dict(row["answers"]),
            "candidate_distinct_token_ids": len(row["candidate_token_ids"]),
            "kept_distinct_token_ids": len(row["kept_token_ids"]),
            "removed_last_lines": dict(row["removed_last_lines"].most_common()),
        }
    return serializable


def write_dataset(records: list[dict[str, Any]], selected_token_ids: set[int], output_root: Path) -> None:
    kept = [record for record in records if set(record["student_token_ids"]) <= selected_token_ids]
    by_subset = {
        subset: Dataset.from_list([record for record in kept if record["source_subset"] == subset])
        for subset in sorted({record["source_subset"] for record in records})
    }
    dataset_dict = DatasetDict({"combined": Dataset.from_list(kept), **by_subset})
    tmp_output = output_root.with_name(f".{output_root.name}.tmp")
    if tmp_output.exists():
        shutil.rmtree(tmp_output)
    if output_root.exists():
        shutil.rmtree(output_root)
    dataset_dict.save_to_disk(tmp_output)
    tmp_output.rename(output_root)


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    if args.output_root.exists() and not args.force:
        raise FileExistsError(f"{args.output_root} exists. Pass --force to replace it.")

    records = collect_yes_no_prompts(
        source_root=args.source_root,
        subsets=[str(subset) for subset in args.subsets],
        tokenizer=tokenizer,
        max_examples_per_subset=args.max_examples_per_subset,
        strip_last_line=args.strip_last_line,
    )
    selected_token_ids = select_token_vocabulary(records, args.token_budget)
    selected_token_set = set(selected_token_ids)
    subset_summary = summarize(records, selected_token_set)
    kept_records = [record for record in records if set(record["student_token_ids"]) <= selected_token_set]

    write_dataset(records, selected_token_set, args.output_root)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(args.source_root),
        "output_root": str(args.output_root),
        "subsets": [str(subset) for subset in args.subsets],
        "tokenizer": args.tokenizer,
        "token_budget": args.token_budget,
        "strip_last_line": args.strip_last_line,
        "student_prompt": "teacher prompt with final non-empty line removed before token analysis",
        "teacher_prompt": "original prompt retained for teacher-side distillation",
        "token_selection": "top token IDs by occurrence count over combined compact student prompts",
        "candidate_prompts": len(records),
        "kept_prompts": len(kept_records),
        "kept_prompt_fraction": len(kept_records) / len(records) if records else 0.0,
        "selected_token_ids": selected_token_ids,
        "subsets_summary": subset_summary,
    }
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote dataset: {args.output_root}")
    print(f"Wrote report: {args.report}")
    print(
        f"Kept {report['kept_prompts']} / {report['candidate_prompts']} prompts "
        f"({100 * report['kept_prompt_fraction']:.2f}%)"
    )


if __name__ == "__main__":
    main()
