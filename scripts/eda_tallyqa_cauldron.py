from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import load_from_disk

from eda_tallyqa import (
    analyze_rows,
    load_pruned_suffixes,
    strip_plot_data,
    write_item_top200,
    write_split_plots,
)


ANSWER_RE = re.compile(r"-?\d+")
BRIEF_ANSWER_INSTRUCTION = "Give a very brief answer."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TallyQA EDA on the Cauldron-formatted subset.")
    parser.add_argument("--dataset", type=Path, default=Path("data/the_cauldron/tallyqa"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/reports/tallyqa_cauldron_eda"))
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--include-split", choices=["all", "trainval", "test"], default="all")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument(
        "--coverage-points",
        type=int,
        nargs="*",
        default=[10, 25, 50, 100, 250, 500, 1000],
    )
    parser.add_argument("--suffix-min-support", type=int, default=250)
    parser.add_argument("--suffix-min-children", type=int, default=25)
    parser.add_argument("--suffix-max-depth", type=int, default=8)
    parser.add_argument("--suffix-max-top-child-fraction", type=float, default=0.5)
    parser.add_argument(
        "--pruned-suffixes",
        type=Path,
        default=Path("artifacts/reports/tallyqa_cauldron_eda/frontier_suffixes_pruned.txt"),
        help="Optional tab-separated pruned suffix list to use for template item filtering.",
    )
    return parser.parse_args()


def clean_question(question: str) -> str:
    lines = [line.strip() for line in question.splitlines() if line.strip()]
    if lines and lines[-1] == BRIEF_ANSWER_INSTRUCTION:
        lines = lines[:-1]
    return " ".join(lines).strip()


def parse_answer(answer: Any) -> int | None:
    if isinstance(answer, int):
        return answer
    if not isinstance(answer, str):
        return None
    match = ANSWER_RE.search(answer)
    return int(match.group(0)) if match else None


def load_split_indices(path: Path | None, include_split: str) -> set[int] | None:
    if path is None or include_split == "all":
        return None
    indices: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if str(row["split"]) == include_split:
                indices.add(int(row["source_row_index"]))
    return indices


def load_cauldron_rows(
    dataset_path: Path,
    include_image_indices: set[int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset = load_from_disk(str(dataset_path)).select_columns(["texts"])
    rows: list[dict[str, Any]] = []
    image_rows = 0
    included_image_rows = 0
    skipped_messages = 0
    for image_index, sample in enumerate(dataset):
        image_rows += 1
        if include_image_indices is not None and image_index not in include_image_indices:
            continue
        included_image_rows += 1
        texts = sample.get("texts")
        if not isinstance(texts, list):
            continue
        for text_index, message in enumerate(texts):
            if not isinstance(message, dict):
                skipped_messages += 1
                continue
            question = message.get("user")
            answer = parse_answer(message.get("assistant"))
            if not isinstance(question, str) or answer is None:
                skipped_messages += 1
                continue
            rows.append(
                {
                    "image": f"cauldron_row/{image_index}",
                    "answer": answer,
                    "data_source": str(message.get("source") or "TallyQA"),
                    "question": clean_question(question),
                    "image_id": image_index,
                    "question_id": f"{image_index}:{text_index}",
                }
            )
    metadata = {
        "image_rows": image_rows,
        "included_image_rows": included_image_rows,
        "qa_pairs": len(rows),
        "skipped_messages": skipped_messages,
        "question_cleaning": f"removed trailing '{BRIEF_ANSWER_INSTRUCTION}' instruction line",
        "answer_normalization": "first signed integer parsed from assistant string, e.g. '2.' -> 2",
    }
    return rows, metadata


def write_suffix_lists(report: dict[str, Any], output_dir: Path) -> list[Path]:
    outputs = []
    suffix_list = output_dir / "frontier_suffixes.txt"
    combined_suffixes = report["combined"]["suffix_trie"]["frontier_covering_suffixes"]
    suffix_list.write_text(
        "\n".join(
            f"{index:03d}\t{row['suffix']}\t{row['support']}\t{row['frequency']:.8f}"
            for index, row in enumerate(combined_suffixes, start=1)
        )
        + "\n",
        encoding="utf-8",
    )
    outputs.append(suffix_list)

    selected_suffix_list = output_dir / "filter_suffixes_used.txt"
    selected_suffixes = report["combined"]["suffix_trie"]["selected_filter_suffixes"]
    selected_suffix_list.write_text(
        "\n".join(
            f"{index:03d}\t{row['suffix']}\t{row['support']}\t{row['frequency']:.8f}"
            for index, row in enumerate(selected_suffixes, start=1)
        )
        + "\n",
        encoding="utf-8",
    )
    outputs.append(selected_suffix_list)
    return outputs


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    include_image_indices = load_split_indices(args.split_manifest, args.include_split)
    rows, dataset_metadata = load_cauldron_rows(args.dataset, include_image_indices)
    pruned_suffixes = load_pruned_suffixes(args.pruned_suffixes)
    filter_suffix_source = str(args.pruned_suffixes) if pruned_suffixes else "auto_frontier_covering_suffixes"

    summary = analyze_rows(
        rows,
        "train",
        args.top_k,
        args.coverage_points,
        args.suffix_min_support,
        args.suffix_min_children,
        args.suffix_max_depth,
        args.suffix_max_top_child_fraction,
        pruned_suffixes or None,
        filter_suffix_source,
    )
    combined = analyze_rows(
        rows,
        "combined",
        args.top_k,
        args.coverage_points,
        args.suffix_min_support,
        args.suffix_min_children,
        args.suffix_max_depth,
        args.suffix_max_top_child_fraction,
        pruned_suffixes or None,
        filter_suffix_source,
    )

    for item in (summary, combined):
        write_split_plots(item, args.output_dir, args.top_k)
    item_outputs = [write_item_top200(item, args.output_dir) for item in (summary, combined)]

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": "uv run python scripts/eda_tallyqa_cauldron.py",
        "dataset_path": str(args.dataset),
        "split_manifest": str(args.split_manifest) if args.split_manifest is not None else None,
        "include_split": args.include_split,
        "dataset_metadata": dataset_metadata,
        "splits": {"train": strip_plot_data(summary)},
        "combined": strip_plot_data(combined),
        "normalization": "lowercase plus regex [a-z0-9]+(?:'[a-z0-9]+)?",
        "manual_intervention": {
            "pruned_suffixes_path": str(args.pruned_suffixes),
            "status": "used" if pruned_suffixes else "not_found_using_auto_frontier",
            "instruction": (
                "Inspect frontier_suffixes.txt and copy/edit desired rows into "
                "frontier_suffixes_pruned.txt, then rerun this script to update item filtering."
            ),
        },
        "reasoning": (
            "This repeats the raw TallyQA suffix-frontier and item-frequency EDA on the "
            "Cauldron-formatted tallyqa subset after removing the prompt instruction suffix."
        ),
    }
    output = args.output_dir / "summary.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    suffix_outputs = write_suffix_lists(report, args.output_dir)

    print(output)
    for path in suffix_outputs + item_outputs:
        print(path)


if __name__ == "__main__":
    main()
