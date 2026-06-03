from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


DEFAULT_MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"
DEFAULT_DATASET = Path("data/tallyqa_cauldron_target_mobilenet224")
DEFAULT_OUTPUT = Path("artifacts/models/tallyqa_smolvlm_prompt_embeddings.pt")
DEFAULT_REPORT = Path("artifacts/reports/tallyqa_prompt_embeddings_summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute compact SmolVLM prompt embeddings for TallyQA item classes."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--classes", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_classes(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a JSON list.")
    rows = sorted(rows, key=lambda row: int(row["class_id"]))
    expected = list(range(len(rows)))
    actual = [int(row["class_id"]) for row in rows]
    if actual != expected:
        raise ValueError(f"class_id values must be contiguous from 0; got {actual[:10]}...")
    return rows


def tokenize_classes(
    processor: Any,
    class_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], tuple[int, ...], int, dict[int, int]]:
    tokenized_rows: list[dict[str, Any]] = []
    token_counter: Counter[int] = Counter()
    max_length = 0
    for row in class_rows:
        item = str(row["item"])
        token_ids = [int(token_id) for token_id in processor.tokenizer(
            item,
            add_special_tokens=False,
        )["input_ids"]]
        if not token_ids:
            raise ValueError(f"Class {item!r} produced no token IDs.")
        tokens = processor.tokenizer.convert_ids_to_tokens(token_ids)
        token_counter.update(token_ids)
        max_length = max(max_length, len(token_ids))
        tokenized_rows.append(
            {
                **row,
                "teacher_token_ids": token_ids,
                "teacher_tokens": tokens,
                "token_count": len(token_ids),
            }
        )
    teacher_token_ids = tuple(sorted(token_counter))
    teacher_to_compact = {
        teacher_id: compact_id
        for compact_id, teacher_id in enumerate(teacher_token_ids, start=1)
    }
    return tokenized_rows, teacher_token_ids, max_length, teacher_to_compact


def compact_prompt_tensors(
    tokenized_rows: list[dict[str, Any]],
    teacher_to_compact: dict[int, int],
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    prompt_token_ids = torch.zeros((len(tokenized_rows), max_length), dtype=torch.long)
    prompt_attention_mask = torch.zeros((len(tokenized_rows), max_length), dtype=torch.bool)
    enriched_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(tokenized_rows):
        compact_ids = [teacher_to_compact[int(token_id)] for token_id in row["teacher_token_ids"]]
        length = len(compact_ids)
        prompt_token_ids[row_index, :length] = torch.tensor(compact_ids, dtype=torch.long)
        prompt_attention_mask[row_index, :length] = True
        enriched_rows.append(
            {
                **row,
                "compact_token_ids": compact_ids,
                "attention_mask": [True] * length,
            }
        )
    return prompt_token_ids, prompt_attention_mask, enriched_rows


def load_embedding_rows(
    model_name: str,
    teacher_token_ids: tuple[int, ...],
    local_files_only: bool,
    trust_remote_code: bool,
    torch_dtype: str,
) -> torch.Tensor:
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
    )
    embedding = model.get_input_embeddings()
    if embedding is None:
        raise ValueError(f"{model_name} does not expose input embeddings.")
    return embedding.weight.detach().cpu()[list(teacher_token_ids)].float().clone()


def masked_mean_prompt_embeddings(
    embedding_rows: torch.Tensor,
    prompt_token_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
) -> torch.Tensor:
    pad_row = torch.zeros((1, embedding_rows.shape[1]), dtype=embedding_rows.dtype)
    compact_embedding_rows = torch.cat([pad_row, embedding_rows], dim=0)
    embedded = compact_embedding_rows[prompt_token_ids]
    mask = prompt_attention_mask.unsqueeze(-1).to(embedded.dtype)
    return (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


def length_counts(tokenized_rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(int(row["token_count"]) for row in tokenized_rows)
    return {str(length): counts[length] for length in sorted(counts)}


def main() -> None:
    args = parse_args()
    classes_path = args.classes or args.dataset / "classes.json"
    if args.output.exists() and not args.force and not args.dry_run:
        raise FileExistsError(f"{args.output} exists. Pass --force to replace it.")

    processor = AutoProcessor.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    class_rows = load_classes(classes_path)
    tokenized_rows, teacher_token_ids, max_length, teacher_to_compact = tokenize_classes(
        processor,
        class_rows,
    )
    prompt_token_ids, prompt_attention_mask, enriched_rows = compact_prompt_tensors(
        tokenized_rows,
        teacher_to_compact,
        max_length,
    )

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "teacher_model": args.model,
        "dataset": str(args.dataset),
        "classes": str(classes_path),
        "output": str(args.output),
        "class_count": len(class_rows),
        "unique_teacher_tokens": len(teacher_token_ids),
        "compact_embedding_rows_including_padding": len(teacher_token_ids) + 1,
        "max_prompt_tokens": max_length,
        "token_length_counts": length_counts(tokenized_rows),
        "teacher_token_ids": list(teacher_token_ids),
        "prompt_tensor_shape": list(prompt_token_ids.shape),
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return

    embedding_rows = load_embedding_rows(
        model_name=args.model,
        teacher_token_ids=teacher_token_ids,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=str(args.torch_dtype),
    )
    pooled_prompt_embeddings = masked_mean_prompt_embeddings(
        embedding_rows,
        prompt_token_ids,
        prompt_attention_mask,
    )

    payload = {
        "schema_version": 1,
        "created_at_utc": summary["created_at_utc"],
        "teacher_model": args.model,
        "classes_path": str(classes_path),
        "teacher_token_ids": teacher_token_ids,
        "teacher_to_compact": teacher_to_compact,
        "prompt_classes": enriched_rows,
        "prompt_token_ids": prompt_token_ids,
        "prompt_attention_mask": prompt_attention_mask,
        "embedding_rows": embedding_rows,
        "pooled_prompt_embeddings": pooled_prompt_embeddings,
        "pooling": "masked_mean_over_compact_smolvlm_token_embeddings",
        "pad_id": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)

    summary.update(
        {
            "embedding_dim": int(embedding_rows.shape[1]),
            "embedding_rows_shape": list(embedding_rows.shape),
            "pooled_prompt_embeddings_shape": list(pooled_prompt_embeddings.shape),
            "pooling": payload["pooling"],
            "pad_id": 0,
        }
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote prompt embedding artifact: {args.output}")
    print(f"Wrote report: {args.report}")


if __name__ == "__main__":
    main()
