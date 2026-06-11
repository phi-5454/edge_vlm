#!/usr/bin/env python3
"""Extend a compact TallyQA prompt embedding artifact with prompt aliases.

The existing prompt artifact compacts only the SmolVLM token IDs seen in the
training prompt classes. This script preserves that compact vocabulary exactly
and appends any missing token rows needed by synonyms. That means existing
checkpoints can be expanded in a controlled way instead of silently reordering
their prompt embedding table.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


DEFAULT_MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"
DEFAULT_INPUT = Path("artifacts/models/tallyqa_smolvlm_prompt_embeddings_letterbox.pt")
DEFAULT_SYNONYMS = Path("conf/tallyqa_prompt_synonyms.json")
DEFAULT_OUTPUT = Path("artifacts/models/tallyqa_smolvlm_prompt_embeddings_letterbox_synonyms.pt")
DEFAULT_REPORT = Path("artifacts/reports/tallyqa_prompt_embeddings_letterbox_synonyms_summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--synonyms", type=Path, default=DEFAULT_SYNONYMS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def normalize_prompt(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def load_synonyms(path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    synonyms: dict[str, list[str]] = {}
    if isinstance(payload, dict):
        iterator: Iterable[tuple[str, Any]] = payload.items()
    elif isinstance(payload, list):
        iterator = (
            (str(row["prompt"]), row.get("aliases", row.get("synonyms", [])))
            for row in payload
            if isinstance(row, dict)
        )
    else:
        raise ValueError(f"{path} must contain a JSON object or list of objects.")

    for prompt, aliases in iterator:
        prompt_key = normalize_prompt(prompt)
        if isinstance(aliases, str):
            alias_values = [aliases]
        elif isinstance(aliases, list):
            alias_values = [str(alias) for alias in aliases]
        else:
            raise ValueError(f"Aliases for {prompt!r} must be a string or list.")
        cleaned = []
        seen: set[str] = set()
        for alias in alias_values:
            alias = normalize_prompt(alias)
            if alias and alias != prompt_key and alias not in seen:
                cleaned.append(alias)
                seen.add(alias)
        if cleaned:
            synonyms[prompt_key] = cleaned
    return synonyms


def tokenize_prompt(processor: Any, prompt: str) -> tuple[list[int], list[str]]:
    token_ids = [
        int(token_id)
        for token_id in processor.tokenizer(prompt, add_special_tokens=False)["input_ids"]
    ]
    if not token_ids:
        raise ValueError(f"Prompt {prompt!r} produced no token IDs.")
    tokens = processor.tokenizer.convert_ids_to_tokens(token_ids)
    return token_ids, [str(token) for token in tokens]


def load_missing_embedding_rows(
    model_name: str,
    teacher_token_ids: list[int],
    local_files_only: bool,
    trust_remote_code: bool,
    torch_dtype: str,
) -> torch.Tensor:
    if not teacher_token_ids:
        return torch.empty((0, 0), dtype=torch.float32)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
    )
    embedding = model.get_input_embeddings()
    if embedding is None:
        raise ValueError(f"{model_name} does not expose input embeddings.")
    return embedding.weight.detach().cpu()[teacher_token_ids].float().clone()


def padded_alias_tensors(alias_rows: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    if not alias_rows:
        return torch.zeros((0, 0), dtype=torch.long), torch.zeros((0, 0), dtype=torch.bool)
    max_length = max(len(row["compact_token_ids"]) for row in alias_rows)
    token_ids = torch.zeros((len(alias_rows), max_length), dtype=torch.long)
    attention_mask = torch.zeros((len(alias_rows), max_length), dtype=torch.bool)
    for row_index, row in enumerate(alias_rows):
        compact_ids = [int(value) for value in row["compact_token_ids"]]
        length = len(compact_ids)
        token_ids[row_index, :length] = torch.tensor(compact_ids, dtype=torch.long)
        attention_mask[row_index, :length] = True
        row["alias_prompt_row"] = row_index
        row["attention_mask"] = [True] * length
    return token_ids, attention_mask


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.force and not args.dry_run:
        raise FileExistsError(f"{args.output} exists. Pass --force to replace it.")
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    if not args.synonyms.exists():
        raise FileNotFoundError(args.synonyms)

    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    required = {"teacher_token_ids", "teacher_to_compact", "embedding_rows", "prompt_classes"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"{args.input} is missing required fields: {sorted(missing)}")

    synonyms = load_synonyms(args.synonyms)
    prompt_classes = list(payload["prompt_classes"])
    prompt_by_name = {normalize_prompt(row["item"]): row for row in prompt_classes}
    selected_synonyms = {
        prompt: aliases
        for prompt, aliases in synonyms.items()
        if prompt in prompt_by_name
    }
    skipped_synonyms = {
        prompt: aliases
        for prompt, aliases in synonyms.items()
        if prompt not in prompt_by_name
    }

    processor = AutoProcessor.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    old_teacher_token_ids = [int(value) for value in payload["teacher_token_ids"]]
    teacher_to_compact = {
        int(teacher_id): int(compact_id)
        for teacher_id, compact_id in payload["teacher_to_compact"].items()
    }
    if sorted(teacher_to_compact.values()) != list(range(1, len(teacher_to_compact) + 1)):
        raise ValueError("Base teacher_to_compact mapping is not contiguous from 1.")

    next_compact_id = len(teacher_to_compact) + 1
    new_teacher_token_ids: list[int] = []
    alias_rows: list[dict[str, Any]] = []
    for prompt, aliases in selected_synonyms.items():
        source_row = prompt_by_name[prompt]
        for alias in aliases:
            token_ids, tokens = tokenize_prompt(processor, alias)
            compact_ids = []
            for teacher_id in token_ids:
                if teacher_id not in teacher_to_compact:
                    teacher_to_compact[teacher_id] = next_compact_id
                    next_compact_id += 1
                    new_teacher_token_ids.append(teacher_id)
                compact_ids.append(teacher_to_compact[teacher_id])
            alias_rows.append(
                {
                    "source_class_id": int(source_row["class_id"]),
                    "source_item": str(source_row["item"]),
                    "alias": alias,
                    "teacher_token_ids": token_ids,
                    "teacher_tokens": tokens,
                    "compact_token_ids": compact_ids,
                    "token_count": len(token_ids),
                }
            )

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input),
        "output": str(args.output),
        "synonyms": str(args.synonyms),
        "teacher_model": args.model,
        "base_prompt_classes": len(prompt_classes),
        "synonym_prompts_requested": len(synonyms),
        "synonym_prompts_matched": len(selected_synonyms),
        "synonym_aliases_matched": sum(len(values) for values in selected_synonyms.values()),
        "synonym_prompts_skipped": skipped_synonyms,
        "base_teacher_tokens": len(old_teacher_token_ids),
        "new_teacher_tokens": len(new_teacher_token_ids),
        "expanded_teacher_tokens": len(old_teacher_token_ids) + len(new_teacher_token_ids),
        "alias_rows": len(alias_rows),
        "new_teacher_token_ids": new_teacher_token_ids,
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return

    base_embedding_rows = payload["embedding_rows"].float()
    if new_teacher_token_ids:
        new_rows = load_missing_embedding_rows(
            model_name=args.model,
            teacher_token_ids=new_teacher_token_ids,
            local_files_only=args.local_files_only,
            trust_remote_code=args.trust_remote_code,
            torch_dtype=str(args.torch_dtype),
        )
        if new_rows.shape[1] != base_embedding_rows.shape[1]:
            raise ValueError(
                f"New embedding dim {new_rows.shape[1]} does not match "
                f"base dim {base_embedding_rows.shape[1]}."
            )
        embedding_rows = torch.cat([base_embedding_rows, new_rows], dim=0)
    else:
        embedding_rows = base_embedding_rows.clone()

    alias_prompt_token_ids, alias_prompt_attention_mask = padded_alias_tensors(alias_rows)
    expanded_teacher_token_ids = tuple(old_teacher_token_ids + new_teacher_token_ids)
    output_payload = {
        **payload,
        "schema_version": max(int(payload.get("schema_version", 1)), 2),
        "created_at_utc": summary["created_at_utc"],
        "base_prompt_artifact": str(args.input),
        "synonym_source": str(args.synonyms),
        "teacher_token_ids": expanded_teacher_token_ids,
        "teacher_to_compact": teacher_to_compact,
        "embedding_rows": embedding_rows,
        "alias_prompt_classes": alias_rows,
        "alias_prompt_token_ids": alias_prompt_token_ids,
        "alias_prompt_attention_mask": alias_prompt_attention_mask,
        "alias_prompt_by_source_item": selected_synonyms,
        "compatibility": (
            "Existing compact IDs are preserved. New synonym-only teacher token IDs are "
            "appended after the base embedding rows."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_payload, args.output)

    summary.update(
        {
            "embedding_dim": int(embedding_rows.shape[1]),
            "embedding_rows_shape": list(embedding_rows.shape),
            "alias_prompt_token_shape": list(alias_prompt_token_ids.shape),
        }
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote synonym-expanded prompt artifact: {args.output}")
    print(f"Wrote report: {args.report}")


if __name__ == "__main__":
    main()
