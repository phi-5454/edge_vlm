#!/usr/bin/env python3
"""Export a small Coral prompt lookup table for board smoke tests."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.cache_coral_micro_tallyqa_teacher import load_examples, normalize_prompt
from scripts.export_tallyqa_prompt_embedding_lookup import write_header


DEFAULT_FULL_LOOKUP = Path("artifacts/exports/coral/prompt_embedding_lookup")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-lookup-dir", type=Path, default=DEFAULT_FULL_LOOKUP)
    parser.add_argument("--dataset", type=Path, default=Path("data/tallyqa_cauldron_target_mobilenet224_letterbox"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--header-name", default="tallyqa_prompt_embedding_lookup.h")
    parser.add_argument("--max-examples", type=int, default=8)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--prompt", action="append", default=[])
    return parser.parse_args()


def prompt_candidates(row: dict[str, Any]) -> list[str]:
    values = [
        row.get("student_prompt"),
        row.get("item"),
        row.get("teacher_prompt_clean"),
        row.get("teacher_prompt"),
    ]
    return [normalize_prompt(str(value)) for value in values if value is not None]


def smoke_indices(total: int, start_index: int, end_index: int | None, max_examples: int) -> list[int]:
    start = max(int(start_index), 0)
    end = total if end_index is None else min(int(end_index), total)
    indices = list(range(start, end))
    if max_examples is not None and max_examples > 0:
        indices = indices[: int(max_examples)]
    return indices


def main() -> None:
    args = parse_args()
    manifest_path = args.full_lookup_dir / "prompt_embedding_lookup_manifest.json"
    npy_path = args.full_lookup_dir / "prompt_embedding_lookup_uint8.npy"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    if not npy_path.exists():
        raise FileNotFoundError(npy_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    table = np.load(npy_path)
    entries = list(manifest.get("entries", []))
    mapping = {
        normalize_prompt(str(entry.get("prompt", ""))): index
        for index, entry in enumerate(entries)
        if str(entry.get("prompt", "")).strip()
    }

    selected_prompts: list[str] = []
    for prompt in args.prompt:
        selected_prompts.append(normalize_prompt(prompt))

    examples = load_examples(args.dataset)
    for index in smoke_indices(len(examples), args.start_index, args.end_index, args.max_examples):
        row = examples[index]
        for prompt in prompt_candidates(row):
            if prompt in mapping:
                selected_prompts.append(prompt)
                break
        else:
            raise KeyError(f"No prompt lookup entry found for dataset index {index}: {prompt_candidates(row)}")

    unique_prompts: list[str] = []
    seen: set[str] = set()
    for prompt in selected_prompts:
        if prompt not in seen:
            unique_prompts.append(prompt)
            seen.add(prompt)

    if not unique_prompts:
        raise RuntimeError("No prompts selected.")

    source_indices = [mapping[prompt] for prompt in unique_prompts]
    subset_entries = [entries[index] for index in source_indices]
    subset_table = table[source_indices]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_npy = args.output_dir / "prompt_embedding_lookup_uint8.npy"
    out_manifest = args.output_dir / "prompt_embedding_lookup_manifest.json"
    out_header = args.output_dir / args.header_name
    np.save(out_npy, subset_table)
    write_header(out_header, subset_table, subset_entries)

    subset_manifest = {
        **manifest,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(manifest_path),
        "source_npy": str(npy_path),
        "subset": {
            "dataset": str(args.dataset),
            "max_examples": args.max_examples,
            "start_index": args.start_index,
            "end_index": args.end_index,
            "source_indices": source_indices,
            "prompts": unique_prompts,
        },
        "counts": {
            **manifest.get("counts", {}),
            "entries": len(subset_entries),
        },
        "entries": subset_entries,
        "outputs": {
            "npy": str(out_npy),
            "header": str(out_header),
            "manifest": str(out_manifest),
        },
    }
    out_manifest.write_text(json.dumps(subset_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "outputs": subset_manifest["outputs"],
                "count": len(subset_entries),
                "prompts": unique_prompts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
