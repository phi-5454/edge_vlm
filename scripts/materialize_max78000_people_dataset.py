#!/usr/bin/env python3
"""Materialize a MAX78000-friendly TallyQA count view.

By default this preserves the original people-only positive-count view used for
bring-up.  Passing --prompt-class-names-file switches to a general prompt-class
subset with classes 0, 1, 2, 3, 4, 5+.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vlm_micro.student.data import collapse_count, load_tallyqa_rows, split_for_image


DEFAULT_SOURCE = Path("data/tallyqa_cauldron_target_mobilenet224_letterbox")
DEFAULT_OUTPUT = Path("data/max78000_tallyqa_people_count_fold2_56")
PEOPLE_LABELS = ("1", "2", "3", "4", "5+")
COUNT_LABELS = ("0", "1", "2", "3", "4", "5+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--prompt-class-names-file",
        type=Path,
        default=None,
        help="Optional newline-separated prompt classes. Enables general 0/1/2/3/4/5+ mode.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_prompt_classes(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    prompts = {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    if not prompts:
        raise ValueError(f"{path} does not contain any prompt classes.")
    return prompts


def record_for_row(
    row: dict[str, Any],
    seed: int,
    prompt_classes: set[str] | None,
) -> dict[str, Any] | None:
    item = str(row.get("item") or row.get("student_prompt") or "").strip().lower()
    answer = int(row["answer"])
    if prompt_classes is None:
        if item != "people" or answer <= 0:
            return None
        count = collapse_count(answer, collapse_at=5)
        label = count - 1
        count_class = PEOPLE_LABELS[label]
    else:
        if item not in prompt_classes:
            return None
        count = collapse_count(answer, collapse_at=5)
        label = count
        count_class = COUNT_LABELS[label]
    image_id = str(row["image_id"])
    return {
        "example_id": int(row["example_id"]),
        "image_id": image_id,
        "image_index": int(row["image_index"]),
        "student_prompt": item,
        "answer": answer,
        "count_class": count_class,
        "label": label,
        "split": split_for_image(image_id, seed),
    }


def main() -> None:
    args = parse_args()
    if not (args.source / "metadata.json").exists():
        raise FileNotFoundError(args.source / "metadata.json")
    if (args.output / "manifest.jsonl").exists() and not args.force:
        raise FileExistsError(f"{args.output / 'manifest.jsonl'} exists. Re-run with --force.")
    args.output.mkdir(parents=True, exist_ok=True)

    prompt_classes = load_prompt_classes(args.prompt_class_names_file)
    rows = load_tallyqa_rows(args.source)
    records = [
        record
        for row in rows
        if (record := record_for_row(row, args.seed, prompt_classes)) is not None
    ]
    if not records:
        raise RuntimeError("No MAX78000 TallyQA records were found for the requested filter.")

    source_metadata = json.loads((args.source / "metadata.json").read_text(encoding="utf-8"))
    labels = COUNT_LABELS if prompt_classes is not None else PEOPLE_LABELS
    task = (
        f"TallyQA prompt-class subset count, {len(prompt_classes)} prompt classes, labels 0/1/2/3/4/5+"
        if prompt_classes is not None
        else "TallyQA people count, positive counts only, labels 1/2/3/4/5+"
    )
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [
            "uv",
            "run",
            "python",
            "scripts/materialize_max78000_people_dataset.py",
            "--source",
            str(args.source),
            "--output",
            str(args.output),
            "--seed",
            str(args.seed),
        ],
        "source_dataset": str(args.source),
        "prompt_class_names_file": str(args.prompt_class_names_file)
        if args.prompt_class_names_file is not None
        else None,
        "prompt_classes": sorted(prompt_classes) if prompt_classes is not None else ["people"],
        "task": task,
        "adapter_image_size": 112,
        "folded_input": {
            "fold": "2x2 spatial-to-channel",
            "shape": [12, 56, 56],
            "per_channel_bytes": 56 * 56,
            "max78000_per_channel_limit_bytes": 8192,
        },
        "adapter_preprocessing": (
            "ADI dataset adapter resizes the stored square 224x224 RGB tensor to 112x112, "
            "then performs 2x2 spatial-to-channel folding to produce a 12x56x56 tensor. "
            "A direct lossless 224x224 -> 56x56 fold would produce 48 channels, not 12."
        ),
        "seed": args.seed,
        "labels": list(labels),
        "classes": {str(index): label for index, label in enumerate(labels)},
        "split_counts": dict(Counter(record["split"] for record in records)),
        "label_counts": dict(Counter(record["count_class"] for record in records)),
        "prompt_counts": dict(Counter(record["student_prompt"] for record in records)),
        "records": len(records),
        "dropped": {
            "not_selected": len(rows) - len(records),
        },
        "image": {
            **source_metadata["image"],
            "tensor_file": str((args.source / source_metadata["image"]["tensor_file"]).resolve()),
            "source_metadata": str((args.source / "metadata.json").resolve()),
        },
    }

    with (args.output / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {len(records)} records to {args.output / 'manifest.jsonl'}")
    print(f"Split counts: {metadata['split_counts']}")
    print(f"Label counts: {metadata['label_counts']}")


if __name__ == "__main__":
    main()
