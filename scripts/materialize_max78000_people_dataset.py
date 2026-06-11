#!/usr/bin/env python3
"""Materialize a MAX78000-friendly people-only view of the TallyQA target dataset."""

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
LABELS = ("1", "2", "3", "4", "5+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def record_for_row(row: dict[str, Any], seed: int) -> dict[str, Any] | None:
    item = str(row.get("item") or row.get("student_prompt") or "").strip().lower()
    if item != "people":
        return None
    answer = int(row["answer"])
    if answer <= 0:
        return None
    count = collapse_count(answer, collapse_at=5)
    label = count - 1
    image_id = str(row["image_id"])
    return {
        "example_id": int(row["example_id"]),
        "image_id": image_id,
        "image_index": int(row["image_index"]),
        "answer": answer,
        "count_class": LABELS[label],
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

    rows = load_tallyqa_rows(args.source)
    records = [record for row in rows if (record := record_for_row(row, args.seed)) is not None]
    if not records:
        raise RuntimeError("No people-count records with positive answers were found.")

    source_metadata = json.loads((args.source / "metadata.json").read_text(encoding="utf-8"))
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
        "task": "TallyQA people count, positive counts only, labels 1/2/3/4/5+",
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
        "labels": list(LABELS),
        "classes": {str(index): label for index, label in enumerate(LABELS)},
        "split_counts": dict(Counter(record["split"] for record in records)),
        "label_counts": dict(Counter(record["count_class"] for record in records)),
        "records": len(records),
        "dropped": {
            "non_people_or_zero_count": len(rows) - len(records),
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
