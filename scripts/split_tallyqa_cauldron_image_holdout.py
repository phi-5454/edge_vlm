from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from datasets import load_from_disk

from eda_tallyqa_cauldron import parse_answer


DEFAULT_DATASET = Path("data/the_cauldron/tallyqa")
DEFAULT_OUTPUT_DIR = Path("artifacts/reports/final_dataset/splits")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an explicit image-level 80/20 trainval/test split for Cauldron TallyQA."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trainval-fraction", type=float, default=0.8)
    return parser.parse_args()


def split_for_image_index(image_index: int, seed: int, trainval_fraction: float) -> str:
    digest = hashlib.blake2b(f"{seed}:tallyqa:{image_index}".encode(), digest_size=8).digest()
    value = int.from_bytes(digest, "big") / float(2**64)
    return "trainval" if value < trainval_fraction else "test"


def count_qa_pairs(texts: Any) -> int:
    if not isinstance(texts, list):
        return 0
    count = 0
    for message in texts:
        if not isinstance(message, dict):
            continue
        if isinstance(message.get("user"), str) and parse_answer(message.get("assistant")) is not None:
            count += 1
    return count


def main() -> None:
    args = parse_args()
    if not 0 < args.trainval_fraction < 1:
        raise ValueError("--trainval-fraction must be between 0 and 1.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_from_disk(str(args.dataset)).select_columns(["texts"])

    split_rows = []
    image_counts: Counter[str] = Counter()
    qa_counts: Counter[str] = Counter()
    for image_index, row in enumerate(dataset):
        split = split_for_image_index(image_index, args.seed, args.trainval_fraction)
        qa_pairs = count_qa_pairs(row.get("texts"))
        split_rows.append(
            {
                "source_row_index": image_index,
                "image_id": f"tallyqa:{image_index}",
                "split": split,
                "qa_pairs": qa_pairs,
            }
        )
        image_counts[split] += 1
        qa_counts[split] += qa_pairs

    jsonl_path = args.output_dir / "tallyqa_cauldron_image_holdout_seed42.jsonl"
    if args.seed != 42:
        jsonl_path = args.output_dir / f"tallyqa_cauldron_image_holdout_seed{args.seed}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in split_rows:
            handle.write(json.dumps(row) + "\n")

    index_paths = {}
    for split in ("trainval", "test"):
        path = args.output_dir / f"tallyqa_cauldron_{split}_source_row_indices_seed{args.seed}.txt"
        path.write_text(
            "\n".join(
                str(row["source_row_index"]) for row in split_rows if row["split"] == split
            )
            + "\n",
            encoding="utf-8",
        )
        index_paths[split] = str(path)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "split_jsonl": str(jsonl_path),
        "seed": args.seed,
        "trainval_fraction": args.trainval_fraction,
        "split_unit": "Cauldron image row / source_row_index",
        "hash": "blake2b(seed + ':tallyqa:' + source_row_index), digest_size=8",
        "image_counts": dict(image_counts),
        "qa_pair_counts": dict(qa_counts),
        "image_fractions": {
            split: image_counts[split] / len(split_rows) if split_rows else 0.0
            for split in ("trainval", "test")
        },
        "qa_pair_fractions": {
            split: qa_counts[split] / sum(qa_counts.values()) if sum(qa_counts.values()) else 0.0
            for split in ("trainval", "test")
        },
        "index_files": index_paths,
        "isolation": "No image row appears in both trainval and test.",
    }
    manifest_path = args.output_dir / f"tallyqa_cauldron_image_holdout_seed{args.seed}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
