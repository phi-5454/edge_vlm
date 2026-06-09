from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


DEFAULT_OUTPUT_DIR = Path("artifacts/reports/final_dataset/trainval_eda")
DEFAULT_SPLIT = Path("artifacts/reports/final_dataset/splits/tallyqa_cauldron_image_holdout_seed42.json")
DEFAULT_MATERIALIZED = Path("artifacts/reports/final_dataset/materialized/manifest.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a manifest for the split-aware train/val EDA bundle.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--materialized", type=Path, default=DEFAULT_MATERIALIZED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split = json.loads(args.split.read_text(encoding="utf-8"))
    source_summary = args.output_dir / "source_eda/summary.json"
    context_manifest = args.output_dir / "context/manifest.json"
    source = json.loads(source_summary.read_text(encoding="utf-8"))
    context = json.loads(context_manifest.read_text(encoding="utf-8"))
    materialized = json.loads(args.materialized.read_text(encoding="utf-8")) if args.materialized.exists() else None
    pending_steps = ["Regenerate teacher caches and teacher-comparison plots on the new materialized datasets."]
    if materialized is None:
        pending_steps.insert(
            0,
            "Materialize trainval and test target datasets with the locked pruned suffix/item files and the split manifest.",
        )
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Split-aware final-dataset EDA. Test images are held out before prompt-pruning "
            "decisions; this EDA uses only the trainval image rows."
        ),
        "split": split,
        "source_eda": {
            "summary": str(source_summary),
            "records": source["dataset_metadata"]["qa_pairs"],
            "included_image_rows": source["dataset_metadata"]["included_image_rows"],
            "figures": str(args.output_dir / "source_eda/figures/combined"),
            "top_200_candidates": str(args.output_dir / "source_eda/template_items_top200_combined.txt"),
            "frontier_suffix_candidates": str(args.output_dir / "source_eda/frontier_suffixes.txt"),
        },
        "context": {
            "manifest": str(context_manifest),
            "figures": str(args.output_dir / "context/figures"),
            "lists": str(args.output_dir / "context/lists"),
            "retention_csv": str(args.output_dir / "context/tables/prompt_class_retention.csv"),
            "retention": context["retention"],
        },
        "locked_pruning_files": {
            "suffixes": str(args.output_dir / "source_eda/frontier_suffixes_pruned.txt"),
            "items": str(args.output_dir / "source_eda/template_items_top200_pruned.txt"),
        },
        "materialized": {
            "manifest": str(args.materialized),
            "summary": materialized,
        }
        if materialized is not None
        else None,
        "pending_steps": pending_steps,
    }
    output = args.output_dir / "manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
