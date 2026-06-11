#!/usr/bin/env python3
"""Materialize a MAX78000-friendly TallyQA count view.

By default this preserves the original people-only positive-count view used for
bring-up.  Passing --prompt-class-names-file switches to a general prompt-class
subset with classes 0, 1, 2, 3, 4, 5+.  Passing --tiered-curriculum-dir writes
one materialized dataset per tier directory under --output.
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
DEFAULT_OUTPUT = Path("data/max78000_tallyqa_count_fold2_56")
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
    parser.add_argument(
        "--tiered-curriculum-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing tier_*/prompt_classes.txt. When set, --output is treated "
            "as a root and one dataset is written per tier subdirectory."
        ),
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


def discover_tier_prompt_files(tiered_curriculum_dir: Path) -> list[tuple[str, Path]]:
    prompt_files = sorted(tiered_curriculum_dir.glob("tier_*/prompt_classes.txt"))
    if not prompt_files:
        raise FileNotFoundError(
            f"No tier_*/prompt_classes.txt files found under {tiered_curriculum_dir}"
        )
    return [(path.parent.name, path) for path in prompt_files]


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


def command_payload(
    args: argparse.Namespace,
    output: Path,
    prompt_classes_file: Path | None,
) -> list[str]:
    command = [
        "uv",
        "run",
        "python",
        "scripts/materialize_max78000_tallyqa_dataset.py",
        "--source",
        str(args.source),
        "--output",
        str(output),
        "--seed",
        str(args.seed),
    ]
    if prompt_classes_file is not None:
        command.extend(["--prompt-class-names-file", str(prompt_classes_file)])
    if args.tiered_curriculum_dir is not None:
        command.extend(["--tiered-curriculum-dir", str(args.tiered_curriculum_dir)])
    if args.force:
        command.append("--force")
    return command


def materialize_dataset(
    *,
    args: argparse.Namespace,
    output: Path,
    rows: list[dict[str, Any]],
    source_metadata: dict[str, Any],
    prompt_classes_file: Path | None,
    tier_name: str | None = None,
    tier_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if (output / "manifest.jsonl").exists() and not args.force:
        raise FileExistsError(f"{output / 'manifest.jsonl'} exists. Re-run with --force.")
    output.mkdir(parents=True, exist_ok=True)

    prompt_classes = load_prompt_classes(prompt_classes_file)
    records = [
        record
        for row in rows
        if (record := record_for_row(row, args.seed, prompt_classes)) is not None
    ]
    if not records:
        raise RuntimeError("No MAX78000 TallyQA records were found for the requested filter.")

    labels = COUNT_LABELS if prompt_classes is not None else PEOPLE_LABELS
    task = (
        f"TallyQA prompt-class subset count, {len(prompt_classes)} prompt classes, labels 0/1/2/3/4/5+"
        if prompt_classes is not None
        else "TallyQA people count, positive counts only, labels 1/2/3/4/5+"
    )
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command_payload(args, output, prompt_classes_file),
        "source_dataset": str(args.source),
        "tier_name": tier_name,
        "tier_metadata": tier_metadata,
        "prompt_class_names_file": str(prompt_classes_file)
        if prompt_classes_file is not None
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

    with (output / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {len(records)} records to {output / 'manifest.jsonl'}")
    print(f"Split counts: {metadata['split_counts']}")
    print(f"Label counts: {metadata['label_counts']}")
    return metadata


def main() -> None:
    args = parse_args()
    if args.prompt_class_names_file is not None and args.tiered_curriculum_dir is not None:
        raise ValueError("Use either --prompt-class-names-file or --tiered-curriculum-dir, not both.")
    if not (args.source / "metadata.json").exists():
        raise FileNotFoundError(args.source / "metadata.json")

    rows = load_tallyqa_rows(args.source)
    source_metadata = json.loads((args.source / "metadata.json").read_text(encoding="utf-8"))

    if args.tiered_curriculum_dir is None:
        materialize_dataset(
            args=args,
            output=args.output,
            rows=rows,
            source_metadata=source_metadata,
            prompt_classes_file=args.prompt_class_names_file,
        )
        return

    manifest_path = args.tiered_curriculum_dir / "manifest.json"
    curriculum_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    )
    tier_definitions = curriculum_manifest.get("tier_definitions", {})
    summary: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command_payload(args, args.output, None),
        "source_dataset": str(args.source),
        "tiered_curriculum_dir": str(args.tiered_curriculum_dir),
        "curriculum_manifest": str(manifest_path) if manifest_path.exists() else None,
        "output_root": str(args.output),
        "tiers": {},
    }
    for tier_name, prompt_file in discover_tier_prompt_files(args.tiered_curriculum_dir):
        print(f"\nMaterializing {tier_name} from {prompt_file}")
        metadata = materialize_dataset(
            args=args,
            output=args.output / tier_name,
            rows=rows,
            source_metadata=source_metadata,
            prompt_classes_file=prompt_file,
            tier_name=tier_name,
            tier_metadata=tier_definitions.get(tier_name),
        )
        summary["tiers"][tier_name] = {
            "output": str(args.output / tier_name),
            "prompt_class_names_file": str(prompt_file),
            "prompt_classes": len(metadata["prompt_classes"]),
            "records": metadata["records"],
            "split_counts": metadata["split_counts"],
            "label_counts": metadata["label_counts"],
        }

    args.output.mkdir(parents=True, exist_ok=True)
    summary_path = args.output / "tiered_materialization_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
