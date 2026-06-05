from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vlm_micro.student.data import load_tallyqa_rows, split_for_image


DEFAULT_DATASET = Path("data/tallyqa_cauldron_target_mobilenet224_letterbox")
DEFAULT_OUTPUT_DIR = Path("artifacts/reports/tallyqa_prompt_subsets")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write prompt-class subset files for TallyQA student experiments."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-train-count", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--teacher-accuracy-csv", type=Path, default=None)
    parser.add_argument("--teacher-thresholds", default="0.5,0.6")
    return parser.parse_args()


def write_prompt_file(path: Path, prompts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(prompts) + "\n", encoding="utf-8")


def load_teacher_prompts(path: Path, threshold: float) -> list[str]:
    prompts: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if float(row["accuracy"]) >= threshold:
                prompts.append(str(row["student_prompt"]))
    return sorted(set(prompts))


def main() -> None:
    args = parse_args()
    rows = load_tallyqa_rows(args.dataset)
    train_counts: Counter[str] = Counter()
    total_counts: Counter[str] = Counter()
    for row in rows:
        prompt = str(row["student_prompt"])
        total_counts[prompt] += 1
        if split_for_image(str(row["image_id"]), args.seed) == "train":
            train_counts[prompt] += 1

    sorted_by_train_count = sorted(train_counts, key=lambda prompt: (-train_counts[prompt], prompt))
    all_prompts = sorted(train_counts)
    min_count_prompts = [
        prompt for prompt in sorted_by_train_count if train_counts[prompt] >= args.min_train_count
    ]
    top_k_prompts = sorted_by_train_count[: args.top_k]

    outputs: dict[str, dict[str, Any]] = {}

    def record(name: str, prompts: list[str]) -> None:
        file_name = f"{name}.txt"
        write_prompt_file(args.output_dir / file_name, prompts)
        outputs[name] = {
            "file": str(args.output_dir / file_name),
            "prompt_count": len(prompts),
            "train_examples": sum(train_counts[prompt] for prompt in prompts),
            "total_examples": sum(total_counts[prompt] for prompt in prompts),
            "prompts": prompts,
        }

    record("all", all_prompts)
    record(f"train_count_ge_{args.min_train_count}", min_count_prompts)
    record(f"top_{args.top_k}_train_count", top_k_prompts)
    record("people", ["people"] if "people" in train_counts else [])

    teacher_thresholds = [
        float(value.strip()) for value in args.teacher_thresholds.split(",") if value.strip()
    ]
    if args.teacher_accuracy_csv is not None:
        for threshold in teacher_thresholds:
            prompts = [
                prompt
                for prompt in load_teacher_prompts(args.teacher_accuracy_csv, threshold)
                if prompt in train_counts
            ]
            record(f"teacher_acc_ge_{str(threshold).replace('.', 'p')}", prompts)
            count_filtered = [
                prompt for prompt in prompts if train_counts[prompt] >= args.min_train_count
            ]
            record(
                f"teacher_acc_ge_{str(threshold).replace('.', 'p')}_and_train_count_ge_{args.min_train_count}",
                count_filtered,
            )

    curriculum_schedule = [
        {
            "start_epoch": 1,
            "prompt_class_names_file": str(args.output_dir / f"top_{args.top_k}_train_count.txt"),
            "train_sampling": "natural",
        },
        {
            "start_epoch": 3,
            "prompt_class_names_file": str(
                args.output_dir
                / (
                    f"teacher_acc_ge_0p6_and_train_count_ge_{args.min_train_count}.txt"
                    if args.teacher_accuracy_csv is not None
                    else f"train_count_ge_{args.min_train_count}.txt"
                )
            ),
            "train_sampling": "natural",
        },
        {
            "start_epoch": 6,
            "prompt_class_names_file": str(
                args.output_dir
                / (
                    f"teacher_acc_ge_0p5_and_train_count_ge_{args.min_train_count}.txt"
                    if args.teacher_accuracy_csv is not None
                    else f"train_count_ge_{args.min_train_count}.txt"
                )
            ),
            "train_sampling": "prompt_class_tempered",
            "prompt_class_sampling_temperature": 0.5,
        },
        {
            "start_epoch": 9,
            "prompt_class_names_file": str(args.output_dir / "all.txt"),
            "train_sampling": "prompt_class_tempered",
            "prompt_class_sampling_temperature": 0.5,
        },
    ]
    curriculum_path = args.output_dir / "curriculum_10epoch_default.json"
    curriculum_path.write_text(json.dumps(curriculum_schedule, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "seed": args.seed,
        "min_train_count": args.min_train_count,
        "top_k": args.top_k,
        "teacher_accuracy_csv": str(args.teacher_accuracy_csv)
        if args.teacher_accuracy_csv is not None
        else None,
        "total_prompt_classes": len(all_prompts),
        "curriculum_schedule": str(curriculum_path),
        "outputs": outputs,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote prompt subset manifest: {manifest_path}")
    for name, payload in outputs.items():
        print(
            f"{name}: prompts={payload['prompt_count']} "
            f"train_examples={payload['train_examples']} file={payload['file']}"
        )


if __name__ == "__main__":
    main()
