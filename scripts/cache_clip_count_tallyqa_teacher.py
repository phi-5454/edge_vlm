from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from datasets import load_from_disk
from tqdm.auto import tqdm

from scripts.cache_smolvlm_tallyqa_teacher import load_examples
from scripts.visualize_clip_count_inference import (
    SCALE_FACTOR,
    build_model,
    infer,
    prepare_image,
    resolve_device,
)


DEFAULT_DATASET = Path("data/tallyqa_cauldron_target_mobilenet224_letterbox")
DEFAULT_SOURCE_DATASET = Path("data/the_cauldron/tallyqa")
DEFAULT_CHECKPOINT = Path("external_models/clipcount_pretrained.ckpt")
DEFAULT_CLIP_COUNT_REPO = Path("../CLIP-Count")
DEFAULT_OUTPUT = Path("artifacts/teacher_cache/clip_count_tallyqa_people_128.jsonl")
VERY_LOW_LOG_LIKELIHOOD = -1.0e9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache CLIP-Count predictions for TallyQA.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--source-dataset",
        type=Path,
        default=DEFAULT_SOURCE_DATASET,
        help="Original Cauldron TallyQA dataset used for full-resolution images.",
    )
    parser.add_argument(
        "--image-source",
        choices=["original", "target"],
        default="original",
        help="Use original Cauldron images or target MobileNet-ready images.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--clip-count-repo", type=Path, default=DEFAULT_CLIP_COUNT_REPO)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--prompt", default="people")
    parser.add_argument("--max-examples", type=int, default=128)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--answer-min", type=int, default=0)
    parser.add_argument("--answer-max", type=int, default=15)
    parser.add_argument("--collapse-at", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--flush-every", type=int, default=25)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative")
    if args.end_index is not None and args.end_index < args.start_index:
        raise ValueError("--end-index must be >= --start-index")
    if args.max_examples is not None and args.max_examples < 0:
        raise ValueError("--max-examples must be non-negative")
    if args.answer_min > args.answer_max:
        raise ValueError("--answer-min must be <= --answer-max")


def selected_indices(examples: list[dict[str, Any]], args: argparse.Namespace) -> list[int]:
    validate_args(args)
    stop = len(examples) if args.end_index is None else min(len(examples), args.end_index)
    indices: list[int] = []
    for index in range(args.start_index, stop):
        if str(examples[index]["student_prompt"]) != args.prompt:
            continue
        indices.append(index)
        if args.max_examples is not None and len(indices) >= args.max_examples:
            break
    return indices


def completed_indices(path: Path) -> set[int]:
    if not path.exists():
        return set()
    completed: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            completed.add(int(json.loads(line)["dataset_index"]))
    return completed


def candidate_scores(prediction: int, answer_min: int, answer_max: int) -> list[dict[str, Any]]:
    clamped = max(answer_min, min(answer_max, int(prediction)))
    return [
        {
            "answer": answer,
            "candidate_probability": 1.0 if answer == clamped else 0.0,
            "candidate_log_likelihood": 0.0
            if answer == clamped
            else VERY_LOW_LOG_LIKELIHOOD,
        }
        for answer in range(answer_min, answer_max + 1)
    ]


def update_stats(stats: Counter, answer: int, raw_prediction: float, prediction: int, collapse_at: int) -> None:
    stats["total"] += 1
    stats["absolute_error"] += abs(raw_prediction - answer)
    stats["squared_error"] += (raw_prediction - answer) ** 2
    stats["correct"] += int(prediction == answer)
    stats["within_1"] += int(abs(prediction - answer) <= 1)
    stats["raw_within_1"] += int(abs(raw_prediction - answer) <= 1.0)
    stats["collapsed_correct"] += int(min(prediction, collapse_at) == min(answer, collapse_at))


def main() -> None:
    args = parse_args()
    if args.force and args.output.exists():
        args.output.unlink()
    if args.output.exists() and not args.force and not args.resume:
        raise FileExistsError(f"{args.output} exists. Pass --force or --resume.")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    examples = load_examples(args.dataset)
    indices = selected_indices(examples, args)
    completed = completed_indices(args.output) if args.resume else set()
    remaining = [index for index in indices if index not in completed]
    device = resolve_device(args.device)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "dataset": str(args.dataset),
                    "prompt": args.prompt,
                    "selected_records": len(indices),
                    "remaining_records": len(remaining),
                    "device": str(device),
                    "output": str(args.output),
                },
                indent=2,
            )
        )
        return

    model = build_model(args.clip_count_repo, args.checkpoint, device=device)
    if args.image_source == "original":
        source_dataset = load_from_disk(str(args.source_dataset))
        image_store = None
    else:
        from scripts.cache_smolvlm_tallyqa_teacher import Uint8ImageStore, load_metadata

        source_dataset = None
        image_store = Uint8ImageStore(args.dataset, load_metadata(args.dataset))
    stats: Counter = Counter()
    output_mode = "a" if args.resume else "w"

    with args.output.open(output_mode, encoding="utf-8") as handle:
        progress = tqdm(
            total=len(indices),
            initial=len(completed),
            desc="Caching CLIP-Count TallyQA predictions",
            unit="example",
        )
        for index in remaining:
            row = examples[index]
            if args.image_source == "original":
                source_row_index = int(row["source_row_index"])
                image = source_dataset[source_row_index]["images"][0].convert("RGB")
                image_identity = {
                    "source": "original_cauldron_tallyqa",
                    "source_row_index": source_row_index,
                    "image_slot": 0,
                }
            else:
                image, image_identity = image_store.get(int(row["image_index"]))
            image_tensor = prepare_image(image.convert("RGB"), height=args.height, device=device)
            density = infer(model, image_tensor, prompt=args.prompt, stride=args.stride, device=device)
            raw_prediction = float(density[0].detach().cpu().float().numpy().sum() / SCALE_FACTOR)
            prediction = int(round(raw_prediction))
            answer = int(row["answer"])
            update_stats(stats, answer, raw_prediction, prediction, args.collapse_at)
            record = {
                "schema_version": 1,
                "dataset_index": int(index),
                "example_id": row.get("example_id"),
                "image_id": row.get("image_id"),
                "image_identity": image_identity,
                "student_prompt": str(row["student_prompt"]),
                "teacher_prompt": args.prompt,
                "answer": answer,
                "teacher_model": {
                    "name": "CLIP-Count",
                    "family": "clip-count-density",
                    "checkpoint": str(args.checkpoint),
                    "height": args.height,
                    "stride": args.stride,
                    "scale_factor": SCALE_FACTOR,
                    "image_source": args.image_source,
                },
                "teacher_metrics": {
                    "numeric_answer": {
                        "prediction": prediction,
                        "prediction_text": str(prediction),
                        "raw_prediction": raw_prediction,
                        "absolute_error": abs(raw_prediction - answer),
                        "correct": prediction == answer,
                        "within_1": abs(prediction - answer) <= 1,
                        "raw_within_1": abs(raw_prediction - answer) <= 1.0,
                        "collapsed_correct": min(prediction, args.collapse_at)
                        == min(answer, args.collapse_at),
                    }
                },
                "teacher_logits": {
                    "numeric_answer_candidates": candidate_scores(
                        prediction,
                        args.answer_min,
                        args.answer_max,
                    )
                },
            }
            handle.write(json.dumps(record) + "\n")
            if args.flush_every > 0 and stats["total"] % args.flush_every == 0:
                handle.flush()
            progress.update(1)
        progress.close()

    total = int(stats["total"])
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "output": str(args.output),
        "prompt": args.prompt,
        "image_source": args.image_source,
        "source_dataset": str(args.source_dataset) if args.image_source == "original" else None,
        "selected_records": len(indices),
        "written_records_this_invocation": total,
        "accuracy": stats["correct"] / total if total else None,
        "within_1_accuracy": stats["within_1"] / total if total else None,
        "raw_within_1_accuracy": stats["raw_within_1"] / total if total else None,
        "collapsed_accuracy": stats["collapsed_correct"] / total if total else None,
        "mae": stats["absolute_error"] / total if total else None,
        "rmse": (stats["squared_error"] / total) ** 0.5 if total else None,
        "device": str(device),
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
