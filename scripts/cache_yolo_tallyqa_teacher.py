from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
import platform
from pathlib import Path
from typing import Any

os.environ["MPLBACKEND"] = "Agg"

from tqdm.auto import tqdm

from scripts.cache_smolvlm_tallyqa_teacher import (
    Uint8ImageStore,
    load_examples,
    load_metadata,
)


DEFAULT_DATASET = Path("data/tallyqa_cauldron_target_mobilenet224_letterbox")
DEFAULT_OUTPUT = Path("artifacts/teacher_cache/yolo11n_tallyqa_coco_letterbox.jsonl")
VERY_LOW_LOG_LIKELIHOOD = -1.0e9

COCO_ALIASES = {
    "people": "person",
    "persons": "person",
    "chairs": "chair",
    "cars": "car",
    "giraffes": "giraffe",
    "zebras": "zebra",
    "elephants": "elephant",
    "dogs": "dog",
    "cups": "cup",
    "trains": "train",
    "buses": "bus",
    "cats": "cat",
    "bowls": "bowl",
    "birds": "bird",
    "umbrellas": "umbrella",
    "cows": "cow",
    "motorcycles": "motorcycle",
    "boats": "boat",
    "trucks": "truck",
    "benches": "bench",
    "sheep": "sheep",
    "pizzas": "pizza",
    "bottles": "bottle",
    "couches": "couch",
    "clocks": "clock",
    "bears": "bear",
    "bananas": "banana",
    "airplanes": "airplane",
    "books": "book",
    "laptops": "laptop",
    "sinks": "sink",
    "toilets": "toilet",
    "suitcases": "suitcase",
    "beds": "bed",
    "cakes": "cake",
    "planes": "airplane",
    "potted plants": "potted plant",
    "sandwiches": "sandwich",
    "broccolis": "broccoli",
    "tvs": "tv",
    "donuts": "donut",
    "bicycles": "bicycle",
    "vases": "vase",
    "apples": "apple",
    "oranges": "orange",
    "carrots": "carrot",
    "kites": "kite",
    "surfboards": "surfboard",
    "wine glasses": "wine glass",
    "bikes": "bicycle",
    "skateboards": "skateboard",
    "handbags": "handbag",
    "traffic lights": "traffic light",
    "keyboards": "keyboard",
    "backpacks": "backpack",
    "hot dogs": "hot dog",
    "ovens": "oven",
    "tennis rackets": "tennis racket",
    "forks": "fork",
    "knives": "knife",
    "refrigerators": "refrigerator",
    "cell phones": "cell phone",
    "ties": "tie",
    "stop signs": "stop sign",
    "spoons": "spoon",
    "remotes": "remote",
    "fire hydrants": "fire hydrant",
    "microwaves": "microwave",
    "toothbrushes": "toothbrush",
    "snowboards": "snowboard",
    "frisbees": "frisbee",
    "parking meters": "parking meter",
    "glasses": "wine glass",
    "scissors": "scissors",
    "balls": "sports ball",
    "phones": "cell phone",
    "monitors": "tv",
    "tables": "dining table",
    "dining tables": "dining table",
    "animals": [
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
    ],
    "vehicles": [
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
    ],
}

COCO_LABELS = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache a YOLO COCO detector counting baseline for TallyQA."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--prompt", default=None, help="Optional single TallyQA prompt class.")
    parser.add_argument("--prompt-class-names-file", type=Path, default=None)
    parser.add_argument("--include-groups", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--answer-min", type=int, default=0)
    parser.add_argument("--answer-max", type=int, default=15)
    parser.add_argument("--collapse-at", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--flush-every", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_yolo(model_name: str) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Package `ultralytics` is required for YOLO baselines. "
            "Run with `uv run --with ultralytics python scripts/cache_yolo_tallyqa_teacher.py ...` "
            "or install it into the environment with `uv pip install ultralytics`."
        ) from exc
    return YOLO(model_name)


def yolo_device_arg(value: str, require_cuda: bool) -> str:
    if value == "auto":
        import torch

        value = "0" if torch.cuda.is_available() else "cpu"
    if require_cuda and value == "cpu":
        raise RuntimeError("--require-cuda was set, but CUDA is not available/selected.")
    if value == "cuda":
        return "0"
    return value


def prompt_filter(args: argparse.Namespace) -> set[str] | None:
    prompts: set[str] = set()
    if args.prompt:
        prompts.add(args.prompt)
    if args.prompt_class_names_file is not None:
        prompts.update(
            line.strip()
            for line in args.prompt_class_names_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return prompts or None


def prompt_targets(prompt: str, include_groups: bool) -> list[str] | None:
    target = COCO_ALIASES.get(prompt)
    if target is None and prompt in COCO_LABELS:
        target = prompt
    if target is None:
        return None
    if isinstance(target, list):
        return target if include_groups else None
    return [target]


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.shard_count <= 0:
        raise ValueError("--shard-count must be positive")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("--shard-index must be in [0, shard_count)")
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative")
    if args.end_index is not None and args.end_index < args.start_index:
        raise ValueError("--end-index must be greater than or equal to --start-index")
    if args.max_examples is not None and args.max_examples < 0:
        raise ValueError("--max-examples must be non-negative")
    if args.answer_min > args.answer_max:
        raise ValueError("--answer-min must be <= --answer-max")


def selected_indices(examples: list[dict[str, Any]], args: argparse.Namespace) -> list[int]:
    validate_args(args)
    allowed_prompts = prompt_filter(args)
    stop = len(examples) if args.end_index is None else min(len(examples), args.end_index)
    selected: list[int] = []
    for index in range(args.start_index, stop):
        if index % args.shard_count != args.shard_index:
            continue
        prompt = str(examples[index]["student_prompt"])
        if allowed_prompts is not None and prompt not in allowed_prompts:
            continue
        if prompt_targets(prompt, args.include_groups) is None:
            continue
        selected.append(index)
        if args.max_examples is not None and len(selected) >= args.max_examples:
            break
    return selected


def completed_indices(path: Path) -> set[int]:
    if not path.exists():
        return set()
    completed: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                completed.add(int(json.loads(line)["dataset_index"]))
            except json.JSONDecodeError:
                continue
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


def update_stats(stats: Counter, prompt: str, answer: int, prediction: int, collapse_at: int) -> None:
    for key in ("overall", f"prompt::{prompt}"):
        stats[(key, "total")] += 1
        stats[(key, "correct")] += int(prediction == answer)
        stats[(key, "within_1")] += int(abs(prediction - answer) <= 1)
        stats[(key, "collapsed_correct")] += int(
            min(prediction, collapse_at) == min(answer, collapse_at)
        )


def accuracy_block(stats: Counter, key: str) -> dict[str, Any]:
    total = int(stats[(key, "total")])
    return {
        "records": total,
        "accuracy": stats[(key, "correct")] / total if total else None,
        "within_1_accuracy": stats[(key, "within_1")] / total if total else None,
        "collapsed_accuracy": stats[(key, "collapsed_correct")] / total if total else None,
    }


def count_result(result: Any, target_names: list[str], names: dict[int, str], threshold: float) -> tuple[int, list[dict[str, Any]]]:
    boxes = result.boxes
    if boxes is None:
        return 0, []
    target_set = set(target_names)
    detections: list[dict[str, Any]] = []
    for xyxy, conf, cls in zip(boxes.xyxy.cpu(), boxes.conf.cpu(), boxes.cls.cpu(), strict=True):
        class_id = int(cls.item())
        label_name = names[class_id]
        score = float(conf.item())
        if label_name not in target_set or score < threshold:
            continue
        detections.append(
            {
                "box_xyxy": [float(value) for value in xyxy.tolist()],
                "score": score,
                "label": class_id,
                "label_name": label_name,
            }
        )
    return len(detections), detections


def main() -> None:
    args = parse_args()
    if args.force and args.output.exists():
        args.output.unlink()
    if args.output.exists() and not args.force and not args.resume:
        raise FileExistsError(f"{args.output} exists. Pass --force or --resume.")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(args.dataset)
    examples = load_examples(args.dataset)
    all_selected = selected_indices(examples, args)
    completed = completed_indices(args.output) if args.resume else set()
    indices = [index for index in all_selected if index not in completed]
    device = yolo_device_arg(args.device, args.require_cuda)

    if args.dry_run:
        prompt_counts = Counter(str(examples[index]["student_prompt"]) for index in all_selected)
        print(
            json.dumps(
                {
                    "dataset": str(args.dataset),
                    "selected_records": len(all_selected),
                    "remaining_records": len(indices),
                    "selected_prompt_classes": len(prompt_counts),
                    "top_prompt_classes": prompt_counts.most_common(20),
                    "model": args.model,
                    "score_threshold": args.score_threshold,
                    "iou_threshold": args.iou_threshold,
                    "imgsz": args.imgsz,
                    "device": device,
                    "output": str(args.output),
                },
                indent=2,
            )
        )
        return

    model = load_yolo(args.model)
    names = {int(key): str(value) for key, value in model.names.items()}
    image_store = Uint8ImageStore(args.dataset, metadata)
    output_mode = "a" if args.resume else "w"
    stats: Counter = Counter()
    by_prompt_target: dict[str, list[str]] = {}

    with args.output.open(output_mode, encoding="utf-8") as handle:
        progress = tqdm(
            total=len(all_selected),
            initial=len(completed),
            desc="Caching YOLO detector counts",
            unit="example",
        )
        for start in range(0, len(indices), args.batch_size):
            batch_indices = indices[start : start + args.batch_size]
            rows = [examples[index] for index in batch_indices]
            images_and_identities = [image_store.get(int(row["image_index"])) for row in rows]
            images = [image for image, _identity in images_and_identities]
            results = model.predict(
                images,
                conf=args.score_threshold,
                iou=args.iou_threshold,
                imgsz=args.imgsz,
                device=device,
                verbose=False,
            )
            for dataset_index, row, (_image, image_identity), result in zip(
                batch_indices,
                rows,
                images_and_identities,
                results,
                strict=True,
            ):
                prompt = str(row["student_prompt"])
                targets = prompt_targets(prompt, args.include_groups)
                if targets is None:
                    raise RuntimeError(f"Unexpected unsupported prompt in selected data: {prompt}")
                by_prompt_target[prompt] = targets
                prediction, detections = count_result(result, targets, names, args.score_threshold)
                answer = int(row["answer"])
                update_stats(stats, prompt, answer, prediction, args.collapse_at)
                record = {
                    "schema_version": 1,
                    "dataset_index": int(dataset_index),
                    "example_id": row.get("example_id"),
                    "image_id": row.get("image_id"),
                    "image_identity": image_identity,
                    "student_prompt": prompt,
                    "teacher_prompt": " + ".join(targets),
                    "answer": answer,
                    "teacher_model": {
                        "name": args.model,
                        "family": "ultralytics-yolo-coco",
                        "score_threshold": args.score_threshold,
                        "iou_threshold": args.iou_threshold,
                        "imgsz": args.imgsz,
                        "coco_targets": targets,
                    },
                    "teacher_metrics": {
                        "numeric_answer": {
                            "prediction": int(prediction),
                            "prediction_text": str(prediction),
                            "correct": prediction == answer,
                            "within_1": abs(prediction - answer) <= 1,
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
                    "detections": detections,
                }
                handle.write(json.dumps(record) + "\n")
                if args.flush_every > 0 and stats[("overall", "total")] % args.flush_every == 0:
                    handle.flush()
            progress.update(len(batch_indices))
        progress.close()

    prompt_keys = sorted(key.removeprefix("prompt::") for key, metric in stats if metric == "total" and key.startswith("prompt::"))
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "output": str(args.output),
        "model": args.model,
        "selected_records": len(all_selected),
        "written_records_this_invocation": int(stats[("overall", "total")]),
        "include_groups": args.include_groups,
        "score_threshold": args.score_threshold,
        "iou_threshold": args.iou_threshold,
        "imgsz": args.imgsz,
        **accuracy_block(stats, "overall"),
        "by_prompt": {
            prompt: {
                **accuracy_block(stats, f"prompt::{prompt}"),
                "coco_targets": by_prompt_target.get(prompt, prompt_targets(prompt, args.include_groups)),
            }
            for prompt in prompt_keys
        },
        "runtime": {
            "python": platform.python_version(),
            "ultralytics": __import__("ultralytics").__version__,
            "device": device,
        },
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
