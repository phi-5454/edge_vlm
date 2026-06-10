from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
import platform
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from PIL import Image
from tqdm import tqdm

from scripts.cache_smolvlm_tallyqa_teacher import (
    Uint8ImageStore,
    load_examples,
    load_metadata,
)
from scripts.cache_yolo_tallyqa_teacher import prompt_targets


DEFAULT_DATASET = Path("data/tallyqa_cauldron_target_mobilenet224_letterbox")
DEFAULT_MODEL = Path("artifacts/models/ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite")
DEFAULT_OUTPUT = Path("artifacts/teacher_cache/tflite_ssdlite_mobiledet_coco_letterbox.jsonl")
VERY_LOW_LOG_LIKELIHOOD = -1.0e9

COCO80_LABELS = [
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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache a TFLite COCO detector counting baseline for TallyQA."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--prompt", default=None, help="Optional single TallyQA prompt class.")
    parser.add_argument("--prompt-class-names-file", type=Path, default=None)
    parser.add_argument("--include-groups", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument(
        "--class-id-offset",
        type=int,
        default=0,
        help=(
            "Offset applied to raw detection class ids before indexing COCO80 labels. "
            "Use -1 if a model emits one-based COCO ids."
        ),
    )
    parser.add_argument(
        "--delegate",
        choices=["auto", "edgetpu", "none"],
        default="auto",
        help="TFLite delegate to use. EdgeTPU-compiled models require edgetpu.",
    )
    parser.add_argument("--answer-min", type=int, default=0)
    parser.add_argument("--answer-max", type=int, default=15)
    parser.add_argument("--collapse-at", type=int, default=5)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--flush-every", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


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


def validate_args(args: argparse.Namespace) -> None:
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
            except (KeyError, json.JSONDecodeError, TypeError, ValueError):
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


def make_interpreter(model_path: Path, delegate: str) -> Any:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required for TFLite detector baselines. "
            "Install it or run with an environment that provides `tensorflow`."
        ) from exc

    delegates = []
    if delegate in {"auto", "edgetpu"}:
        try:
            delegates.append(tf.lite.experimental.load_delegate("libedgetpu.so.1"))
        except (OSError, ValueError) as exc:
            if delegate == "edgetpu":
                raise RuntimeError(
                    "Could not load libedgetpu.so.1. EdgeTPU-compiled models require the "
                    "EdgeTPU runtime/delegate, or use a non-EdgeTPU TFLite model with "
                    "--delegate none."
                ) from exc
    interpreter = tf.lite.Interpreter(
        model_path=str(model_path),
        experimental_delegates=delegates or None,
    )
    try:
        interpreter.allocate_tensors()
    except RuntimeError as exc:
        if "edgetpu-custom-op" in str(exc):
            raise RuntimeError(
                f"{model_path} is EdgeTPU-compiled and cannot run with the plain CPU "
                "TFLite interpreter. Install the EdgeTPU runtime and run with "
                "--delegate edgetpu, or provide a non-EdgeTPU .tflite model."
            ) from exc
        raise
    return interpreter


def prepare_input(image: Image.Image, input_detail: dict[str, Any]) -> np.ndarray:
    shape = input_detail["shape"]
    height = int(shape[1])
    width = int(shape[2])
    dtype = input_detail["dtype"]
    resized = image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
    array = np.asarray(resized)
    if dtype == np.float32:
        array = array.astype(np.float32) / 255.0
    else:
        array = array.astype(dtype)
    return np.expand_dims(array, axis=0)


def invoke_detector(interpreter: Any, image: Image.Image) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    input_detail = interpreter.get_input_details()[0]
    interpreter.set_tensor(input_detail["index"], prepare_input(image, input_detail))
    interpreter.invoke()
    outputs = [
        interpreter.get_tensor(detail["index"])
        for detail in interpreter.get_output_details()
    ]
    boxes = next((output for output in outputs if output.ndim == 3 and output.shape[-1] == 4), None)
    count = next((output for output in outputs if output.size == 1), None)
    vectors = [output.reshape(-1) for output in outputs if output.ndim == 2]
    if boxes is None or count is None or len(vectors) < 2:
        shapes = [list(output.shape) for output in outputs]
        raise RuntimeError(f"Could not parse TFLite detection outputs with shapes: {shapes}")
    first, second = vectors[:2]
    first_score_like = float(np.nanmax(first)) <= 1.5 if first.size else False
    second_score_like = float(np.nanmax(second)) <= 1.5 if second.size else False
    if first_score_like and not second_score_like:
        scores, classes = first, second
    elif second_score_like and not first_score_like:
        classes, scores = first, second
    else:
        # Most TFLite Detection_PostProcess models emit boxes, classes, scores, count.
        classes, scores = first, second
    valid_count = int(round(float(np.ravel(count)[0])))
    return boxes.reshape(-1, 4), classes, scores, valid_count


def count_detections(
    boxes: np.ndarray,
    classes: np.ndarray,
    scores: np.ndarray,
    valid_count: int,
    targets: list[str],
    score_threshold: float,
    class_id_offset: int,
) -> tuple[int, list[dict[str, Any]]]:
    target_set = set(targets)
    detections: list[dict[str, Any]] = []
    for box, raw_class, score in zip(
        boxes[:valid_count],
        classes[:valid_count],
        scores[:valid_count],
        strict=False,
    ):
        score_float = float(score)
        class_id = int(round(float(raw_class))) + class_id_offset
        if score_float < score_threshold or not 0 <= class_id < len(COCO80_LABELS):
            continue
        label_name = COCO80_LABELS[class_id]
        if label_name not in target_set:
            continue
        ymin, xmin, ymax, xmax = [float(value) for value in box.tolist()]
        detections.append(
            {
                "box_yxyx_normalized": [ymin, xmin, ymax, xmax],
                "score": score_float,
                "label": class_id,
                "raw_label": float(raw_class),
                "label_name": label_name,
            }
        )
    return len(detections), detections


def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def main() -> None:
    args = parse_args()
    if args.force and args.output.exists():
        args.output.unlink()
    if args.output.exists() and not args.force and not args.resume:
        raise FileExistsError(f"{args.output} exists. Pass --force or --resume.")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(args.dataset)
    examples = load_examples(args.dataset)
    selected = selected_indices(examples, args)
    completed = completed_indices(args.output) if args.resume else set()
    indices = [index for index in selected if index not in completed]
    model_name = args.model_name or args.model.stem

    if args.dry_run:
        prompt_counts = Counter(str(examples[index]["student_prompt"]) for index in selected)
        print(
            json.dumps(
                {
                    "dataset": str(args.dataset),
                    "model": str(args.model),
                    "model_name": model_name,
                    "output": str(args.output),
                    "selected_records": len(selected),
                    "remaining_records": len(indices),
                    "selected_prompt_classes": len(prompt_counts),
                    "top_prompt_classes": prompt_counts.most_common(20),
                    "score_threshold": args.score_threshold,
                    "delegate": args.delegate,
                },
                indent=2,
            )
        )
        return

    interpreter = make_interpreter(args.model, args.delegate)
    input_detail = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()
    image_store = Uint8ImageStore(args.dataset, metadata)
    output_mode = "a" if args.resume else "w"
    stats: Counter = Counter()
    by_prompt_target: dict[str, list[str]] = {}
    inference_seconds: list[float] = []

    print(
        json.dumps(
            {
                "event": "tflite_cache_start",
                "dataset": str(args.dataset),
                "model": str(args.model),
                "model_name": model_name,
                "output": str(args.output),
                "selected_records": len(selected),
                "completed_records": len(completed),
                "remaining_records": len(indices),
                "delegate": args.delegate,
                "score_threshold": args.score_threshold,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    with args.output.open(output_mode, encoding="utf-8") as handle:
        progress = tqdm(
            total=len(selected),
            initial=len(completed),
            desc=f"Caching {model_name} TFLite detector counts",
            unit="example",
            dynamic_ncols=True,
            mininterval=1.0,
            file=sys.stdout,
        )
        for dataset_index in indices:
            row = examples[dataset_index]
            image, image_identity = image_store.get(int(row["image_index"]))
            prompt = str(row["student_prompt"])
            targets = prompt_targets(prompt, args.include_groups)
            if targets is None:
                raise RuntimeError(f"Unexpected unsupported prompt in selected data: {prompt}")
            by_prompt_target[prompt] = targets
            inference_start = perf_counter()
            boxes, classes, scores, valid_count = invoke_detector(interpreter, image)
            inference_seconds.append(perf_counter() - inference_start)
            prediction, detections = count_detections(
                boxes,
                classes,
                scores,
                valid_count,
                targets,
                args.score_threshold,
                args.class_id_offset,
            )
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
                    "name": model_name,
                    "path": str(args.model),
                    "family": "tflite-detection-coco",
                    "score_threshold": args.score_threshold,
                    "class_id_offset": args.class_id_offset,
                    "delegate": args.delegate,
                    "input_shape": [int(value) for value in input_detail["shape"].tolist()],
                    "input_dtype": str(input_detail["dtype"]),
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
            progress.set_postfix(
                {
                    "last_ms": f"{inference_seconds[-1] * 1000.0:.1f}",
                    "mean_ms": f"{np.mean(inference_seconds) * 1000.0:.1f}",
                    "pred": prediction,
                }
            )
            progress.update(1)
        progress.close()

    prompt_keys = sorted(
        key.removeprefix("prompt::")
        for key, metric in stats
        if metric == "total" and key.startswith("prompt::")
    )
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "output": str(args.output),
        "model": str(args.model),
        "model_name": model_name,
        "include_groups": args.include_groups,
        "score_threshold": args.score_threshold,
        "class_id_offset": args.class_id_offset,
        "delegate": args.delegate,
        "selected_records": len(selected),
        "written_records_this_invocation": int(stats[("overall", "total")]),
        **accuracy_block(stats, "overall"),
        "by_prompt": {
            prompt: {
                **accuracy_block(stats, f"prompt::{prompt}"),
                "coco_targets": by_prompt_target.get(
                    prompt,
                    prompt_targets(prompt, args.include_groups),
                ),
            }
            for prompt in prompt_keys
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "tensorflow": __import__("tensorflow").__version__,
            "output_shapes": [
                [int(value) for value in detail["shape"].tolist()] for detail in output_details
            ],
        },
        "benchmark": {
            "records": len(inference_seconds),
            "total_inference_seconds": float(sum(inference_seconds)),
            "mean_inference_ms": float(np.mean(inference_seconds) * 1000.0)
            if inference_seconds
            else None,
            "median_inference_ms": float(np.median(inference_seconds) * 1000.0)
            if inference_seconds
            else None,
            "p95_inference_ms": float(np.percentile(inference_seconds, 95) * 1000.0)
            if inference_seconds
            else None,
        },
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
