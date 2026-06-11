from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
from time import perf_counter
from typing import Any

os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from tqdm import tqdm

from scripts.cache_smolvlm_tallyqa_teacher import (
    Uint8ImageStore,
    load_examples,
    load_metadata,
)


DEFAULT_DATASET = Path("data/tallyqa_cauldron_target_mobilenet224_letterbox")
DEFAULT_MODEL = Path(
    "artifacts/exports/coral/skeletons/"
    "prompt_patch_mlp_static_prompt_minimalistic_compile_probe_docker/model_int8.tflite"
)
DEFAULT_OUTPUT = Path(
    "artifacts/teacher_cache/"
    "tflite_prompt_patch_mlp_static_prompt_minimalistic_smoke.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache a TFLite count-classifier as a TallyQA teacher cache."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-name", default=None)
    parser.add_argument(
        "--delegate",
        choices=["auto", "edgetpu", "none"],
        default="auto",
        help="Use none for ordinary CPU TFLite models; use edgetpu for compiled models.",
    )
    parser.add_argument(
        "--disable-default-delegates",
        action="store_true",
        help=(
            "Run with the builtin TFLite kernels only. This is useful when XNNPACK "
            "fails to prepare a model during local simulator smoke tests."
        ),
    )
    parser.add_argument("--answer-min", type=int, default=0)
    parser.add_argument("--answer-max", type=int, default=5)
    parser.add_argument(
        "--collapse-at",
        type=int,
        default=5,
        help="Ground-truth answers >= this value are evaluated as this final bucket.",
    )
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--flush-every", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--example-output",
        type=Path,
        default=None,
        help="Optional image/prompt/true/pred/probability grid written after caching.",
    )
    parser.add_argument("--example-count", type=int, default=12)
    parser.add_argument("--example-cols", type=int, default=3)
    parser.add_argument("--example-seed", type=int, default=42)
    return parser.parse_args()


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
    if args.collapse_at < args.answer_min or args.collapse_at > args.answer_max:
        raise ValueError("--collapse-at must be inside [answer-min, answer-max]")
    if args.example_count <= 0:
        raise ValueError("--example-count must be positive")
    if args.example_cols <= 0:
        raise ValueError("--example-cols must be positive")


def selected_indices(examples: list[dict[str, Any]], args: argparse.Namespace) -> list[int]:
    validate_args(args)
    stop = len(examples) if args.end_index is None else min(len(examples), args.end_index)
    selected: list[int] = []
    for index in range(args.start_index, stop):
        if index % args.shard_count != args.shard_index:
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


def make_interpreter(model_path: Path, delegate: str, disable_default_delegates: bool = False) -> Any:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required for TFLite teacher caching. "
            "Install it or run in an environment that provides `tensorflow`."
        ) from exc

    delegates = []
    if delegate in {"auto", "edgetpu"}:
        try:
            delegates.append(tf.lite.experimental.load_delegate("libedgetpu.so.1"))
        except (OSError, ValueError) as exc:
            if delegate == "edgetpu":
                raise RuntimeError(
                    "Could not load libedgetpu.so.1. Use --delegate none with a non-EdgeTPU "
                    "TFLite model, or install the EdgeTPU runtime for compiled models."
                ) from exc
    interpreter_kwargs: dict[str, Any] = {
        "model_path": str(model_path),
        "experimental_delegates": delegates or None,
    }
    if disable_default_delegates:
        interpreter_kwargs["experimental_op_resolver_type"] = (
            tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        )
        interpreter_kwargs["experimental_delegates"] = delegates or []
    interpreter = tf.lite.Interpreter(**interpreter_kwargs)
    try:
        interpreter.allocate_tensors()
    except RuntimeError as exc:
        if "edgetpu-custom-op" in str(exc):
            raise RuntimeError(
                f"{model_path} is EdgeTPU-compiled and cannot run with the plain CPU "
                "TFLite interpreter. Run with --delegate edgetpu, or provide the "
                "non-compiled int8 .tflite model with --delegate none."
            ) from exc
        raise
    return interpreter


def chw_image_to_hwc_uint8(image: Any) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def quantize_from_preprocessed_float(array: np.ndarray, detail: dict[str, Any]) -> np.ndarray:
    scale, zero_point = detail["quantization"]
    if not scale:
        raise ValueError("Quantized input has no quantization scale in TFLite metadata.")
    quantized = np.rint(array / float(scale) + int(zero_point))
    dtype = detail["dtype"]
    info = np.iinfo(dtype)
    return np.clip(quantized, info.min, info.max).astype(dtype)


def prepare_input(image: Any, input_detail: dict[str, Any]) -> np.ndarray:
    shape = [int(dim) for dim in input_detail["shape"].tolist()]
    if len(shape) != 4 or shape[0] != 1 or shape[-1] != 3:
        raise ValueError(f"Expected single RGB image input [1,H,W,3], got {shape}")
    height, width = shape[1], shape[2]
    resized = image.convert("RGB").resize((width, height))
    array = chw_image_to_hwc_uint8(resized)
    dtype = input_detail["dtype"]
    if dtype == np.uint8:
        prepared = array.astype(np.uint8)
    elif dtype == np.int8:
        prepared = quantize_from_preprocessed_float(array.astype(np.float32) / 127.5 - 1.0, input_detail)
    elif dtype == np.float32:
        prepared = array.astype(np.float32) / 127.5 - 1.0
    else:
        raise ValueError(f"Unsupported TFLite image input dtype: {dtype}")
    return np.expand_dims(prepared, axis=0)


def prepare_scalar_input(input_detail: dict[str, Any], token_id: int = 0) -> np.ndarray:
    shape = [int(dim) for dim in input_detail["shape"].tolist()]
    dtype = input_detail["dtype"]
    if dtype in {np.int32, np.int64}:
        return np.full(shape, token_id, dtype=dtype)
    if dtype in {np.uint8, np.int8}:
        scale, zero_point = input_detail["quantization"]
        value = token_id if not scale else int(round(token_id / float(scale) + int(zero_point)))
        info = np.iinfo(dtype)
        return np.full(shape, np.clip(value, info.min, info.max), dtype=dtype)
    return np.zeros(shape, dtype=dtype)


def dequantize_output(output: np.ndarray, detail: dict[str, Any]) -> np.ndarray:
    if output.dtype == np.float32:
        return output.astype(np.float32)
    scale, zero_point = detail["quantization"]
    if scale:
        return (output.astype(np.float32) - int(zero_point)) * float(scale)
    return output.astype(np.float32)


def invoke_classifier(interpreter: Any, image: Any) -> tuple[np.ndarray, list[dict[str, Any]]]:
    input_details = interpreter.get_input_details()
    image_inputs = [
        detail
        for detail in input_details
        if len(detail["shape"]) == 4 and int(detail["shape"][-1]) == 3
    ]
    if len(image_inputs) != 1:
        shapes = [[int(value) for value in detail["shape"].tolist()] for detail in input_details]
        raise RuntimeError(f"Expected exactly one image input, got input shapes {shapes}")
    for detail in input_details:
        if detail["index"] == image_inputs[0]["index"]:
            tensor = prepare_input(image, detail)
        else:
            tensor = prepare_scalar_input(detail)
        interpreter.set_tensor(detail["index"], tensor)
    interpreter.invoke()
    output_details = interpreter.get_output_details()
    if len(output_details) != 1:
        shapes = [[int(value) for value in detail["shape"].tolist()] for detail in output_details]
        raise RuntimeError(f"Expected one classifier output, got output shapes {shapes}")
    detail = output_details[0]
    raw = interpreter.get_tensor(detail["index"])
    logits = dequantize_output(raw, detail).reshape(-1)
    return logits, tensor_details(input_details + output_details)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64) - float(np.max(logits))
    exp = np.exp(shifted)
    total = float(exp.sum())
    if total <= 0 or not math.isfinite(total):
        return np.full(logits.shape, 1.0 / logits.size, dtype=np.float64)
    return exp / total


def candidate_scores(
    probabilities: np.ndarray,
    logits: np.ndarray,
    answer_min: int,
    answer_max: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for answer in range(answer_min, answer_max + 1):
        offset = answer - answer_min
        if 0 <= offset < len(probabilities):
            prob = float(probabilities[offset])
            logit = float(logits[offset])
        else:
            prob = 0.0
            logit = float("-inf")
        candidates.append(
            {
                "answer": int(answer),
                "candidate_probability": prob,
                "candidate_log_likelihood": math.log(max(prob, 1.0e-45)),
                "candidate_logit": logit,
            }
        )
    return candidates


def normalized_answer(answer: int, collapse_at: int) -> int:
    return min(int(answer), int(collapse_at))


def numeric_metrics(
    probabilities: np.ndarray,
    prediction: int,
    answer: int,
    answer_min: int,
    collapse_at: int,
) -> dict[str, Any]:
    target = normalized_answer(answer, collapse_at)
    target_offset = target - answer_min
    target_probability = (
        float(probabilities[target_offset])
        if 0 <= target_offset < len(probabilities)
        else 0.0
    )
    correct = int(prediction) == target
    return {
        "prediction": int(prediction),
        "prediction_text": f"{prediction}+" if prediction == collapse_at else str(prediction),
        "target_answer": int(target),
        "target_answer_text": f"{target}+" if target == collapse_at else str(target),
        "correct": bool(correct),
        "within_1": abs(int(prediction) - target) <= 1,
        "collapsed_correct": bool(correct),
        "target_probability": target_probability,
        "nll": -math.log(max(target_probability, 1.0e-45)),
    }


def update_stats(
    stats: Counter,
    prompt: str,
    answer: int,
    prediction: int,
    probabilities: np.ndarray,
    answer_min: int,
    collapse_at: int,
) -> None:
    metrics = numeric_metrics(probabilities, prediction, answer, answer_min, collapse_at)
    for key in ("overall", f"prompt::{prompt}"):
        stats[(key, "total")] += 1
        stats[(key, "correct")] += int(metrics["correct"])
        stats[(key, "within_1")] += int(metrics["within_1"])
        stats[(key, "nll_sum")] += float(metrics["nll"])
        stats[(key, "target_probability_sum")] += float(metrics["target_probability"])


def aggregate_stats(stats: Counter, key: str) -> dict[str, Any]:
    total = int(stats[(key, "total")])
    return {
        "records": total,
        "accuracy": stats[(key, "correct")] / total if total else None,
        "within_1_accuracy": stats[(key, "within_1")] / total if total else None,
        "mean_nll": stats[(key, "nll_sum")] / total if total else None,
        "mean_target_probability": stats[(key, "target_probability_sum")] / total
        if total
        else None,
    }


def tensor_details(details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for detail in details:
        result.append(
            {
                "name": detail.get("name"),
                "index": int(detail["index"]),
                "shape": [int(value) for value in detail["shape"].tolist()],
                "dtype": str(detail["dtype"]),
                "quantization": [
                    float(detail["quantization"][0]),
                    int(detail["quantization"][1]),
                ],
            }
        )
    return result


def write_example_grid(args: argparse.Namespace) -> None:
    if args.example_output is None:
        return
    from scripts.visualize_tallyqa_teacher_logits import (
        image_memmap,
        load_metadata as load_visual_metadata,
        plot_examples,
        selected_cache_rows,
    )

    visual_args = argparse.Namespace(
        dataset=args.dataset,
        cache=args.output,
        output=args.example_output,
        count=args.example_count,
        cols=args.example_cols,
        seed=args.example_seed,
        start_index=None,
        student_prompt=None,
        answer=None,
        incorrect_only=False,
        collapse_at=args.collapse_at,
        answer_min=args.answer_min,
        answer_max=args.answer_max,
    )
    metadata = load_visual_metadata(args.dataset)
    images = image_memmap(args.dataset, metadata)
    rows = selected_cache_rows(visual_args)
    plot_examples(rows, images, visual_args)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "cache": str(args.output),
        "output": str(args.example_output),
        "selected_records": len(rows),
        "dataset_indices": [int(row["dataset_index"]) for row in rows],
        "filters": {
            "seed": args.example_seed,
            "collapse_at": args.collapse_at,
            "answer_min": args.answer_min,
            "answer_max": args.answer_max,
        },
    }
    args.example_output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote example grid: {args.example_output}")


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
    selection_hash = hashlib.sha256(
        json.dumps(selected, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

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
                    "selection_sha256": selection_hash,
                },
                indent=2,
            )
        )
        return

    interpreter = make_interpreter(args.model, args.delegate, args.disable_default_delegates)
    image_store = Uint8ImageStore(args.dataset, metadata)
    output_mode = "a" if args.resume else "w"
    stats: Counter = Counter()
    class_counts: Counter = Counter()
    confusion: dict[int, Counter] = defaultdict(Counter)
    inference_seconds: list[float] = []
    io_details = tensor_details(interpreter.get_input_details() + interpreter.get_output_details())

    print(
        json.dumps(
            {
                "event": "tflite_count_cache_start",
                "dataset": str(args.dataset),
                "model": str(args.model),
                "model_name": model_name,
                "output": str(args.output),
                "selected_records": len(selected),
                "completed_records": len(completed),
                "remaining_records": len(indices),
                "delegate": args.delegate,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    with args.output.open(output_mode, encoding="utf-8") as handle:
        progress = tqdm(
            total=len(selected),
            initial=len(completed),
            desc=f"Caching {model_name} TFLite counts",
            unit="example",
            dynamic_ncols=True,
            mininterval=1.0,
            file=sys.stdout,
        )
        for dataset_index in indices:
            row = examples[dataset_index]
            image, image_identity = image_store.get(int(row["image_index"]))
            inference_start = perf_counter()
            logits, _ = invoke_classifier(interpreter, image)
            inference_seconds.append(perf_counter() - inference_start)
            probabilities = softmax(logits)
            prediction = int(np.argmax(probabilities) + args.answer_min)
            answer = int(row["answer"])
            target = normalized_answer(answer, args.collapse_at)
            prompt = str(row["student_prompt"])
            metrics = numeric_metrics(
                probabilities,
                prediction,
                answer,
                args.answer_min,
                args.collapse_at,
            )
            update_stats(
                stats,
                prompt,
                answer,
                prediction,
                probabilities,
                args.answer_min,
                args.collapse_at,
            )
            class_counts[target] += 1
            confusion[target][prediction] += 1
            candidates = candidate_scores(
                probabilities,
                logits,
                args.answer_min,
                args.answer_max,
            )
            record = {
                "cache_schema_version": 1,
                "dataset_index": int(dataset_index),
                "example_id": row["example_id"],
                "source_subset": row["source_subset"],
                "source": row["source"],
                "source_row_index": int(row["source_row_index"]),
                "qa_index": int(row["qa_index"]),
                "answer": answer,
                "answer_text": row["answer_text"],
                "teacher_prompt": row.get("teacher_prompt_clean", prompt),
                "teacher_prompt_clean": row.get("teacher_prompt_clean", prompt),
                "student_prompt": prompt,
                "item": row["item"],
                "item_class_id": int(row["item_class_id"]),
                "matched_suffix": row["matched_suffix"],
                "image_id": row["image_id"],
                "image_index": int(row["image_index"]),
                "input_identity": {
                    "filtered_dataset": str(args.dataset),
                    "image_source": image_identity["image_source"],
                    "model": str(args.model),
                    "model_name": model_name,
                    "student_prompt_sha256": hashlib.sha256(
                        prompt.encode("utf-8")
                    ).hexdigest(),
                    "image_identity": image_identity,
                },
                "teacher_input": {
                    "image_only_static_prompt": len(interpreter.get_input_details()) == 1,
                },
                "image_preprocessing": {
                    "cached_image_size": list(image.size),
                    "cached_image_mode": image.mode,
                    "image_source": image_identity["image_source"],
                    "image_identity": image_identity,
                },
                "teacher_logits": {
                    "numeric_answer_candidates": candidates,
                    "raw_logits": [float(value) for value in logits.tolist()],
                },
                "teacher_metrics": {
                    "numeric_answer": metrics,
                    "metric_definitions": {
                        "accuracy": "argmax over TFLite count classes equals answer collapsed at --collapse-at",
                        "nll": "negative log-likelihood of collapsed answer under softmax over TFLite logits",
                        "target_probability": "probability assigned to collapsed answer class",
                    },
                },
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
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
        "selected_records": len(selected),
        "records_already_done": len(completed),
        "written_records_this_invocation": int(stats[("overall", "total")]),
        "selection": {
            "first_index": selected[0] if selected else None,
            "last_index": selected[-1] if selected else None,
            "selected_indices_sha256": selection_hash,
        },
        "teacher_metrics": {
            "numeric_answer": aggregate_stats(stats, "overall"),
            "by_student_prompt": {
                prompt: aggregate_stats(stats, f"prompt::{prompt}") for prompt in prompt_keys
            },
            "class_counts": {str(key): int(value) for key, value in sorted(class_counts.items())},
            "confusion": {
                str(true_label): {
                    str(pred_label): int(count)
                    for pred_label, count in sorted(pred_counts.items())
                }
                for true_label, pred_counts in sorted(confusion.items())
            },
        },
        "timing": {
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
        "tflite_io": io_details,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "tensorflow": __import__("tensorflow").__version__,
            "hostname": platform.node(),
            "pid": os.getpid(),
        },
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_example_grid(args)
    print(f"Wrote cache: {args.output}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
