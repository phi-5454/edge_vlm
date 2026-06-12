#!/usr/bin/env python3
"""Compare folded static-prompt Keras and full-int8 TFLite TallyQA metrics."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import tensorflow as tf
from tqdm import tqdm

from scripts.export_tallyqa_keras_static_prompt_coral import (
    build_student_for_weights,
    export_full_int8_tflite,
    folded_prompt_patch_mlp_model,
    inspect_tflite,
    prompt_token_ids,
    static_prompt_model,
)
from scripts.train_tallyqa_keras_student import (
    KerasTallyQAData,
    MulticlassAccumulator,
    collapse_count,
    dequantize_tflite_output,
    git_revision,
    metric_deltas,
    quantize_tflite_input,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="tallyqa_keras_student")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--weights-graph", choices=("auto", "float", "qat"), default="auto")
    parser.add_argument("--skip-mismatch", action="store_true")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--max-examples-per-prompt", type=int, default=None)
    parser.add_argument("--representative-samples", type=int, default=1024)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/reports/tallyqa_keras_static_prompt_quantization"),
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Hydra-style overrides for tallyqa_keras_student. Prefix with -- before overrides.",
    )
    return parser.parse_args()


def clean_overrides(values: list[str]) -> list[str]:
    if values and values[0] == "--":
        return values[1:]
    return values


def compose_cfg(config_name: str, overrides: list[str]) -> DictConfig:
    config_dir = str((Path(__file__).resolve().parents[1] / "conf").resolve())
    with hydra.initialize_config_dir(version_base=None, config_dir=config_dir):
        return hydra.compose(config_name=config_name, overrides=overrides)


def prompt_indices(data: KerasTallyQAData, split: str) -> dict[str, list[int]]:
    by_prompt: dict[str, list[int]] = defaultdict(list)
    for index in data.indices[split]:
        by_prompt[str(data.rows[int(index)]["student_prompt"])].append(int(index))
    return dict(sorted(by_prompt.items()))


def labels_for_indices(data: KerasTallyQAData, indices: list[int]) -> np.ndarray:
    return np.asarray(
        [
            collapse_count(int(data.rows[int(index)]["answer"]), data.collapse_at)
            for index in indices
        ],
        dtype=np.int32,
    )


def images_for_indices(data: KerasTallyQAData, indices: list[int]) -> np.ndarray:
    return np.stack(
        [data.image(int(data.rows[int(index)]["image_index"])) for index in indices],
    ).astype(np.float32)


def evaluate_keras_static_prompt(
    model: tf.keras.Model,
    data: KerasTallyQAData,
    prompt: str,
    indices: list[int],
    batch_size: int,
    accumulator: MulticlassAccumulator,
) -> None:
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        images = images_for_indices(data, batch_indices)
        labels = labels_for_indices(data, batch_indices)
        logits = model(images, training=False).numpy()
        accumulator.update(labels, logits, [prompt] * len(batch_indices))


def map_image_input(input_details: list[dict[str, Any]], images: np.ndarray) -> dict[int, np.ndarray]:
    mapped: dict[int, np.ndarray] = {}
    for detail in input_details:
        shape = list(detail.get("shape", []))
        if len(shape) == 4 or "image" in str(detail.get("name", "")).lower():
            mapped[int(detail["index"])] = images
        else:
            raise ValueError(
                "Static prompt TFLite model should have only image inputs; "
                f"got {detail.get('name')} shape={shape} dtype={detail['dtype']}"
            )
    return mapped


def evaluate_tflite_static_prompt(
    tflite_path: Path,
    data: KerasTallyQAData,
    prompt: str,
    indices: list[int],
    batch_size: int,
    accumulator: MulticlassAccumulator,
) -> None:
    interpreter = tf.lite.Interpreter(
        model_path=str(tflite_path),
        experimental_delegates=[],
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    )
    interpreter.allocate_tensors()
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        images = images_for_indices(data, batch_indices)
        labels = labels_for_indices(data, batch_indices)
        input_details = interpreter.get_input_details()
        resized = False
        for detail in input_details:
            current_shape = list(detail["shape"])
            desired_shape = list(images.shape)
            if current_shape != desired_shape:
                interpreter.resize_tensor_input(int(detail["index"]), desired_shape, strict=False)
                resized = True
        if resized:
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
        for detail in input_details:
            tensor = map_image_input(input_details, images)[int(detail["index"])]
            interpreter.set_tensor(
                int(detail["index"]),
                quantize_tflite_input(tensor, detail),
            )
        interpreter.invoke()
        output_detail = interpreter.get_output_details()[0]
        logits = dequantize_tflite_output(
            interpreter.get_tensor(int(output_detail["index"])),
            output_detail,
        )
        if logits.ndim == 1:
            logits = logits[np.newaxis, :]
        accumulator.update(labels, logits, [prompt] * len(batch_indices))


def main() -> None:
    args = parse_args()
    overrides = clean_overrides(args.overrides)
    if not any(item.startswith("data.require_teacher_cache=") for item in overrides):
        overrides.append("data.require_teacher_cache=false")
    if not any(item.startswith("data.missing_teacher_policy=") for item in overrides):
        overrides.append("data.missing_teacher_policy=keep")
    if not any(item.startswith("keras_model.batch_size=") for item in overrides):
        overrides.append("+keras_model.batch_size=1")
    cfg = compose_cfg(args.config_name, overrides)
    data = KerasTallyQAData(cfg)
    prompt_length = int(data.prompt_token_ids.shape[1])
    student, loaded_graph = build_student_for_weights(
        cfg,
        data,
        prompt_length,
        args.weights,
        args.weights_graph,
        args.skip_mismatch,
    )

    prompts = list(prompt_indices(data, args.split).items())
    if args.max_prompts is not None:
        prompts = prompts[: max(0, int(args.max_prompts))]

    run_name = args.run_name or f"static_prompt_quant_{args.weights.stem}_{args.split}"
    output_dir = args.output_dir / run_name
    tflite_dir = output_dir / "tflite"
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = int(cfg.data.batch_size)
    keras_accumulator = MulticlassAccumulator(int(cfg.model.num_outputs))
    float_tflite_accumulator = MulticlassAccumulator(int(cfg.model.num_outputs))
    tflite_accumulator = MulticlassAccumulator(int(cfg.model.num_outputs))
    prompt_reports: list[dict[str, Any]] = []

    for prompt, indices in tqdm(prompts, desc="static prompts", unit="prompt"):
        if args.max_examples_per_prompt is not None:
            indices = indices[: max(0, int(args.max_examples_per_prompt))]
        if not indices:
            continue
        tokens = prompt_token_ids(data, prompt)
        if str(cfg.keras_model.get("fusion_mode", "normformer")) == "prompt_patch_mlp":
            static_model = folded_prompt_patch_mlp_model(student, cfg, tokens)
        else:
            static_model = static_prompt_model(student, cfg, tokens)
        safe_prompt = "".join(ch if ch.isalnum() else "_" for ch in prompt).strip("_")[:80]
        prompt_tflite_dir = tflite_dir / safe_prompt
        float_path = prompt_tflite_dir / "model_float.tflite"
        int8_path = prompt_tflite_dir / "model_int8.tflite"
        prompt_tflite_dir.mkdir(parents=True, exist_ok=True)
        float_path.write_bytes(tf.lite.TFLiteConverter.from_keras_model(static_model).convert())
        export_full_int8_tflite(static_model, int8_path, data, args.representative_samples, cfg)

        prompt_keras = MulticlassAccumulator(int(cfg.model.num_outputs))
        prompt_float_tflite = MulticlassAccumulator(int(cfg.model.num_outputs))
        prompt_tflite = MulticlassAccumulator(int(cfg.model.num_outputs))
        evaluate_keras_static_prompt(
            static_model,
            data,
            prompt,
            indices,
            batch_size,
            prompt_keras,
        )
        evaluate_tflite_static_prompt(
            float_path,
            data,
            prompt,
            indices,
            batch_size,
            prompt_float_tflite,
        )
        evaluate_tflite_static_prompt(
            int8_path,
            data,
            prompt,
            indices,
            batch_size,
            prompt_tflite,
        )
        evaluate_keras_static_prompt(
            static_model,
            data,
            prompt,
            indices,
            batch_size,
            keras_accumulator,
        )
        evaluate_tflite_static_prompt(
            float_path,
            data,
            prompt,
            indices,
            batch_size,
            float_tflite_accumulator,
        )
        evaluate_tflite_static_prompt(
            int8_path,
            data,
            prompt,
            indices,
            batch_size,
            tflite_accumulator,
        )
        prompt_reports.append(
            {
                "prompt": prompt,
                "examples": len(indices),
                "float_tflite": str(float_path),
                "int8_tflite": str(int8_path),
                "int8_inspection": inspect_tflite(int8_path),
                "keras_static_metrics": prompt_keras.metrics(),
                "float_tflite_metrics": prompt_float_tflite.metrics(),
                "int8_tflite_metrics": prompt_tflite.metrics(),
                "float_tflite_minus_keras": metric_deltas(
                    prompt_keras.metrics(),
                    prompt_float_tflite.metrics(),
                ),
                "int8_tflite_minus_keras": metric_deltas(
                    prompt_keras.metrics(),
                    prompt_tflite.metrics(),
                ),
                "int8_tflite_minus_float_tflite": metric_deltas(
                    prompt_float_tflite.metrics(),
                    prompt_tflite.metrics(),
                ),
            }
        )

    keras_metrics = keras_accumulator.metrics()
    float_tflite_metrics = float_tflite_accumulator.metrics()
    tflite_metrics = tflite_accumulator.metrics()
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
        "run_name": run_name,
        "weights": str(args.weights),
        "weights_graph": loaded_graph,
        "config_name": args.config_name,
        "overrides": overrides,
        "split": args.split,
        "representative_samples": int(args.representative_samples),
        "max_prompts": args.max_prompts,
        "max_examples_per_prompt": args.max_examples_per_prompt,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "keras_static_metrics": keras_metrics,
        "float_tflite_metrics": float_tflite_metrics,
        "int8_tflite_metrics": tflite_metrics,
        "float_tflite_minus_keras": metric_deltas(keras_metrics, float_tflite_metrics),
        "int8_tflite_minus_keras": metric_deltas(keras_metrics, tflite_metrics),
        "int8_tflite_minus_float_tflite": metric_deltas(float_tflite_metrics, tflite_metrics),
        "prompts": prompt_reports,
    }
    report_path = output_dir / "static_prompt_quantization_metrics.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(
        {
            "report": str(report_path),
            "keras_static_metrics": keras_metrics,
            "float_tflite_metrics": float_tflite_metrics,
            "int8_tflite_metrics": tflite_metrics,
            "float_tflite_minus_keras": report["float_tflite_minus_keras"],
            "int8_tflite_minus_keras": report["int8_tflite_minus_keras"],
            "int8_tflite_minus_float_tflite": report["int8_tflite_minus_float_tflite"],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
