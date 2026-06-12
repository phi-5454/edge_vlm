#!/usr/bin/env python3
"""Evaluate one TallyQA TFLite model with image and raw prompt-embedding inputs.

The dynamic token-id model includes runtime prompt embedding, pooling, and
normalization ops that full-integer TFLite quantizes without equivalent QAT
coverage. This script keeps a single deployable model, but moves prompt token
lookup and pooling outside TFLite: the exported graph accepts the image plus a
precomputed raw SmolVLM prompt embedding vector.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import tensorflow as tf
from tqdm import tqdm

from scripts.export_tallyqa_keras_static_prompt_coral import (
    build_student_for_weights,
    clean_overrides,
    compose_cfg,
    inspect_tflite,
    layer_by_name,
)
from scripts.train_tallyqa_keras_student import (
    KerasTallyQAData,
    MulticlassAccumulator,
    TilePromptQueryToFeatureMap,
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
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--representative-samples", type=int, default=1024)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/reports/tallyqa_prompt_query_quantization"),
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Hydra-style overrides for tallyqa_keras_student. Prefix with -- before overrides.",
    )
    return parser.parse_args()


def pooled_prompt_embeddings_for_class_ids(
    data: KerasTallyQAData,
    class_ids: np.ndarray,
) -> np.ndarray:
    compact_rows = np.concatenate(
        [
            np.zeros((1, data.embedding_rows.shape[1]), dtype=np.float32),
            data.embedding_rows.astype(np.float32),
        ],
        axis=0,
    )
    class_ids = class_ids.astype(np.int32)
    token_ids = data.prompt_token_ids[class_ids].astype(np.int32)
    mask = data.prompt_attention_mask[class_ids].astype(np.float32)
    embedded = compact_rows[token_ids]
    denom = np.maximum(mask.sum(axis=1, keepdims=True), 1.0)
    return (embedded * mask[..., np.newaxis]).sum(axis=1) / denom


def projected_prompt_embedding(
    prompt_embedding: tf.Tensor,
    student: tf.keras.Model,
    cfg: DictConfig,
) -> tf.Tensor:
    dense_weights = trainable_layer_weights(
        student,
        "prompt_projection_dense",
        "quant_prompt_projection_dense",
        count=2,
    )
    query = tf.keras.layers.Dense(
        int(dense_weights[0].shape[-1]),
        activation=str(cfg.keras_model.get("activation", "gelu")),
        name="prompt_projection_dense",
    )(prompt_embedding)
    query = tf.keras.layers.LayerNormalization(
        epsilon=1e-5,
        name="prompt_projection_norm",
    )(query)
    return query


def trainable_layer_weights(
    model: tf.keras.Model,
    *names: str,
    count: int,
) -> list[np.ndarray]:
    layer = layer_by_name(model, *names)
    weights = layer.get_weights()
    if len(weights) < count:
        inner = getattr(layer, "layer", None)
        if inner is not None:
            weights = inner.get_weights()
    if len(weights) < count:
        raise ValueError(
            f"Layer {layer.name!r} has {len(weights)} weights, expected at least {count}."
        )
    return [np.asarray(weight) for weight in weights[:count]]


def build_raw_prompt_embedding_student(student: tf.keras.Model, cfg: DictConfig) -> tf.keras.Model:
    if str(cfg.keras_model.get("fusion_mode", "normformer")) != "prompt_patch_mlp":
        raise ValueError(
            "Raw prompt-embedding export is currently implemented for prompt_patch_mlp only."
        )

    concat = layer_by_name(student, "prompt_patch_concat", "quant_prompt_patch_concat")
    image_features_tensor = concat.input[0] if isinstance(concat.input, list) else concat.input
    image_backbone = tf.keras.Model(
        student.get_layer("images").input,
        image_features_tensor,
        name=f"{student.name}_image_backbone_for_raw_prompt_embedding",
    )

    fusion_dim = int(cfg.keras_model.get("fusion_dim", cfg.model.fusion_dim))
    embedding_dim = int(student.get_layer("compact_prompt_embedding").output_dim)
    batch_size = cfg.keras_model.get("batch_size", None)
    batch_size = None if batch_size is None else int(batch_size)
    images = tf.keras.Input(
        shape=(int(cfg.keras_model.image_size), int(cfg.keras_model.image_size), 3),
        batch_size=batch_size,
        dtype=tf.float32,
        name="images",
    )
    prompt_embedding = tf.keras.Input(
        shape=(embedding_dim,),
        batch_size=batch_size,
        dtype=tf.float32,
        name="prompt_embedding",
    )

    image_features = image_backbone(images)
    prompt_query = projected_prompt_embedding(prompt_embedding, student, cfg)
    query_map = tf.keras.layers.Lambda(
        lambda value: tf.reshape(value, (-1, 1, 1, fusion_dim)),
        name="prompt_query_map",
    )(
        prompt_query
    )
    query_features = TilePromptQueryToFeatureMap(name="prompt_query_tile")(
        [query_map, image_features]
    )
    x = tf.keras.layers.Concatenate(axis=-1, name="prompt_patch_concat")(
        [image_features, query_features]
    )
    conv1_weights = trainable_layer_weights(
        student,
        "prompt_patch_conv1x1",
        "quant_prompt_patch_conv1x1",
        count=2,
    )
    conv3_weights = trainable_layer_weights(
        student,
        "prompt_patch_conv3x3",
        "quant_prompt_patch_conv3x3",
        count=2,
    )
    logits_weights = trainable_layer_weights(student, "logits", "quant_logits", count=2)

    x = tf.keras.layers.Conv2D(
        int(conv1_weights[0].shape[-1]),
        kernel_size=1,
        padding="same",
        activation=str(cfg.keras_model.get("activation", "relu")),
        name="prompt_patch_conv1x1",
    )(x)
    x = tf.keras.layers.Conv2D(
        int(conv3_weights[0].shape[-1]),
        kernel_size=3,
        padding="same",
        activation=str(cfg.keras_model.get("activation", "relu")),
        name="prompt_patch_conv3x3",
    )(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="prompt_patch_mean_pool")(x)
    output = tf.keras.layers.Dense(int(cfg.model.num_outputs), name="logits")(x)
    model = tf.keras.Model(
        inputs={"images": images, "prompt_embedding": prompt_embedding},
        outputs=output,
        name=f"{student.name}_raw_prompt_embedding_input",
    )
    model.get_layer("prompt_projection_dense").set_weights(
        trainable_layer_weights(
            student,
            "prompt_projection_dense",
            "quant_prompt_projection_dense",
            count=2,
        )
    )
    model.get_layer("prompt_projection_norm").set_weights(
        student.get_layer("prompt_projection_norm").get_weights()
    )
    model.get_layer("prompt_patch_conv1x1").set_weights(conv1_weights)
    model.get_layer("prompt_patch_conv3x3").set_weights(conv3_weights)
    model.get_layer("logits").set_weights(logits_weights)
    return model


def indices_for_split(data: KerasTallyQAData, split: str, max_examples: int | None) -> list[int]:
    indices = [int(index) for index in data.indices[split]]
    if max_examples is not None:
        indices = indices[: max(0, int(max_examples))]
    return indices


def batch_for_indices(
    data: KerasTallyQAData,
    indices: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    rows = [data.rows[int(index)] for index in indices]
    images = np.stack([data.image(int(row["image_index"])) for row in rows]).astype(np.float32)
    class_ids = np.asarray([int(row["item_class_id"]) for row in rows], dtype=np.int32)
    labels = np.asarray(
        [collapse_count(int(row["answer"]), data.collapse_at) for row in rows],
        dtype=np.int32,
    )
    prompts = [str(row.get("student_prompt")) for row in rows]
    embeddings = pooled_prompt_embeddings_for_class_ids(data, class_ids).astype(np.float32)
    return images, embeddings, labels, prompts


def representative_prompt_embedding_dataset(
    data: KerasTallyQAData,
    max_samples: int,
    cfg: DictConfig,
) -> Iterable[dict[str, np.ndarray]]:
    quant_cfg = cfg.export.quantization
    for token_ids, image in data.representative_examples(
        max_samples=max_samples,
        strategy=str(quant_cfg.get("representative_strategy", "prompt_tempered")),
        prompt_temperature=float(quant_cfg.get("representative_prompt_temperature", 0.5)),
        min_per_prompt=int(quant_cfg.get("representative_min_per_prompt", 4)),
        min_per_output_class=int(quant_cfg.get("representative_min_per_output_class", 32)),
    ):
        token_ids_1d = token_ids.reshape(-1).astype(np.int32)
        mask = (token_ids_1d != 0).astype(np.float32)
        compact_rows = np.concatenate(
            [
                np.zeros((1, data.embedding_rows.shape[1]), dtype=np.float32),
                data.embedding_rows.astype(np.float32),
            ],
            axis=0,
        )
        embedded = compact_rows[token_ids_1d]
        denom = max(float(mask.sum()), 1.0)
        prompt_embedding = ((embedded * mask[:, np.newaxis]).sum(axis=0) / denom)[
            np.newaxis,
            ...,
        ].astype(np.float32)
        yield {
            "images": image.astype(np.float32),
            "prompt_embedding": prompt_embedding,
        }


def export_float_tflite(model: tf.keras.Model, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(tf.lite.TFLiteConverter.from_keras_model(model).convert())


def export_int8_tflite(
    model: tf.keras.Model,
    output: Path,
    data: KerasTallyQAData,
    cfg: DictConfig,
    representative_samples: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_prompt_embedding_dataset(
        data,
        representative_samples,
        cfg,
    )
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    if bool(cfg.export.quantization.get("full_integer", True)):
        input_type = str(cfg.export.quantization.get("inference_input_type", "uint8"))
        output_type = str(cfg.export.quantization.get("inference_output_type", "int8"))
        converter.inference_input_type = getattr(tf, input_type)
        converter.inference_output_type = getattr(tf, output_type)
    output.write_bytes(converter.convert())


def map_prompt_embedding_tflite_inputs(
    input_details: list[dict[str, Any]],
    images: np.ndarray,
    embeddings: np.ndarray,
) -> dict[int, np.ndarray]:
    mapped: dict[int, np.ndarray] = {}
    for detail in input_details:
        name = str(detail.get("name", "")).lower()
        shape = list(detail.get("shape_signature", detail.get("shape", [])))
        rank = len(shape)
        if "prompt" in name or rank == 2:
            mapped[int(detail["index"])] = embeddings
        elif "image" in name or rank == 4:
            mapped[int(detail["index"])] = images
        else:
            raise ValueError(
                "Could not map TFLite input "
                f"{detail.get('name')} shape={detail.get('shape')} dtype={detail['dtype']}."
            )
    return mapped


def evaluate_keras(
    model: tf.keras.Model,
    data: KerasTallyQAData,
    indices: list[int],
    batch_size: int,
) -> MulticlassAccumulator:
    accumulator = MulticlassAccumulator(int(data.cfg.model.num_outputs))
    for start in tqdm(range(0, len(indices), batch_size), desc="keras", unit="batch", leave=False):
        batch_indices = indices[start : start + batch_size]
        images, embeddings, labels, prompts = batch_for_indices(data, batch_indices)
        logits = model({"images": images, "prompt_embedding": embeddings}, training=False).numpy()
        accumulator.update(labels, logits, prompts)
    return accumulator


def evaluate_tflite(
    path: Path,
    data: KerasTallyQAData,
    indices: list[int],
    batch_size: int,
    description: str,
) -> MulticlassAccumulator:
    interpreter = tf.lite.Interpreter(
        model_path=str(path),
        experimental_delegates=[],
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    )
    interpreter.allocate_tensors()
    accumulator = MulticlassAccumulator(int(data.cfg.model.num_outputs))
    for start in tqdm(range(0, len(indices), batch_size), desc=description, unit="batch", leave=False):
        batch_indices = indices[start : start + batch_size]
        images, embeddings, labels, prompts = batch_for_indices(data, batch_indices)
        desired_shapes = {"images": list(images.shape), "prompt_embedding": list(embeddings.shape)}
        input_details = interpreter.get_input_details()
        resized = False
        for detail in input_details:
            value = map_prompt_embedding_tflite_inputs(input_details, images, embeddings)[
                int(detail["index"])
            ]
            if list(detail["shape"]) != list(value.shape):
                interpreter.resize_tensor_input(int(detail["index"]), list(value.shape), strict=False)
                resized = True
        if resized:
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
        mapped = map_prompt_embedding_tflite_inputs(input_details, images, embeddings)
        for detail in input_details:
            interpreter.set_tensor(
                int(detail["index"]),
                quantize_tflite_input(mapped[int(detail["index"])], detail),
            )
        interpreter.invoke()
        output_detail = interpreter.get_output_details()[0]
        logits = dequantize_tflite_output(
            interpreter.get_tensor(int(output_detail["index"])),
            output_detail,
        )
        if logits.ndim == 1:
            logits = logits[np.newaxis, :]
        accumulator.update(labels, logits, prompts)
        _ = desired_shapes
    return accumulator


def main() -> None:
    args = parse_args()
    overrides = clean_overrides(args.overrides)
    if not any(item.startswith("data.require_teacher_cache=") for item in overrides):
        overrides.append("data.require_teacher_cache=false")
    if not any(item.startswith("data.missing_teacher_policy=") for item in overrides):
        overrides.append("data.missing_teacher_policy=keep")
    if not any(
        item.startswith("keras_model.batch_size=") or item.startswith("+keras_model.batch_size=")
        for item in overrides
    ):
        overrides.append("+keras_model.batch_size=1")
    cfg = compose_cfg(args.config_name, overrides)
    data = KerasTallyQAData(cfg)
    student, loaded_graph = build_student_for_weights(
        cfg,
        data,
        int(data.prompt_token_ids.shape[1]),
        args.weights,
        args.weights_graph,
        args.skip_mismatch,
    )
    model = build_raw_prompt_embedding_student(student, cfg)
    indices = indices_for_split(data, args.split, args.max_examples)
    batch_size = int(cfg.data.batch_size)
    run_name = args.run_name or f"prompt_embedding_quant_{args.weights.stem}_{args.split}"
    output_dir = args.output_dir / run_name
    tflite_dir = output_dir / "tflite"
    output_dir.mkdir(parents=True, exist_ok=True)
    float_path = tflite_dir / "model_float.tflite"
    int8_path = tflite_dir / "model_int8.tflite"

    export_float_tflite(model, float_path)
    export_int8_tflite(model, int8_path, data, cfg, args.representative_samples)

    keras_metrics = evaluate_keras(model, data, indices, batch_size).metrics()
    float_tflite_metrics = evaluate_tflite(
        float_path,
        data,
        indices,
        batch_size,
        "float tflite",
    ).metrics()
    int8_tflite_metrics = evaluate_tflite(
        int8_path,
        data,
        indices,
        batch_size,
        "int8 tflite",
    ).metrics()
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
        "run_name": run_name,
        "weights": str(args.weights),
        "weights_graph": loaded_graph,
        "config_name": args.config_name,
        "overrides": overrides,
        "split": args.split,
        "max_examples": args.max_examples,
        "representative_samples": int(args.representative_samples),
        "model_contract": {
            "description": "single TFLite model with image and raw pooled prompt_embedding inputs",
            "prompt_input": "576-d raw SmolVLM prompt embedding pooled outside TFLite",
            "rationale": (
                "Avoids token-id Embedding/Gather/pooling ops inside full-int8 TFLite while "
                "keeping one model for all prompts. The prompt projection is part of the model "
                "and is quantized by the same TFLite conversion path as test-time inference."
            ),
        },
        "outputs": {
            "float_tflite": str(float_path),
            "int8_tflite": str(int8_path),
        },
        "float_tflite_inspection": inspect_tflite(float_path),
        "int8_tflite_inspection": inspect_tflite(int8_path),
        "keras_prompt_embedding_metrics": keras_metrics,
        "float_tflite_metrics": float_tflite_metrics,
        "int8_tflite_metrics": int8_tflite_metrics,
        "float_tflite_minus_keras": metric_deltas(keras_metrics, float_tflite_metrics),
        "int8_tflite_minus_keras": metric_deltas(keras_metrics, int8_tflite_metrics),
        "int8_tflite_minus_float_tflite": metric_deltas(
            float_tflite_metrics,
            int8_tflite_metrics,
        ),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    report_path = output_dir / "prompt_embedding_quantization_metrics.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(
        {
            "report": str(report_path),
            "outputs": report["outputs"],
            "keras_prompt_embedding_metrics": keras_metrics,
            "float_tflite_metrics": float_tflite_metrics,
            "int8_tflite_metrics": int8_tflite_metrics,
            "int8_tflite_minus_keras": report["int8_tflite_minus_keras"],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
