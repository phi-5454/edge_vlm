#!/usr/bin/env python3
"""Train a Keras/TFLite-oriented TallyQA student with teacher distillation."""

from __future__ import annotations

import html
import io
import json
import math
import os
import csv
import re
import subprocess
import textwrap
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import tensorflow as tf
import torch
import wandb
from dotenv import load_dotenv
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm


def absolute_path(value: str) -> Path:
    return Path(to_absolute_path(value))


def export_path_for_run(path_value: str, run_name: str, cfg: DictConfig) -> Path:
    path = absolute_path(path_value)
    if not bool(cfg.export.get("use_run_subdir", True)):
        return path
    return path.parent / run_name / path.name


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def split_for_image(image_id: str, seed: int) -> str:
    import hashlib

    digest = hashlib.blake2b(f"{seed}:{image_id}".encode(), digest_size=8).digest()
    bucket = int.from_bytes(digest, "big") % 100
    if bucket < 70:
        return "train"
    if bucket < 80:
        return "val"
    return "test"


def collapse_count(answer: int, collapse_at: int = 5) -> int:
    return min(int(answer), collapse_at)


def load_tallyqa_rows(dataset_root: Path) -> list[dict[str, Any]]:
    columns = [
        "example_id",
        "source_subset",
        "source_row_index",
        "qa_index",
        "answer",
        "student_prompt",
        "item",
        "item_class_id",
        "image_id",
        "image_index",
    ]
    return pq.read_table(dataset_root / "examples.parquet", columns=columns).to_pylist()


def load_tallyqa_teacher_targets(
    path: Path | None,
    num_classes: int,
    collapse_at: int,
    probability_temperature: float,
) -> dict[int, np.ndarray]:
    if path is None:
        return {}
    targets: dict[int, np.ndarray] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                probabilities = np.zeros((num_classes,), dtype=np.float32)
                for candidate in payload["teacher_logits"]["numeric_answer_candidates"]:
                    class_id = collapse_count(int(candidate["answer"]), collapse_at)
                    probabilities[class_id] += float(candidate["candidate_probability"])
                total = float(probabilities.sum())
                if total <= 0:
                    raise ValueError("teacher candidate probabilities sum to zero")
                if probability_temperature != 1.0:
                    probabilities = np.where(
                        probabilities > 0,
                        probabilities ** (1.0 / probability_temperature),
                        probabilities,
                    )
                    total = float(probabilities.sum())
                targets[int(payload["dataset_index"])] = probabilities / total
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid teacher cache record at {path}:{line_number}") from error
    return targets


def load_prompt_filter(cfg: DictConfig) -> set[str] | None:
    prompts: set[str] = set()
    names = cfg.data.get("prompt_class_names", None)
    if names is not None:
        prompts.update(str(name) for name in names)
    names_file = cfg.data.get("prompt_class_names_file", None)
    if names_file:
        path = absolute_path(str(names_file))
        prompts.update(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return prompts or None


class KerasTallyQAData:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.dataset_root = absolute_path(cfg.paths.dataset_root)
        prompt_payload = torch.load(
            absolute_path(cfg.paths.prompt_embeddings),
            map_location="cpu",
            weights_only=False,
        )
        self.prompt_token_ids = prompt_payload["prompt_token_ids"].long().numpy().astype(np.int32)
        self.prompt_attention_mask = (
            prompt_payload["prompt_attention_mask"].bool().numpy().astype(np.float32)
        )
        self.embedding_rows = prompt_payload["embedding_rows"].float().numpy().astype(np.float32)
        self.rows = load_tallyqa_rows(self.dataset_root)
        self.num_classes = int(cfg.model.num_outputs)
        self.collapse_at = int(cfg.data.collapse_at)
        teacher_cache = absolute_path(cfg.paths.teacher_cache) if cfg.paths.teacher_cache else None
        self.teacher_targets = load_tallyqa_teacher_targets(
            teacher_cache,
            num_classes=self.num_classes,
            collapse_at=self.collapse_at,
            probability_temperature=float(cfg.data.get("teacher_probability_temperature", 1.0)),
        )
        self.full_indices = {"train": [], "val": [], "test": []}
        self.indices = {"train": [], "val": [], "test": []}
        missing_teacher_policy = str(cfg.data.missing_teacher_policy)
        prompt_filter = load_prompt_filter(cfg)
        for index, row in enumerate(self.rows):
            if prompt_filter is not None and str(row["student_prompt"]) not in prompt_filter:
                continue
            split = split_for_image(str(row["image_id"]), int(cfg.seed))
            self.full_indices[split].append(index)
            if missing_teacher_policy == "keep" or index in self.teacher_targets:
                self.indices[split].append(index)
        if cfg.data.get("train_example_limit", None) is not None:
            self.indices["train"] = self.indices["train"][: int(cfg.data.train_example_limit)]
        if missing_teacher_policy == "filter" and not sum(map(len, self.indices.values())):
            raise ValueError("The teacher cache does not contain any usable TallyQA targets.")
        self.train_epoch = 0
        self.train_steps_per_epoch = max(
            1,
            math.ceil(len(self.indices["train"]) / int(cfg.data.batch_size)),
        )

        metadata = json.loads((self.dataset_root / "metadata.json").read_text(encoding="utf-8"))
        image_meta = metadata["image"]
        self.image_shape = tuple(int(dim) for dim in image_meta["shape"])
        self.image_path = self.dataset_root / image_meta.get("tensor_file", "images.uint8.bin")
        self._images: np.memmap | None = None
        self.mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    @property
    def images(self) -> np.memmap:
        if self._images is None:
            self._images = np.memmap(
                self.image_path,
                dtype=np.uint8,
                mode="r",
                shape=self.image_shape,
            )
        return self._images

    def image(self, image_index: int) -> np.ndarray:
        image_preprocessing = str(self.cfg.data.get("image_preprocessing", "imagenet_standard"))
        chw = np.asarray(self.images[int(image_index)], dtype=np.float32)
        if image_preprocessing == "mobilenet_v3_keras":
            return np.transpose(chw, (1, 2, 0)).astype(np.float32)
        if image_preprocessing == "mobilenet_v3_external":
            chw = (chw / 127.5) - 1.0
            return np.transpose(chw, (1, 2, 0)).astype(np.float32)
        if image_preprocessing != "imagenet_standard":
            raise ValueError(
                "data.image_preprocessing must be one of "
                "{'imagenet_standard', 'mobilenet_v3_keras', 'mobilenet_v3_external'}."
            )
        chw = chw / 255.0
        chw = (chw - self.mean) / self.std
        return np.transpose(chw, (1, 2, 0)).astype(np.float32)

    def display_image(self, image_index: int) -> np.ndarray:
        chw = np.asarray(self.images[int(image_index)], dtype=np.float32) / 255.0
        return np.transpose(chw, (1, 2, 0)).clip(0.0, 1.0)

    def split_sizes(self) -> dict[str, int]:
        return {split: len(indices) for split, indices in self.indices.items()}

    def full_split_sizes(self) -> dict[str, int]:
        return {split: len(indices) for split, indices in self.full_indices.items()}

    def label_counts(self, split: str = "train") -> dict[int, int]:
        counts = {class_id: 0 for class_id in range(self.num_classes)}
        for index in self.indices[split]:
            label = collapse_count(int(self.rows[index]["answer"]), self.collapse_at)
            counts[label] += 1
        return counts

    def cache_coverage(self) -> dict[str, float | int | str]:
        covered = sum(index in self.teacher_targets for index in range(len(self.rows)))
        return {
            "policy": str(self.cfg.data.missing_teacher_policy),
            "covered_prompts": covered,
            "total_prompts": len(self.rows),
            "covered_fraction": covered / len(self.rows) if self.rows else 0.0,
            "active_prompts": sum(map(len, self.indices.values())),
        }

    def set_train_epoch(self, epoch: int) -> None:
        self.train_epoch = max(0, int(epoch))

    def set_train_steps_per_epoch(self, steps: int) -> None:
        self.train_steps_per_epoch = max(1, int(steps))

    def prompt_sampling_temperature(self) -> float:
        start_temperature = float(self.cfg.data.get("prompt_class_sampling_temperature", 0.5))
        end_temperature = self.cfg.data.get("prompt_class_sampling_end_temperature", None)
        if end_temperature is None:
            return start_temperature
        end_temperature = float(end_temperature)
        elapsed_steps = self.train_epoch * self.train_steps_per_epoch
        ramp_start_step = int(self.cfg.data.get("prompt_class_sampling_ramp_start_step", 0) or 0)
        if elapsed_steps <= ramp_start_step:
            return start_temperature
        decay_steps = self.cfg.data.get("prompt_class_sampling_decay_steps", None)
        if decay_steps is None:
            max_epochs = int(self.cfg.trainer.get("max_epochs", self.train_epoch + 1))
            decay_steps = max(1, max_epochs * self.train_steps_per_epoch - ramp_start_step)
        progress = min(1.0, (elapsed_steps - ramp_start_step) / max(1, int(decay_steps)))
        return start_temperature + (end_temperature - start_temperature) * progress

    def train_epoch_indices(self) -> list[int]:
        indices = list(self.indices["train"])
        sampling = str(self.cfg.data.get("train_sampling", "natural"))
        rng = np.random.default_rng(int(self.cfg.seed) + self.train_epoch)
        if sampling == "natural":
            if bool(self.cfg.data.get("shuffle_train", True)):
                rng.shuffle(indices)
            return indices
        if sampling != "prompt_class_tempered":
            raise ValueError("data.train_sampling must be 'natural' or 'prompt_class_tempered'.")
        temperature = self.prompt_sampling_temperature()
        by_prompt: dict[str, list[int]] = {}
        for index in indices:
            by_prompt.setdefault(str(self.rows[index]["student_prompt"]), []).append(index)
        prompt_names = sorted(by_prompt)
        counts = np.asarray([len(by_prompt[prompt]) for prompt in prompt_names], dtype=np.float64)
        if temperature <= 0:
            target = counts.copy()
        else:
            target = counts ** (1.0 - temperature)
            target *= counts.sum() / target.sum()
        quotas = np.floor(target).astype(np.int64)
        leftovers = int(counts.sum()) - int(quotas.sum())
        if leftovers > 0:
            fractional = target - quotas
            probabilities = fractional / fractional.sum() if fractional.sum() > 0 else None
            chosen = rng.choice(
                np.arange(len(prompt_names)),
                size=leftovers,
                replace=True,
                p=probabilities,
            )
            for prompt_index in chosen.tolist():
                quotas[prompt_index] += 1
        sampled: list[int] = []
        for prompt, quota in zip(prompt_names, quotas.tolist(), strict=True):
            prompt_indices = by_prompt[prompt]
            if quota <= len(prompt_indices):
                chosen = rng.choice(np.asarray(prompt_indices), size=int(quota), replace=False)
                sampled.extend(int(value) for value in chosen.tolist())
            else:
                sampled.extend(prompt_indices)
                extra = int(quota) - len(prompt_indices)
                chosen = rng.choice(np.asarray(prompt_indices), size=extra, replace=True)
                sampled.extend(int(value) for value in chosen.tolist())
        rng.shuffle(sampled)
        return sampled

    def representative_indices(
        self,
        max_samples: int,
        strategy: str,
        prompt_temperature: float,
        min_per_prompt: int,
        min_per_output_class: int,
    ) -> list[int]:
        train_indices = list(self.indices["train"])
        if max_samples <= 0 or not train_indices:
            return []
        rng = np.random.default_rng(int(self.cfg.seed) + 7919)
        selected: list[int] = []
        selected_set: set[int] = set()

        def add(candidates: list[int], count: int) -> None:
            if count <= 0:
                return
            remaining = [index for index in candidates if index not in selected_set]
            if not remaining:
                return
            take = min(count, len(remaining), max_samples - len(selected))
            if take <= 0:
                return
            chosen = rng.choice(np.asarray(remaining, dtype=np.int64), size=take, replace=False)
            for value in chosen.tolist():
                selected.append(int(value))
                selected_set.add(int(value))

        by_prompt: dict[str, list[int]] = {}
        by_class: dict[int, list[int]] = {class_id: [] for class_id in range(self.num_classes)}
        for index in train_indices:
            row = self.rows[index]
            by_prompt.setdefault(str(row["student_prompt"]), []).append(index)
            by_class[collapse_count(int(row["answer"]), self.collapse_at)].append(index)

        for indices in by_class.values():
            add(indices, min_per_output_class)
        for prompt in sorted(by_prompt):
            add(by_prompt[prompt], min_per_prompt)
        remaining_budget = max_samples - len(selected)
        if remaining_budget <= 0:
            return selected[:max_samples]

        remaining = [index for index in train_indices if index not in selected_set]
        if not remaining:
            return selected[:max_samples]
        if strategy == "natural":
            add(remaining, remaining_budget)
        elif strategy == "prompt_tempered":
            if prompt_temperature <= 0:
                raise ValueError("representative_prompt_temperature must be positive.")
            prompt_names = sorted(by_prompt)
            counts = np.asarray([len(by_prompt[prompt]) for prompt in prompt_names], dtype=np.float64)
            weights = counts ** prompt_temperature
            weights = weights / weights.sum()
            quotas = np.floor(weights * remaining_budget).astype(np.int64)
            leftovers = remaining_budget - int(quotas.sum())
            if leftovers > 0:
                order = rng.choice(
                    np.arange(len(prompt_names)),
                    size=leftovers,
                    replace=True,
                    p=weights,
                )
                for prompt_index in order.tolist():
                    quotas[prompt_index] += 1
            for prompt, quota in zip(prompt_names, quotas.tolist(), strict=True):
                add(by_prompt[prompt], int(quota))
            if len(selected) < max_samples:
                add(remaining, max_samples - len(selected))
        else:
            raise ValueError(
                "export.quantization.representative_strategy must be one of "
                "{'natural', 'prompt_tempered'}."
            )
        return selected[:max_samples]

    def representative_examples(
        self,
        max_samples: int,
        strategy: str,
        prompt_temperature: float,
        min_per_prompt: int,
        min_per_output_class: int,
    ) -> Iterable[tuple[np.ndarray, np.ndarray]]:
        for index in self.representative_indices(
            max_samples,
            strategy,
            prompt_temperature,
            min_per_prompt,
            min_per_output_class,
        ):
            row = self.rows[index]
            item_class_id = int(row["item_class_id"])
            yield (
                self.prompt_token_ids[item_class_id : item_class_id + 1],
                self.image(int(row["image_index"]))[np.newaxis, ...],
            )

    def prediction_examples(self, split: str, max_samples: int) -> dict[str, Any]:
        indices: list[int] = []
        seen_keys: set[tuple[str, str]] = set()
        for index in self.indices[split]:
            row = self.rows[index]
            image_key = str(row.get("image_id", row["image_index"]))
            prompt_key = str(row["student_prompt"])
            key = (image_key, prompt_key)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            indices.append(index)
            if len(indices) >= max(0, int(max_samples)):
                break
        rows = [self.rows[index] for index in indices]
        item_class_ids = np.asarray([int(row["item_class_id"]) for row in rows], dtype=np.int64)
        return {
            "indices": indices,
            "rows": rows,
            "inputs": {
                "token_ids": self.prompt_token_ids[item_class_ids],
                "images": np.stack([self.image(int(row["image_index"])) for row in rows])
                if rows
                else np.empty(
                    (
                        0,
                        int(self.cfg.keras_model.image_size),
                        int(self.cfg.keras_model.image_size),
                        3,
                    ),
                    dtype=np.float32,
                ),
            },
            "display_images": [
                self.display_image(int(row["image_index"]))
                for row in rows
            ],
            "labels": np.asarray(
                [collapse_count(int(row["answer"]), self.collapse_at) for row in rows],
                dtype=np.int32,
            ),
        }

    def _batch_from_indices(
        self,
        batch_indices: list[int],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        rows = [self.rows[index] for index in batch_indices]
        item_class_ids = np.asarray([int(row["item_class_id"]) for row in rows], dtype=np.int64)
        images = np.stack([self.image(int(row["image_index"])) for row in rows])
        labels = np.asarray(
            [collapse_count(int(row["answer"]), self.collapse_at) for row in rows],
            dtype=np.int32,
        )
        teacher_probs = np.stack(
            [
                self.teacher_targets.get(
                    index,
                    np.full((self.num_classes,), np.nan, dtype=np.float32),
                )
                for index in batch_indices
            ]
        ).astype(np.float32)
        return (
            {
                "token_ids": self.prompt_token_ids[item_class_ids],
                "images": images,
            },
            {
                "labels": labels,
                "teacher_probs": teacher_probs,
                "dataset_index": np.asarray(batch_indices, dtype=np.int64),
            },
        )

    def batches(self, split: str) -> Iterable[tuple[dict[str, np.ndarray], dict[str, np.ndarray]]]:
        batch_size = int(self.cfg.data.batch_size)
        indices = self.train_epoch_indices() if split == "train" else list(self.indices[split])
        batch_indices = [
            indices[start : start + batch_size]
            for start in range(0, len(indices), batch_size)
        ]
        workers = max(0, int(self.cfg.data.get("keras_batch_workers", 0) or 0))
        prefetch_batches = max(1, int(self.cfg.data.get("keras_prefetch_batches", 1) or 1))
        if workers <= 0 or prefetch_batches <= 1:
            for batch in batch_indices:
                yield self._batch_from_indices(batch)
            return

        with ThreadPoolExecutor(max_workers=workers) as executor:
            pending: deque[Future[tuple[dict[str, np.ndarray], dict[str, np.ndarray]]]] = deque()
            iterator = iter(batch_indices)
            for _ in range(min(prefetch_batches, len(batch_indices))):
                pending.append(executor.submit(self._batch_from_indices, next(iterator)))
            for batch in iterator:
                future = pending.popleft()
                pending.append(executor.submit(self._batch_from_indices, batch))
                yield future.result()
            while pending:
                yield pending.popleft().result()


def class_weights_from_config(cfg: DictConfig, data: KerasTallyQAData) -> np.ndarray | None:
    explicit_weights = cfg.distillation.get("class_weights", None)
    weight_mode = cfg.distillation.get("class_weight_mode", None)
    if explicit_weights is not None and weight_mode is not None:
        raise ValueError("Use either distillation.class_weights or class_weight_mode, not both.")
    if explicit_weights is not None:
        return np.asarray([float(weight) for weight in explicit_weights], dtype=np.float32)
    if weight_mode is None:
        return None
    if str(weight_mode) != "balanced":
        raise ValueError("distillation.class_weight_mode must be null or 'balanced'.")
    counts = data.label_counts("train")
    total = sum(counts.values())
    num_classes = int(cfg.model.num_outputs)
    if total <= 0 or any(counts[class_id] <= 0 for class_id in range(num_classes)):
        raise ValueError(f"Cannot compute balanced class weights from counts: {counts}")
    return np.asarray(
        [total / (num_classes * counts[class_id]) for class_id in range(num_classes)],
        dtype=np.float32,
    )


def make_data(cfg: DictConfig) -> KerasTallyQAData:
    beta = float(cfg.distillation.beta)
    require_teacher_cache = bool(cfg.data.get("require_teacher_cache", True))
    if beta > 0 and not require_teacher_cache:
        raise ValueError("data.require_teacher_cache=false requires distillation.beta=0.")
    teacher_cache = (
        absolute_path(cfg.paths.teacher_cache)
        if require_teacher_cache and cfg.paths.teacher_cache
        else None
    )
    missing_teacher_policy = str(cfg.data.missing_teacher_policy) if require_teacher_cache else "keep"
    if beta > 0 and missing_teacher_policy == "keep":
        raise ValueError("beta > 0 requires data.missing_teacher_policy=filter.")

    _ = teacher_cache
    return KerasTallyQAData(cfg)


def make_tf_dataset(
    data: KerasTallyQAData,
    split: str,
    cfg: DictConfig,
    prompt_length: int,
) -> tf.data.Dataset:
    image_size = int(cfg.keras_model.image_size)
    num_classes = int(cfg.model.num_outputs)
    signature = (
        {
            "token_ids": tf.TensorSpec(shape=(None, prompt_length), dtype=tf.int32),
            "images": tf.TensorSpec(shape=(None, image_size, image_size, 3), dtype=tf.float32),
        },
        {
            "labels": tf.TensorSpec(shape=(None,), dtype=tf.int32),
            "teacher_probs": tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32),
            "dataset_index": tf.TensorSpec(shape=(None,), dtype=tf.int64),
        },
    )
    return tf.data.Dataset.from_generator(lambda: data.batches(split), output_signature=signature)


def limit_steps(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    numeric = float(value)
    if numeric <= 0:
        return None
    if numeric < 1:
        raise ValueError("Keras trainer limit_*_batches values must be positive integers.")
    return int(numeric)


def inferred_steps(data: KerasTallyQAData, split: str, cfg: DictConfig) -> int:
    limit_key = {
        "train": "limit_train_batches",
        "val": "limit_val_batches",
        "test": "limit_test_batches",
    }[split]
    limited = limit_steps(cfg.trainer.get(limit_key, None))
    if limited is not None:
        return limited
    batch_size = int(cfg.data.batch_size)
    return max(1, (len(data.indices[split]) + batch_size - 1) // batch_size)


def parameter_counts(model: tf.keras.Model) -> dict[str, int]:
    trainable = int(sum(np.prod(weight.shape) for weight in model.trainable_weights))
    non_trainable = int(sum(np.prod(weight.shape) for weight in model.non_trainable_weights))
    return {
        "total": trainable + non_trainable,
        "trainable": trainable,
        "non_trainable": non_trainable,
    }


def iter_leaf_layers(model: tf.keras.Model) -> Iterable[tuple[str, tf.keras.layers.Layer]]:
    def visit(layer: tf.keras.layers.Layer, prefix: str = "") -> Iterable[tuple[str, tf.keras.layers.Layer]]:
        path = f"{prefix}/{layer.name}" if prefix else layer.name
        if isinstance(layer, tf.keras.Model) and layer.layers:
            for child in layer.layers:
                yield from visit(child, path)
            return
        yield path, layer

    for top_layer in model.layers:
        yield from visit(top_layer)


def layer_parameter_rows(model: tf.keras.Model) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def visit(layer: tf.keras.layers.Layer, prefix: str = "") -> None:
        path = f"{prefix}/{layer.name}" if prefix else layer.name
        if isinstance(layer, tf.keras.Model) and layer.layers:
            for child in layer.layers:
                visit(child, path)
            return
        trainable = int(sum(np.prod(weight.shape) for weight in layer.trainable_weights))
        non_trainable = int(sum(np.prod(weight.shape) for weight in layer.non_trainable_weights))
        rows.append(
            {
                "name": path,
                "class": layer.__class__.__name__,
                "output_shape": str(getattr(layer, "output_shape", "unknown")),
                "trainable": trainable,
                "non_trainable": non_trainable,
                "total": trainable + non_trainable,
            }
        )

    for top_layer in model.layers:
        visit(top_layer)
    return rows


def format_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return ""
    widths = {
        column: max(len(column), *(len(str(row.get(column, ""))) for row in rows))
        for column in columns
    }
    header = "  ".join(column.ljust(widths[column]) for column in columns)
    divider = "  ".join("-" * widths[column] for column in columns)
    body = [
        "  ".join(str(row.get(column, "")).ljust(widths[column]) for column in columns)
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def write_model_readable_reports(
    model: tf.keras.Model,
    output_prefix: Path,
) -> dict[str, str]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = output_prefix.with_suffix(".summary.txt")
    layer_csv_path = output_prefix.with_suffix(".layers.csv")
    layer_table_path = output_prefix.with_suffix(".layers.txt")
    parameter_plot_path = output_prefix.with_suffix(".parameter_bars.png")

    summary_buffer = io.StringIO()
    try:
        model.summary(
            print_fn=lambda line: summary_buffer.write(line + "\n"),
            expand_nested=True,
            show_trainable=True,
        )
    except TypeError:
        model.summary(print_fn=lambda line: summary_buffer.write(line + "\n"))
    summary_path.write_text(summary_buffer.getvalue(), encoding="utf-8")

    rows = layer_parameter_rows(model)
    with layer_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["name", "class", "output_shape", "trainable", "non_trainable", "total"],
        )
        writer.writeheader()
        writer.writerows(rows)

    sorted_rows = sorted(rows, key=lambda row: int(row["total"]), reverse=True)
    table_rows = sorted_rows[:80]
    table_text = format_table(
        table_rows,
        ["name", "class", "output_shape", "trainable", "non_trainable", "total"],
    )
    layer_table_path.write_text(
        "Layer parameter table, sorted by parameter count. "
        f"Showing {len(table_rows)} of {len(rows)} leaf layers.\n\n{table_text}\n",
        encoding="utf-8",
    )

    plot_rows = [row for row in sorted_rows if int(row["total"]) > 0][:25]
    if plot_rows:
        fig_height = max(4.0, 0.28 * len(plot_rows))
        fig, ax = plt.subplots(figsize=(10.0, fig_height))
        names = [str(row["name"])[-70:] for row in reversed(plot_rows)]
        totals = [int(row["total"]) for row in reversed(plot_rows)]
        ax.barh(names, totals, color="#4c78a8")
        ax.set_xlabel("Parameters")
        ax.set_title("Largest Leaf Layers by Parameter Count")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(parameter_plot_path, dpi=160)
        plt.close(fig)
    return {
        "summary_txt": str(summary_path),
        "layer_csv": str(layer_csv_path),
        "layer_table_txt": str(layer_table_path),
        "parameter_plot_png": str(parameter_plot_path),
    }


class MulticlassAccumulator:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.prompt_totals = PromptClassAccumulator(num_classes)

    def update(
        self,
        labels: np.ndarray,
        logits: np.ndarray,
        prompts: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        predictions = np.argmax(logits, axis=1)
        for true_label, predicted_label in zip(labels.tolist(), predictions.tolist(), strict=True):
            self.confusion[int(true_label), int(predicted_label)] += 1
        self.prompt_totals.update(labels, predictions, prompts)

    def metrics(self) -> dict[str, float]:
        correct = int(np.trace(self.confusion))
        total = int(self.confusion.sum())
        labels = [index for index in range(self.num_classes) if int(self.confusion[index].sum()) > 0]
        absolute_error = 0
        within_one = 0
        for true_label in range(self.num_classes):
            for predicted_label in range(self.num_classes):
                count = int(self.confusion[true_label, predicted_label])
                absolute_error += abs(predicted_label - true_label) * count
                within_one += int(abs(predicted_label - true_label) <= 1) * count
        class_weighted_accuracy = (
            sum(self.confusion[label, label] / self.confusion[label].sum() for label in labels)
            / len(labels)
            if labels
            else 0.0
        )
        class_weighted_within_one = (
            sum(
                self.confusion[
                    label,
                    max(0, label - 1) : min(self.num_classes, label + 2),
                ].sum()
                / self.confusion[label].sum()
                for label in labels
            )
            / len(labels)
            if labels
            else 0.0
        )
        class_weighted_mae = (
            sum(
                sum(
                    abs(predicted_label - label) * self.confusion[label, predicted_label]
                    for predicted_label in range(self.num_classes)
                )
                / self.confusion[label].sum()
                for label in labels
            )
            / len(labels)
            if labels
            else 0.0
        )
        return {
            "accuracy": correct / total if total else 0.0,
            "within_1_accuracy": within_one / total if total else 0.0,
            "mae": absolute_error / total if total else 0.0,
            "class_weighted_accuracy": float(class_weighted_accuracy),
            "class_weighted_within_1_accuracy": float(class_weighted_within_one),
            "class_weighted_mae": float(class_weighted_mae),
            **self.prompt_totals.metrics(),
        }


class PromptClassAccumulator:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.by_prompt: dict[str, np.ndarray] = {}

    def update(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        prompts: list[str] | tuple[str, ...] | None,
    ) -> None:
        if prompts is None or len(prompts) != len(labels):
            return
        for prompt, true_label, predicted_label in zip(
            prompts,
            labels.tolist(),
            predictions.tolist(),
            strict=True,
        ):
            confusion = self.by_prompt.setdefault(
                str(prompt),
                np.zeros((self.num_classes, self.num_classes), dtype=np.int64),
            )
            confusion[int(true_label), int(predicted_label)] += 1

    @staticmethod
    def _metrics_for_confusion(confusion: np.ndarray) -> dict[str, float]:
        correct = int(np.trace(confusion))
        total = int(confusion.sum())
        labels = [index for index in range(confusion.shape[0]) if int(confusion[index].sum()) > 0]
        absolute_error = 0
        within_one = 0
        for true_label in range(confusion.shape[0]):
            for predicted_label in range(confusion.shape[1]):
                count = int(confusion[true_label, predicted_label])
                absolute_error += abs(predicted_label - true_label) * count
                within_one += int(abs(predicted_label - true_label) <= 1) * count
        class_weighted_accuracy = (
            sum(confusion[label, label] / confusion[label].sum() for label in labels) / len(labels)
            if labels
            else 0.0
        )
        class_weighted_within_one = (
            sum(
                sum(
                    int(abs(predicted_label - label) <= 1) * confusion[label, predicted_label]
                    for predicted_label in range(confusion.shape[1])
                )
                / confusion[label].sum()
                for label in labels
            )
            / len(labels)
            if labels
            else 0.0
        )
        class_weighted_mae = (
            sum(
                sum(
                    abs(predicted_label - label) * confusion[label, predicted_label]
                    for predicted_label in range(confusion.shape[1])
                )
                / confusion[label].sum()
                for label in labels
            )
            / len(labels)
            if labels
            else 0.0
        )
        return {
            "accuracy": correct / total if total else 0.0,
            "within_1_accuracy": within_one / total if total else 0.0,
            "mae": absolute_error / total if total else 0.0,
            "class_weighted_accuracy": float(class_weighted_accuracy),
            "class_weighted_within_1_accuracy": float(class_weighted_within_one),
            "class_weighted_mae": float(class_weighted_mae),
        }

    def metrics(self) -> dict[str, float]:
        prompt_metrics = [
            self._metrics_for_confusion(confusion)
            for confusion in self.by_prompt.values()
            if int(confusion.sum()) > 0
        ]
        if not prompt_metrics:
            return {
                "prompt_class_weighted_accuracy": 0.0,
                "prompt_class_weighted_within_1_accuracy": 0.0,
                "prompt_class_weighted_mae": 0.0,
                "prompt_class_output_weighted_accuracy": 0.0,
                "prompt_class_output_weighted_within_1_accuracy": 0.0,
                "prompt_class_output_weighted_mae": 0.0,
            }
        return {
            "prompt_class_weighted_accuracy": sum(
                metrics["accuracy"] for metrics in prompt_metrics
            )
            / len(prompt_metrics),
            "prompt_class_weighted_within_1_accuracy": sum(
                metrics["within_1_accuracy"] for metrics in prompt_metrics
            )
            / len(prompt_metrics),
            "prompt_class_weighted_mae": sum(metrics["mae"] for metrics in prompt_metrics)
            / len(prompt_metrics),
            "prompt_class_output_weighted_accuracy": sum(
                metrics["class_weighted_accuracy"] for metrics in prompt_metrics
            )
            / len(prompt_metrics),
            "prompt_class_output_weighted_within_1_accuracy": sum(
                metrics["class_weighted_within_1_accuracy"] for metrics in prompt_metrics
            )
            / len(prompt_metrics),
            "prompt_class_output_weighted_mae": sum(
                metrics["class_weighted_mae"] for metrics in prompt_metrics
            )
            / len(prompt_metrics),
        }


def class_labels(num_classes: int) -> list[str]:
    if num_classes == 6:
        return ["0", "1", "2", "3", "4", "5+"]
    return [str(index) for index in range(num_classes)]


def confusion_matrix_figure(stage: str, accumulator: MulticlassAccumulator) -> plt.Figure:
    counts = accumulator.confusion
    row_totals = counts.sum(axis=1, keepdims=True)
    normalized = counts / np.clip(row_totals, a_min=1, a_max=None)
    labels = class_labels(accumulator.num_classes)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    cmap = plt.get_cmap("magma")
    image = ax.imshow(normalized, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"{stage} output confusion matrix")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Row fraction")
    for row in range(counts.shape[0]):
        for col in range(counts.shape[1]):
            value = normalized[row, col]
            red, green, blue, _alpha = cmap(float(value))
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            ax.text(
                col,
                row,
                str(int(counts[row, col])),
                ha="center",
                va="center",
                color="black" if luminance > 0.5 else "white",
                fontsize=8,
            )
    fig.tight_layout()
    return fig


def confusion_matrix_plot(stage: str, accumulator: MulticlassAccumulator) -> wandb.Image:
    fig = confusion_matrix_figure(stage, accumulator)
    payload = wandb.Image(fig)
    plt.close(fig)
    return payload


def save_confusion_matrix_plot(
    stage: str,
    accumulator: MulticlassAccumulator,
    output: Path,
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig = confusion_matrix_figure(stage, accumulator)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def save_prediction_examples_plot(
    stage: str,
    model: tf.keras.Model,
    data: KerasTallyQAData,
    output: Path,
    max_samples: int,
) -> Path | None:
    examples = data.prediction_examples(stage, max_samples)
    rows = examples["rows"]
    if not rows:
        return None
    logits = model.student(examples["inputs"], training=False).numpy()
    probabilities = tf.nn.softmax(logits, axis=1).numpy()
    predictions = np.argmax(probabilities, axis=1)
    labels = examples["labels"]
    class_names = class_labels(data.num_classes)
    height = max(3.2, 2.15 * len(rows))
    fig, axes = plt.subplots(len(rows), 2, figsize=(10.8, height), squeeze=False)
    for row_index, row in enumerate(rows):
        image_ax, prob_ax = axes[row_index]
        image_ax.imshow(examples["display_images"][row_index])
        prompt = str(row["student_prompt"])
        title = (
            f"{stage} idx={examples['indices'][row_index]} "
            f"true={int(labels[row_index])} pred={int(predictions[row_index])} "
            f"prompt={prompt}"
        )
        image_ax.set_title("\n".join(textwrap.wrap(title, width=58)), fontsize=9)
        image_ax.axis("off")

        colors = ["#8a8f98"] * len(class_names)
        colors[int(labels[row_index])] = "#2b8a3e"
        colors[int(predictions[row_index])] = (
            "#2f6fdd" if predictions[row_index] == labels[row_index] else "#c92a2a"
        )
        prob_ax.bar(class_names, probabilities[row_index], color=colors)
        prob_ax.set_ylim(0, 1)
        prob_ax.set_ylabel("p")
        prob_ax.set_xlabel("count")
        prob_ax.set_title("predicted count distribution", fontsize=9)
        for class_index, probability in enumerate(probabilities[row_index]):
            if probability >= 0.05:
                prob_ax.text(
                    class_index,
                    probability + 0.02,
                    f"{probability:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def normalize_activation_map(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    minimum = float(np.nanmin(values))
    maximum = float(np.nanmax(values))
    if not np.isfinite(minimum) or not np.isfinite(maximum) or maximum <= minimum:
        return np.zeros_like(values, dtype=np.float32)
    return (values - minimum) / (maximum - minimum)


def activation_probe_model(student: tf.keras.Model) -> tf.keras.Model | None:
    try:
        cutoff_layers = [
            layer
            for layer in student.layers
            if isinstance(layer, tf.keras.Model) and str(layer.name).endswith("_cutoff")
        ]
        image_feature_output = cutoff_layers[0].output if cutoff_layers else None
        projected_output = student.get_layer("image_token_projection").output
        if image_feature_output is None:
            return None
        return tf.keras.Model(
            student.inputs,
            [image_feature_output, projected_output, student.output],
            name=f"{student.name}_activation_probe",
        )
    except (KeyError, ValueError, IndexError):
        return None


def save_activation_examples_plot(
    stage: str,
    model: tf.keras.Model,
    data: KerasTallyQAData,
    output: Path,
    max_samples: int,
) -> Path | None:
    probe = activation_probe_model(model.student)
    if probe is None:
        return None
    examples = data.prediction_examples(stage, max_samples)
    rows = examples["rows"]
    if not rows:
        return None
    image_features, projected_features, logits = probe(examples["inputs"], training=False)
    image_features = image_features.numpy()
    projected_features = projected_features.numpy()
    logits = logits.numpy()
    probabilities = tf.nn.softmax(logits, axis=1).numpy()
    predictions = np.argmax(probabilities, axis=1)
    labels = examples["labels"]

    class_names = class_labels(data.num_classes)
    height = max(3.2, 2.2 * len(rows))
    fig, axes = plt.subplots(len(rows), 4, figsize=(14.4, height), squeeze=False)
    for row_index, row in enumerate(rows):
        image_ax, activation_ax, projected_ax, prob_ax = axes[row_index]
        image_ax.imshow(examples["display_images"][row_index])
        title = (
            f"{stage} idx={examples['indices'][row_index]} "
            f"true={int(labels[row_index])} pred={int(predictions[row_index])} "
            f"prompt={row['student_prompt']}"
        )
        image_ax.set_title("\n".join(textwrap.wrap(title, width=42)), fontsize=9)
        image_ax.axis("off")

        activation = normalize_activation_map(image_features[row_index].mean(axis=-1))
        projected = normalize_activation_map(projected_features[row_index].mean(axis=-1))
        activation_ax.imshow(activation, cmap="magma")
        activation_ax.set_title("mean image encoder activation", fontsize=9)
        activation_ax.axis("off")
        projected_ax.imshow(projected, cmap="magma")
        projected_ax.set_title("mean projected image tokens", fontsize=9)
        projected_ax.axis("off")

        colors = ["#8a8f98"] * len(class_names)
        colors[int(labels[row_index])] = "#2b8a3e"
        colors[int(predictions[row_index])] = (
            "#2f6fdd" if predictions[row_index] == labels[row_index] else "#c92a2a"
        )
        prob_ax.bar(class_names, probabilities[row_index], color=colors)
        prob_ax.set_ylim(0, 1)
        prob_ax.set_ylabel("p")
        prob_ax.set_xlabel("count")
        prob_ax.set_title("predicted count distribution", fontsize=9)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def safe_artifact_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "artifact"


def log_wandb_artifact(
    name: str,
    artifact_type: str,
    paths: Iterable[Path],
    aliases: list[str] | None = None,
    base_path: Path | None = None,
) -> None:
    if wandb.run is None or getattr(wandb.run, "disabled", False):
        return
    artifact = wandb.Artifact(safe_artifact_name(name), type=artifact_type)
    added = False
    for path in paths:
        if path.exists():
            if base_path is not None:
                try:
                    artifact.add_file(str(path), name=str(path.relative_to(base_path)))
                except ValueError:
                    artifact.add_file(str(path))
            else:
                artifact.add_file(str(path))
            added = True
    if added:
        wandb.log_artifact(artifact, aliases=aliases)


def save_wandb_file(path: Path, *, policy: str = "now") -> None:
    if wandb.run is None:
        return
    if os.environ.get("VLM_MICRO_WANDB_SAVE_FILES", "0").strip().lower() not in {"1", "true", "yes"}:
        return
    wandb.save(str(path), policy=policy)


def evaluate_split_metrics(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    steps: int,
    num_classes: int,
    data: KerasTallyQAData,
    description: str | None = None,
) -> MulticlassAccumulator:
    accumulator = MulticlassAccumulator(num_classes)
    iterator = iter(dataset)
    progress = tqdm(
        range(steps),
        desc=description,
        unit="batch",
        leave=False,
        disable=description is None,
    )
    for _ in progress:
        inputs, targets = next(iterator)
        logits = model.student(inputs, training=False).numpy()
        labels = targets["labels"].numpy()
        dataset_indices = targets["dataset_index"].numpy().tolist()
        prompts = [str(data.rows[int(index)]["student_prompt"]) for index in dataset_indices]
        accumulator.update(labels, logits, prompts)
    return accumulator


def build_tflite_prior_model(
    cfg: DictConfig,
    embedding_rows: np.ndarray,
    prompt_length: int,
) -> tf.keras.Model:
    num_classes = int(cfg.model.num_outputs)
    prompt_dim = int(cfg.keras_model.prompt_dim)
    image_dim = int(cfg.keras_model.image_dim)
    fusion_dim = int(cfg.keras_model.fusion_dim)
    activation = str(cfg.keras_model.activation)

    batch_size = cfg.keras_model.get("batch_size", None)
    batch_size = None if batch_size is None else int(batch_size)
    token_ids = tf.keras.Input(
        shape=(prompt_length,),
        batch_size=batch_size,
        dtype=tf.int32,
        name="token_ids",
    )
    images = tf.keras.Input(
        shape=(int(cfg.keras_model.image_size), int(cfg.keras_model.image_size), 3),
        batch_size=batch_size,
        dtype=tf.float32,
        name="images",
    )

    pad = np.zeros((1, embedding_rows.shape[1]), dtype=np.float32)
    embedding_init = np.concatenate([pad, embedding_rows.astype(np.float32)], axis=0)
    embedding = tf.keras.layers.Embedding(
        input_dim=embedding_init.shape[0],
        output_dim=embedding_init.shape[1],
        embeddings_initializer=tf.keras.initializers.Constant(embedding_init),
        trainable=not bool(cfg.model.freeze_embeddings),
        mask_zero=True,
        name="compact_prompt_embedding",
    )(token_ids)
    query = tf.keras.layers.GlobalAveragePooling1D(name="mean_prompt_embedding")(embedding)
    query = tf.keras.layers.Dense(prompt_dim, activation=activation, name="prompt_projection")(query)

    x = images
    for block_index, channels in enumerate(cfg.keras_model.conv_channels):
        channels = int(channels)
        x = tf.keras.layers.Conv2D(
            channels,
            kernel_size=3,
            strides=2,
            padding="same",
            use_bias=not bool(cfg.keras_model.use_batch_norm),
            name=f"image_conv_{block_index}",
        )(x)
        if bool(cfg.keras_model.use_batch_norm):
            x = tf.keras.layers.BatchNormalization(name=f"image_bn_{block_index}")(x)
        x = tf.keras.layers.Activation(activation, name=f"image_{activation}_{block_index}")(x)
        x = tf.keras.layers.DepthwiseConv2D(
            kernel_size=3,
            padding="same",
            use_bias=not bool(cfg.keras_model.use_batch_norm),
            name=f"image_dwconv_{block_index}",
        )(x)
        if bool(cfg.keras_model.use_batch_norm):
            x = tf.keras.layers.BatchNormalization(name=f"image_dwbn_{block_index}")(x)
        x = tf.keras.layers.Activation(activation, name=f"image_dw_{activation}_{block_index}")(x)
    image = tf.keras.layers.GlobalAveragePooling2D(name="image_pool")(x)
    image = tf.keras.layers.Dense(image_dim, activation=activation, name="image_projection")(image)

    fused = tf.keras.layers.Concatenate(name="prompt_image_concat")([query, image])
    fused = tf.keras.layers.Dense(fusion_dim, activation=activation, name="fusion_dense")(fused)
    if float(cfg.keras_model.dropout) > 0:
        fused = tf.keras.layers.Dropout(float(cfg.keras_model.dropout), name="fusion_dropout")(fused)
    logits = tf.keras.layers.Dense(num_classes, name="logits")(fused)
    return tf.keras.Model(
        inputs={"token_ids": token_ids, "images": images},
        outputs=logits,
        name="tallyqa_tflite_prior_student",
    )


def keras_mobilenet_cutoff_layer(backbone: str, cutoff: str | int | None) -> str | None:
    if cutoff is None or cutoff == "none":
        return None
    if isinstance(cutoff, int) or str(cutoff).isdigit():
        raise ValueError(
            "Keras MobileNetV3 cutoffs are layer names. Use 'auto', 'none', or a Keras layer name."
        )
    if cutoff != "auto":
        return str(cutoff)
    if backbone == "mobilenet_v3_large":
        return "expanded_conv_11/Add"
    if backbone == "mobilenet_v3_small":
        return "expanded_conv_7/Add"
    raise ValueError("keras_model.image_backbone must be mobilenet_v3_large or mobilenet_v3_small.")


@tf.keras.utils.register_keras_serializable(package="vlm_micro")
class TilePromptQueryToFeatureMap(tf.keras.layers.Layer):
    def call(self, inputs: list[tf.Tensor] | tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        query_map, features = inputs
        return tf.tile(
            query_map,
            [
                1,
                tf.shape(features)[1],
                tf.shape(features)[2],
                1,
            ],
        )


@tf.keras.utils.register_keras_serializable(package="vlm_micro")
class BatchedMatMul(tf.keras.layers.Layer):
    def __init__(self, transpose_b: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.transpose_b = bool(transpose_b)

    def call(self, inputs: list[tf.Tensor] | tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        return tf.matmul(inputs[0], inputs[1], transpose_b=self.transpose_b)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config["transpose_b"] = self.transpose_b
        return config


@tf.keras.utils.register_keras_serializable(package="vlm_micro")
class FirstPromptToken(tf.keras.layers.Layer):
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return inputs[:, 0, :]


def build_keras_mobilenet(
    cfg: DictConfig,
    images: tf.keras.layers.Input,
) -> tf.keras.Model:
    backbone_name = str(cfg.keras_model.get("image_backbone", cfg.model.get("image_backbone", "mobilenet_v3_small")))
    weights = "imagenet" if bool(cfg.model.get("image_pretrained", True)) else None
    kwargs = {
        "include_top": False,
        "weights": weights,
        "input_tensor": images,
        "minimalistic": bool(cfg.keras_model.get("mobilenet_minimalistic", False)),
        "include_preprocessing": bool(cfg.keras_model.get("include_mobilenet_preprocessing", True)),
    }
    if backbone_name == "mobilenet_v3_large":
        backbone = tf.keras.applications.MobileNetV3Large(**kwargs)
    elif backbone_name == "mobilenet_v3_small":
        backbone = tf.keras.applications.MobileNetV3Small(**kwargs)
    else:
        raise ValueError("keras_model.image_backbone must be mobilenet_v3_large or mobilenet_v3_small.")
    cutoff = keras_mobilenet_cutoff_layer(
        backbone_name,
        cfg.keras_model.get("image_feature_cutoff", cfg.model.get("image_feature_cutoff", "auto")),
    )
    if cutoff is None:
        return backbone
    return tf.keras.Model(backbone.input, backbone.get_layer(cutoff).output, name=f"{backbone_name}_cutoff")


def add_learned_position_embeddings(
    tokens: tf.Tensor,
    token_count: int,
    fusion_dim: int,
    name: str,
) -> tf.Tensor:
    positions = tf.range(token_count, dtype=tf.int32)[tf.newaxis, :]
    position_embeddings = tf.keras.layers.Embedding(
        token_count,
        fusion_dim,
        embeddings_initializer="zeros",
        name=f"{name}_embedding",
    )(positions)
    return tf.keras.layers.Add(name=f"{name}_add")([tokens, position_embeddings])


def add_prompt_identity(
    query_token: tf.Tensor,
    fusion_dim: int,
) -> tf.Tensor:
    identity_index = tf.zeros((1, 1), dtype=tf.int32)
    identity = tf.keras.layers.Embedding(
        1,
        fusion_dim,
        embeddings_initializer="zeros",
        name="prompt_identity_embedding",
    )(identity_index)
    return tf.keras.layers.Add(name="prompt_identity_add")([query_token, identity])


def apply_feature_film_2d(
    features: tf.Tensor,
    query: tf.Tensor,
    channels: int,
    name: str,
) -> tf.Tensor:
    scale = tf.keras.layers.Dense(
        channels,
        kernel_initializer="zeros",
        bias_initializer="zeros",
        name=f"{name}_scale",
    )(query)
    shift = tf.keras.layers.Dense(
        channels,
        kernel_initializer="zeros",
        bias_initializer="zeros",
        name=f"{name}_shift",
    )(query)
    scale = tf.keras.layers.Reshape((1, 1, channels), name=f"{name}_scale_reshape")(scale)
    shift = tf.keras.layers.Reshape((1, 1, channels), name=f"{name}_shift_reshape")(shift)
    delta = tf.keras.layers.Multiply(name=f"{name}_scale_mul")([features, scale])
    return tf.keras.layers.Add(name=f"{name}_add")([features, delta, shift])


def apply_feature_film_tokens(
    tokens: tf.Tensor,
    query: tf.Tensor,
    channels: int,
    name: str,
) -> tf.Tensor:
    scale = tf.keras.layers.Dense(
        channels,
        kernel_initializer="zeros",
        bias_initializer="zeros",
        name=f"{name}_scale",
    )(query)
    shift = tf.keras.layers.Dense(
        channels,
        kernel_initializer="zeros",
        bias_initializer="zeros",
        name=f"{name}_shift",
    )(query)
    scale = tf.keras.layers.Reshape((1, channels), name=f"{name}_scale_reshape")(scale)
    shift = tf.keras.layers.Reshape((1, channels), name=f"{name}_shift_reshape")(shift)
    delta = tf.keras.layers.Multiply(name=f"{name}_scale_mul")([tokens, scale])
    return tf.keras.layers.Add(name=f"{name}_add")([tokens, delta, shift])


def broadcast_query_to_feature_map(
    query: tf.Tensor,
    features: tf.Tensor,
    channels: int,
    name: str,
) -> tf.Tensor:
    query_map = tf.keras.layers.Reshape((1, 1, channels), name=f"{name}_reshape")(query)
    return TilePromptQueryToFeatureMap(name=f"{name}_tile")([query_map, features])


def normformer_block(
    tokens: tf.Tensor,
    fusion_dim: int,
    heads: int,
    mlp_ratio: int,
    dropout: float,
    activation: str,
    index: int,
) -> tf.Tensor:
    residual = tokens
    x = tf.keras.layers.LayerNormalization(epsilon=1e-5, name=f"fusion_{index}_attn_norm")(tokens)
    x = tf.keras.layers.MultiHeadAttention(
        num_heads=heads,
        key_dim=fusion_dim // heads,
        dropout=dropout,
        name=f"fusion_{index}_mha",
    )(x, x)
    if dropout > 0:
        x = tf.keras.layers.Dropout(dropout, name=f"fusion_{index}_attn_dropout")(x)
    tokens = tf.keras.layers.Add(name=f"fusion_{index}_attn_residual")([residual, x])

    residual = tokens
    x = tf.keras.layers.LayerNormalization(epsilon=1e-5, name=f"fusion_{index}_mlp_norm")(tokens)
    x = tf.keras.layers.Dense(
        fusion_dim * mlp_ratio,
        activation=activation,
        name=f"fusion_{index}_mlp_up",
    )(x)
    if dropout > 0:
        x = tf.keras.layers.Dropout(dropout, name=f"fusion_{index}_mlp_dropout")(x)
    x = tf.keras.layers.Dense(fusion_dim, name=f"fusion_{index}_mlp_down")(x)
    tokens = tf.keras.layers.Add(name=f"fusion_{index}_mlp_residual")([residual, x])
    return tokens


def static_normformer_block(
    tokens: tf.Tensor,
    token_count: int,
    fusion_dim: int,
    heads: int,
    mlp_ratio: int,
    dropout: float,
    activation: str,
    attention_normalization: str,
    index: int,
) -> tf.Tensor:
    if fusion_dim % heads != 0:
        raise ValueError("fusion_dim must be divisible by fusion_heads.")
    key_dim = fusion_dim // heads

    residual = tokens
    x = tf.keras.layers.LayerNormalization(epsilon=1e-5, name=f"fusion_{index}_attn_norm")(tokens)
    q = tf.keras.layers.Conv1D(fusion_dim, kernel_size=1, name=f"fusion_{index}_q")(x)
    k = tf.keras.layers.Conv1D(fusion_dim, kernel_size=1, name=f"fusion_{index}_k")(x)
    v = tf.keras.layers.Conv1D(fusion_dim, kernel_size=1, name=f"fusion_{index}_v")(x)
    q = tf.keras.layers.Reshape((token_count, heads, key_dim), name=f"fusion_{index}_q_heads")(q)
    k = tf.keras.layers.Reshape((token_count, heads, key_dim), name=f"fusion_{index}_k_heads")(k)
    v = tf.keras.layers.Reshape((token_count, heads, key_dim), name=f"fusion_{index}_v_heads")(v)
    q = tf.keras.layers.Permute((2, 1, 3), name=f"fusion_{index}_q_permute")(q)
    k = tf.keras.layers.Permute((2, 1, 3), name=f"fusion_{index}_k_permute")(k)
    v = tf.keras.layers.Permute((2, 1, 3), name=f"fusion_{index}_v_permute")(v)
    scores = BatchedMatMul(
        transpose_b=True,
        name=f"fusion_{index}_attention_scores",
    )([q, k])
    scores = tf.keras.layers.Rescaling(1.0 / math.sqrt(key_dim), name=f"fusion_{index}_attention_scale")(
        scores
    )
    if attention_normalization == "softmax":
        weights = tf.keras.layers.Softmax(axis=-1, name=f"fusion_{index}_attention_softmax")(scores)
    elif attention_normalization == "none":
        weights = scores
    else:
        raise ValueError("attention_normalization must be one of {'softmax', 'none'}.")
    context = BatchedMatMul(name=f"fusion_{index}_attention_context")([weights, v])
    context = tf.keras.layers.Permute((2, 1, 3), name=f"fusion_{index}_context_permute")(context)
    context = tf.keras.layers.Reshape((token_count, fusion_dim), name=f"fusion_{index}_context_flat")(
        context
    )
    x = tf.keras.layers.Conv1D(fusion_dim, kernel_size=1, name=f"fusion_{index}_attention_output")(
        context
    )
    if dropout > 0:
        x = tf.keras.layers.Dropout(dropout, name=f"fusion_{index}_attn_dropout")(x)
    tokens = tf.keras.layers.Add(name=f"fusion_{index}_attn_residual")([residual, x])

    residual = tokens
    x = tf.keras.layers.LayerNormalization(epsilon=1e-5, name=f"fusion_{index}_mlp_norm")(tokens)
    x = tf.keras.layers.Conv1D(
        fusion_dim * mlp_ratio,
        kernel_size=1,
        activation=activation,
        name=f"fusion_{index}_mlp_up",
    )(x)
    if dropout > 0:
        x = tf.keras.layers.Dropout(dropout, name=f"fusion_{index}_mlp_dropout")(x)
    x = tf.keras.layers.Conv1D(fusion_dim, kernel_size=1, name=f"fusion_{index}_mlp_down")(x)
    tokens = tf.keras.layers.Add(name=f"fusion_{index}_mlp_residual")([residual, x])
    return tokens


def tallyqa_fusion_head_logits(
    image_features: tf.Tensor,
    query: tf.Tensor,
    query_token: tf.Tensor,
    cfg: DictConfig,
    num_classes: int,
) -> tf.Tensor:
    fusion_dim = int(cfg.keras_model.get("fusion_dim", cfg.model.fusion_dim))
    heads = int(cfg.keras_model.get("fusion_heads", cfg.model.fusion_heads))
    if fusion_dim % heads != 0:
        raise ValueError("fusion_dim must be divisible by fusion_heads.")
    fusion_depth = int(cfg.keras_model.get("fusion_depth", cfg.model.fusion_depth))
    mlp_ratio = int(cfg.keras_model.get("fusion_mlp_ratio", cfg.model.fusion_mlp_ratio))
    dropout = float(cfg.keras_model.get("dropout", cfg.model.dropout))
    activation = str(cfg.keras_model.get("activation", "gelu"))
    fusion_mode = str(cfg.keras_model.get("fusion_mode", "normformer"))
    image_film_at = cfg.keras_model.get("image_film_at", None)
    attention_impl = str(cfg.keras_model.get("attention_impl", "keras"))
    attention_normalization = str(cfg.keras_model.get("attention_normalization", "softmax"))
    use_image_positional_embeddings = bool(
        cfg.keras_model.get(
            "use_image_positional_embeddings",
            cfg.model.use_image_positional_embeddings,
        )
    )

    feature_height = int(image_features.shape[1])
    feature_width = int(image_features.shape[2])
    if feature_height <= 0 or feature_width <= 0:
        raise ValueError(f"Image feature spatial shape must be static; got {image_features.shape}.")
    image_tokens = tf.keras.layers.Conv2D(
        fusion_dim,
        kernel_size=1,
        padding="same",
        activation=None,
        name="image_token_projection",
    )(image_features)
    if image_film_at not in (None, "none", "null", False):
        if str(image_film_at) not in {"image_tokens", "token_projection"}:
            raise ValueError(
                "keras_model.image_film_at currently supports only "
                "{'image_tokens', 'token_projection', null}."
            )
        image_tokens = apply_feature_film_2d(
            image_tokens,
            query,
            fusion_dim,
            name="image_token_film",
        )
    height = int(image_tokens.shape[1])
    width = int(image_tokens.shape[2])
    if height <= 0 or width <= 0:
        raise ValueError(f"Image token spatial shape must be static; got {image_tokens.shape}.")
    token_count = height * width
    image_tokens = tf.keras.layers.Reshape((token_count, fusion_dim), name="image_tokens")(image_tokens)
    if use_image_positional_embeddings:
        image_tokens = add_learned_position_embeddings(
            image_tokens,
            token_count,
            fusion_dim,
            name="image_position",
        )

    if fusion_mode == "mlp":
        image = tf.keras.layers.GlobalAveragePooling1D(name="image_token_mean")(image_tokens)
        query_flat = tf.keras.layers.Reshape((fusion_dim,), name="prompt_token_flat")(query_token)
        fused = tf.keras.layers.Concatenate(name="prompt_image_concat")([query_flat, image])
        fused = tf.keras.layers.Dense(
            fusion_dim * mlp_ratio,
            activation=activation,
            name="fusion_mlp_0",
        )(fused)
        if dropout > 0:
            fused = tf.keras.layers.Dropout(dropout, name="fusion_mlp_dropout")(fused)
        fused = tf.keras.layers.Dense(
            fusion_dim,
            activation=activation,
            name="fusion_mlp_1",
        )(fused)
    elif fusion_mode == "prompt_patch_mlp":
        query_map = broadcast_query_to_feature_map(
            query,
            image_features,
            fusion_dim,
            name="prompt_patch_query",
        )
        conditioned = tf.keras.layers.Concatenate(axis=-1, name="prompt_patch_concat")(
            [image_features, query_map]
        )
        conditioned = tf.keras.layers.Conv2D(
            fusion_dim * mlp_ratio,
            kernel_size=1,
            padding="same",
            activation=activation,
            name="prompt_patch_conv1x1",
        )(conditioned)
        if dropout > 0:
            conditioned = tf.keras.layers.SpatialDropout2D(
                dropout,
                name="prompt_patch_dropout",
            )(conditioned)
        conditioned = tf.keras.layers.Conv2D(
            128,
            kernel_size=3,
            padding="same",
            activation=activation,
            name="prompt_patch_conv3x3",
        )(conditioned)
        fused = tf.keras.layers.GlobalAveragePooling2D(name="prompt_patch_mean_pool")(
            conditioned
        )
    elif fusion_mode == "film_mlp":
        image = tf.keras.layers.GlobalAveragePooling1D(name="image_token_mean")(image_tokens)
        fused = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="fusion_mlp_input_norm")(image)
        fused = tf.keras.layers.Dense(
            fusion_dim * mlp_ratio,
            activation=activation,
            name="fusion_mlp_0",
        )(fused)
        if dropout > 0:
            fused = tf.keras.layers.Dropout(dropout, name="fusion_mlp_dropout")(fused)
        fused = tf.keras.layers.Dense(
            fusion_dim,
            activation=activation,
            name="fusion_mlp_1",
        )(fused)
        fused = tf.keras.layers.Dense(
            fusion_dim,
            activation=activation,
            name="fusion_mlp_2",
        )(fused)
    elif fusion_mode == "normformer":
        tokens = tf.keras.layers.Concatenate(axis=1, name="prompt_image_tokens")(
            [query_token, image_tokens]
        )
        for index in range(fusion_depth):
            if attention_impl == "keras":
                tokens = normformer_block(
                    tokens,
                    fusion_dim,
                    heads,
                    mlp_ratio,
                    dropout,
                    activation,
                    index,
                )
            elif attention_impl == "static":
                tokens = static_normformer_block(
                    tokens,
                    token_count + 1,
                    fusion_dim,
                    heads,
                    mlp_ratio,
                    dropout,
                    activation,
                    attention_normalization,
                    index,
                )
            else:
                raise ValueError("keras_model.attention_impl must be one of {'keras', 'static'}.")
        fused = tf.keras.layers.GlobalAveragePooling1D(name="fusion_token_mean")(tokens)
        fused = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="fusion_output_norm")(fused)
    else:
        raise ValueError(
            "keras_model.fusion_mode must be one of "
            "{'normformer', 'mlp', 'film_mlp', 'prompt_patch_mlp'}."
        )

    return tf.keras.layers.Dense(num_classes, name="logits")(fused)


def prompt_query_tensors(
    token_ids: tf.Tensor,
    cfg: DictConfig,
    embedding_rows: np.ndarray,
) -> tuple[tf.Tensor, tf.Tensor]:
    fusion_dim = int(cfg.keras_model.get("fusion_dim", cfg.model.fusion_dim))
    activation = str(cfg.keras_model.get("activation", "gelu"))
    pad = np.zeros((1, embedding_rows.shape[1]), dtype=np.float32)
    embedding_init = np.concatenate([pad, embedding_rows.astype(np.float32)], axis=0)
    embedded = tf.keras.layers.Embedding(
        input_dim=embedding_init.shape[0],
        output_dim=embedding_init.shape[1],
        embeddings_initializer=tf.keras.initializers.Constant(embedding_init),
        trainable=not bool(cfg.model.freeze_embeddings),
        mask_zero=bool(cfg.keras_model.get("mask_zero_prompt_embeddings", True)),
        name="compact_prompt_embedding",
    )(token_ids)
    if bool(cfg.keras_model.get("static_single_prompt_token", False)):
        query = FirstPromptToken(name="first_prompt_embedding")(embedded)
    else:
        query = tf.keras.layers.GlobalAveragePooling1D(name="mean_prompt_embedding")(embedded)
    query = tf.keras.layers.Dense(
        fusion_dim,
        activation=activation,
        name="prompt_projection_dense",
    )(query)
    query = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="prompt_projection_norm")(query)
    query_token = tf.keras.layers.Reshape((1, fusion_dim), name="prompt_token")(query)
    if bool(cfg.keras_model.get("use_prompt_identity", cfg.model.use_prompt_identity)):
        query_token = add_prompt_identity(query_token, fusion_dim)
    return query, query_token


def build_fusion_head_visualization_model(
    cfg: DictConfig,
    embedding_rows: np.ndarray,
    prompt_length: int,
    image_feature_shape: tuple[int, int, int],
) -> tf.keras.Model:
    batch_size = cfg.keras_model.get("batch_size", None)
    batch_size = None if batch_size is None else int(batch_size)
    token_ids = tf.keras.Input(
        shape=(prompt_length,),
        batch_size=batch_size,
        dtype=tf.int32,
        name="token_ids",
    )
    image_features = tf.keras.Input(
        shape=image_feature_shape,
        batch_size=batch_size,
        dtype=tf.float32,
        name="mobilenet_cut_tensor",
    )
    query, query_token = prompt_query_tensors(token_ids, cfg, embedding_rows)
    logits = tallyqa_fusion_head_logits(
        image_features,
        query,
        query_token,
        cfg,
        int(cfg.model.num_outputs),
    )
    return tf.keras.Model(
        inputs={"token_ids": token_ids, "mobilenet_cut_tensor": image_features},
        outputs=logits,
        name=f"tallyqa_{cfg.keras_model.get('fusion_mode', 'fusion')}_head_visualization",
    )


def infer_mobilenet_cut_shape(cfg: DictConfig) -> tuple[int, int, int] | None:
    if str(cfg.keras_model.get("architecture", "legacy_prior")) != "current_student":
        return None
    image_size = int(cfg.keras_model.image_size)
    images = tf.keras.Input(
        shape=(image_size, image_size, 3),
        dtype=tf.float32,
        name="fusion_head_shape_probe_images",
    )
    backbone = build_keras_mobilenet(cfg, images)
    output_shape = tuple(backbone.output_shape[1:])
    if len(output_shape) != 3 or any(dim is None for dim in output_shape):
        raise ValueError(f"Could not infer static MobileNet cut tensor shape: {output_shape}.")
    return tuple(int(dim) for dim in output_shape)


def build_tallyqa_current_student_model(
    cfg: DictConfig,
    embedding_rows: np.ndarray,
    prompt_length: int,
) -> tf.keras.Model:
    num_classes = int(cfg.model.num_outputs)
    fusion_dim = int(cfg.keras_model.get("fusion_dim", cfg.model.fusion_dim))
    heads = int(cfg.keras_model.get("fusion_heads", cfg.model.fusion_heads))
    if fusion_dim % heads != 0:
        raise ValueError("fusion_dim must be divisible by fusion_heads.")
    fusion_mode = str(cfg.keras_model.get("fusion_mode", "normformer"))

    batch_size = cfg.keras_model.get("batch_size", None)
    batch_size = None if batch_size is None else int(batch_size)
    token_ids = tf.keras.Input(
        shape=(prompt_length,),
        batch_size=batch_size,
        dtype=tf.int32,
        name="token_ids",
    )
    images = tf.keras.Input(
        shape=(int(cfg.keras_model.image_size), int(cfg.keras_model.image_size), 3),
        batch_size=batch_size,
        dtype=tf.float32,
        name="images",
    )

    query, query_token = prompt_query_tensors(token_ids, cfg, embedding_rows)

    backbone = build_keras_mobilenet(cfg, images)
    backbone.trainable = not bool(cfg.model.freeze_image_features)
    # Use the backbone output tensor directly. Calling the backbone as a nested
    # layer hides its Conv/DepthwiseConv leaves from older tf-keras clone_model()
    # implementations, which prevents full QAT annotation.
    image_features = backbone.output
    feature_height = int(image_features.shape[1])
    feature_width = int(image_features.shape[2])
    if feature_height <= 0 or feature_width <= 0:
        raise ValueError(f"Image feature spatial shape must be static; got {image_features.shape}.")
    logits = tallyqa_fusion_head_logits(
        image_features,
        query,
        query_token,
        cfg,
        num_classes,
    )
    return tf.keras.Model(
        inputs={"token_ids": token_ids, "images": images},
        outputs=logits,
        name=f"tallyqa_keras_{fusion_mode}_student",
    )


def build_keras_student_model(
    cfg: DictConfig,
    embedding_rows: np.ndarray,
    prompt_length: int,
) -> tf.keras.Model:
    architecture = str(cfg.keras_model.get("architecture", "legacy_prior"))
    if architecture == "legacy_prior":
        return build_tflite_prior_model(cfg, embedding_rows, prompt_length)
    if architecture == "current_student":
        return build_tallyqa_current_student_model(cfg, embedding_rows, prompt_length)
    raise ValueError("keras_model.architecture must be one of {'legacy_prior', 'current_student'}.")


def qat_quantizable_layer_types() -> tuple[type[tf.keras.layers.Layer], ...]:
    return (
        tf.keras.layers.Conv1D,
        tf.keras.layers.Conv2D,
        tf.keras.layers.DepthwiseConv2D,
        tf.keras.layers.Dense,
    )


def is_quantize_wrapper(layer: tf.keras.layers.Layer) -> bool:
    return layer.__class__.__name__.startswith("QuantizeWrapper")


def qat_coverage_report(model: tf.keras.Model) -> dict[str, Any]:
    quantizable_types = qat_quantizable_layer_types()
    wrapped: list[dict[str, str]] = []
    unwrapped: list[dict[str, str]] = []
    wrappers: list[dict[str, str]] = []
    for path, layer in iter_leaf_layers(model):
        if is_quantize_wrapper(layer):
            inner = getattr(layer, "layer", None)
            inner_class = inner.__class__.__name__ if inner is not None else "unknown"
            wrappers.append({"name": path, "wrapped_class": inner_class})
            if inner is not None and isinstance(inner, quantizable_types):
                wrapped.append({"name": path, "class": inner_class})
            continue
        if isinstance(layer, quantizable_types):
            unwrapped.append({"name": path, "class": layer.__class__.__name__})
    total = len(wrapped) + len(unwrapped)
    return {
        "quantize_wrapper_layers": len(wrappers),
        "wrapped_quantizable_leaf_layers": len(wrapped),
        "unwrapped_quantizable_leaf_layers": len(unwrapped),
        "quantizable_leaf_layers": total,
        "wrapped_fraction": float(len(wrapped) / total) if total else 1.0,
        "wrapped_layers": wrapped,
        "unwrapped_layers": unwrapped,
        "all_wrappers": wrappers,
    }


def qat_full_integer_runtime_gap_report(
    model: tf.keras.Model,
    cfg: DictConfig,
) -> dict[str, Any]:
    """Find graph pieces full-int8 TFLite quantizes beyond tfmot QAT wrappers."""

    enabled = (
        str(cfg.export.quantization.mode) == "qat"
        and bool(cfg.export.export_tflite)
        and bool(cfg.export.quantization.get("full_integer", False))
    )
    if not enabled:
        return {
            "enabled": False,
            "uncovered_runtime_layers": [],
            "uncovered_runtime_layer_count": 0,
        }

    risky_layers: list[dict[str, str]] = []
    for path, layer in iter_leaf_layers(model):
        inner = getattr(layer, "layer", None) if is_quantize_wrapper(layer) else layer
        layer_class = inner.__class__.__name__ if inner is not None else layer.__class__.__name__
        lower_path = path.lower()
        reason: str | None = None
        if layer_class == "Embedding":
            reason = "runtime embedding/gather is quantized by full-integer TFLite but is not a QAT-wrapped weight layer"
        elif layer_class == "LayerNormalization":
            reason = "LayerNorm lowers to int8 arithmetic in full-integer TFLite without an equivalent tfmot QAT wrapper"
        elif layer_class == "GlobalAveragePooling1D" and "prompt" in lower_path:
            reason = "mask-aware prompt pooling is quantized by TFLite but not trained through deployed integer arithmetic"
        elif layer_class in {"TilePromptQueryToFeatureMap", "FirstPromptToken"}:
            reason = "custom prompt tensor manipulation is quantized by full-integer TFLite outside tfmot QAT coverage"
        elif layer_class in {"Concatenate", "Multiply", "Add"} and "prompt" in lower_path:
            reason = "prompt fusion arithmetic is quantized by full-integer TFLite outside tfmot QAT coverage"
        if reason is not None:
            risky_layers.append(
                {
                    "name": path,
                    "class": layer_class,
                    "reason": reason,
                }
            )

    return {
        "enabled": True,
        "uncovered_runtime_layers": risky_layers,
        "uncovered_runtime_layer_count": len(risky_layers),
    }


def assert_qat_coverage(model: tf.keras.Model, cfg: DictConfig) -> dict[str, Any]:
    coverage = qat_coverage_report(model)
    coverage["full_integer_runtime_gap"] = qat_full_integer_runtime_gap_report(model, cfg)
    if str(cfg.export.quantization.mode) != "qat":
        return coverage
    if not bool(cfg.export.quantization.get("require_full_qat_coverage", True)):
        return coverage
    unwrapped = coverage["unwrapped_layers"]
    runtime_gaps = coverage["full_integer_runtime_gap"]["uncovered_runtime_layers"]
    if not unwrapped and not runtime_gaps:
        return coverage
    details: list[str] = []
    if unwrapped:
        preview = "\n".join(
            f"  - {layer['name']} ({layer['class']})" for layer in unwrapped[:30]
        )
        remainder = len(unwrapped) - min(len(unwrapped), 30)
        if remainder > 0:
            preview += f"\n  ... {remainder} more"
        details.append(
            f"{len(unwrapped)} quantizable leaf layers are not wrapped with fake quantization:\n"
            f"{preview}"
        )
    if runtime_gaps:
        preview = "\n".join(
            f"  - {layer['name']} ({layer['class']}): {layer['reason']}"
            for layer in runtime_gaps[:30]
        )
        remainder = len(runtime_gaps) - min(len(runtime_gaps), 30)
        if remainder > 0:
            preview += f"\n  ... {remainder} more"
        details.append(
            "full-integer TFLite deployment will quantize runtime layers that are "
            "outside tfmot QAT wrapper coverage:\n"
            f"{preview}"
        )
    raise RuntimeError(
        "QAT coverage is incomplete. Refusing to train a run labeled QAT because "
        "the Keras training graph does not fully simulate the deployed full-integer "
        "TFLite graph.\n"
        f"{chr(10).join(details)}\n"
        "Set export.quantization.require_full_qat_coverage=false only for diagnostic runs."
    )


def maybe_apply_qat(student: tf.keras.Model, cfg: DictConfig) -> tf.keras.Model:
    mode = str(cfg.export.quantization.mode)
    if mode != "qat":
        return student
    try:
        import tensorflow_model_optimization as tfmot
    except ImportError as exc:
        raise SystemExit(
            "export.quantization.mode=qat requires tensorflow-model-optimization. "
            "Install it in the Coral/TensorFlow environment before running QAT."
        ) from exc

    quantize = tfmot.quantization.keras

    def annotate(layer: tf.keras.layers.Layer) -> tf.keras.layers.Layer:
        if isinstance(layer, qat_quantizable_layer_types()):
            return quantize.quantize_annotate_layer(layer)
        return layer

    try:
        annotated = tf.keras.models.clone_model(
            student,
            clone_function=annotate,
            recursive=True,
        )
    except TypeError:
        annotated = tf.keras.models.clone_model(student, clone_function=annotate)
        print(
            "Warning: tf.keras.models.clone_model does not support recursive=True; "
            "nested models may not be QAT-annotated."
        )
    quantized = quantize.quantize_apply(annotated)
    coverage = qat_coverage_report(quantized)
    if coverage["unwrapped_quantizable_leaf_layers"]:
        preview = ", ".join(
            layer["name"] for layer in coverage["unwrapped_layers"][:10]
        )
        print(
            "Warning: QAT left "
            f"{coverage['unwrapped_quantizable_leaf_layers']} quantizable leaf layers "
            f"unwrapped. First unwrapped layers: {preview}"
        )
    return quantized


def local_soft_targets(labels: tf.Tensor, num_classes: int, sigma: float, radius: int) -> tf.Tensor:
    class_ids = tf.range(num_classes, dtype=tf.int32)[tf.newaxis, :]
    distances = tf.abs(class_ids - labels[:, tf.newaxis])
    targets = tf.exp(-(tf.cast(distances, tf.float32) ** 2) / (2.0 * sigma**2))
    if radius > 0:
        targets = tf.where(distances > radius, tf.zeros_like(targets), targets)
    return targets / tf.maximum(tf.reduce_sum(targets, axis=1, keepdims=True), 1e-8)


class DistilledStudent(tf.keras.Model):
    def __init__(
        self,
        student: tf.keras.Model,
        alpha: float,
        beta: float,
        temperature: float,
        target_distribution: str,
        local_soft_sigma: float,
        local_soft_radius: int,
        class_weights: np.ndarray | None,
        kl_class_weights: np.ndarray | None,
        image_learning_rate_scale: float,
        image_learning_rate_scale_schedule: str,
        image_learning_rate_scale_warmup_steps: int,
    ):
        super().__init__(name="distilled_tallyqa_keras_student")
        self.student = student
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.temperature = float(temperature)
        self.target_distribution = target_distribution
        self.local_soft_sigma = float(local_soft_sigma)
        self.local_soft_radius = int(local_soft_radius)
        self.image_learning_rate_scale = float(image_learning_rate_scale)
        if self.image_learning_rate_scale < 0:
            raise ValueError("image_learning_rate_scale must be non-negative.")
        self.image_learning_rate_scale_schedule = str(image_learning_rate_scale_schedule)
        if self.image_learning_rate_scale_schedule not in {"constant", "linear_warmup"}:
            raise ValueError(
                "image_learning_rate_scale_schedule must be one of "
                "{'constant', 'linear_warmup'}."
            )
        self.image_learning_rate_scale_warmup_steps = max(
            1,
            int(image_learning_rate_scale_warmup_steps),
        )
        self.class_weights = (
            tf.constant(class_weights, dtype=tf.float32) if class_weights is not None else None
        )
        self.kl_class_weights = (
            tf.constant(kl_class_weights, dtype=tf.float32) if kl_class_weights is not None else None
        )
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.ce_tracker = tf.keras.metrics.Mean(name="ce_loss")
        self.ce_unweighted_tracker = tf.keras.metrics.Mean(name="ce_loss_unweighted")
        self.kl_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.grad_global_norm_tracker = tf.keras.metrics.Mean(name="grad_global_norm")
        self.grad_max_norm_tracker = tf.keras.metrics.Mean(name="grad_max_norm")
        self.image_lr_scale_tracker = tf.keras.metrics.Mean(name="image_lr_scale")
        self.accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")
        self.mae = tf.keras.metrics.Mean(name="mae")
        self.within_one = tf.keras.metrics.Mean(name="within_1_accuracy")

    @property
    def metrics(self) -> list[tf.keras.metrics.Metric]:
        return [
            self.loss_tracker,
            self.ce_tracker,
            self.ce_unweighted_tracker,
            self.kl_tracker,
            self.accuracy,
            self.mae,
            self.within_one,
        ]

    def reset_metrics(self) -> None:
        super().reset_metrics()
        self.grad_global_norm_tracker.reset_state()
        self.grad_max_norm_tracker.reset_state()
        self.image_lr_scale_tracker.reset_state()

    def call(self, inputs: dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
        return self.student(inputs, training=training)

    def _targets(self, labels: tf.Tensor, num_classes: int) -> tf.Tensor:
        if self.target_distribution == "hard":
            return tf.one_hot(labels, depth=num_classes, dtype=tf.float32)
        return local_soft_targets(
            labels,
            num_classes,
            sigma=self.local_soft_sigma,
            radius=self.local_soft_radius,
        )

    def compute_distillation_losses(
        self,
        logits: tf.Tensor,
        labels: tf.Tensor,
        teacher_probs: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        targets = self._targets(labels, int(logits.shape[1]))
        ce = tf.keras.losses.categorical_crossentropy(targets, logits, from_logits=True)
        ce_loss_unweighted = tf.reduce_mean(ce)
        if self.class_weights is not None:
            ce *= tf.gather(self.class_weights, labels)
        ce_loss = tf.reduce_mean(ce)

        if self.beta > 0:
            teacher_probs = tf.maximum(teacher_probs, 1e-8)
            teacher_probs /= tf.reduce_sum(teacher_probs, axis=1, keepdims=True)
            student_log_probs = tf.nn.log_softmax(logits / self.temperature, axis=1)
            kl = tf.reduce_sum(
                teacher_probs * (tf.math.log(teacher_probs) - student_log_probs),
                axis=1,
            )
            if self.kl_class_weights is not None:
                kl *= tf.gather(self.kl_class_weights, labels)
            kl_loss = tf.reduce_mean(kl) * self.temperature**2
        else:
            kl_loss = tf.zeros((), dtype=tf.float32)
        loss = self.alpha * ce_loss + self.beta * kl_loss
        return loss, ce_loss, ce_loss_unweighted, kl_loss

    def _update_count_metrics(self, labels: tf.Tensor, logits: tf.Tensor) -> None:
        predicted = tf.argmax(logits, axis=1, output_type=tf.int32)
        absolute_error = tf.abs(predicted - labels)
        self.mae.update_state(tf.cast(absolute_error, tf.float32))
        self.within_one.update_state(tf.cast(absolute_error <= 1, tf.float32))

    def effective_image_learning_rate_scale(self) -> tf.Tensor:
        scale = tf.constant(self.image_learning_rate_scale, dtype=tf.float32)
        if self.image_learning_rate_scale_schedule == "constant":
            return scale
        step = tf.cast(self.optimizer.iterations, tf.float32)
        progress = tf.minimum(
            step / float(self.image_learning_rate_scale_warmup_steps),
            1.0,
        )
        return scale * progress

    def train_step(self, data: tuple[dict[str, tf.Tensor], dict[str, tf.Tensor]]) -> dict[str, tf.Tensor]:
        inputs, targets = data
        labels = targets["labels"]
        teacher_probs = targets["teacher_probs"]
        with tf.GradientTape() as tape:
            logits = self.student(inputs, training=True)
            loss, ce_loss, ce_loss_unweighted, kl_loss = self.compute_distillation_losses(
                logits,
                labels,
                teacher_probs,
            )
        gradients = tape.gradient(loss, self.student.trainable_variables)
        image_learning_rate_scale = self.effective_image_learning_rate_scale()
        if self.image_learning_rate_scale_schedule != "constant" or self.image_learning_rate_scale != 1.0:
            gradients = [
                (
                    gradient * image_learning_rate_scale
                    if gradient is not None
                    and (
                        "mobilenet_v3_large_cutoff" in variable.name
                        or "mobilenet_v3_small_cutoff" in variable.name
                    )
                    else gradient
                )
                for gradient, variable in zip(
                    gradients,
                    self.student.trainable_variables,
                    strict=True,
                )
            ]
        self.image_lr_scale_tracker.update_state(image_learning_rate_scale)
        non_none_gradients = [gradient for gradient in gradients if gradient is not None]
        if non_none_gradients:
            self.grad_global_norm_tracker.update_state(tf.linalg.global_norm(non_none_gradients))
            self.grad_max_norm_tracker.update_state(
                tf.reduce_max(
                    tf.stack([tf.reduce_max(tf.abs(gradient)) for gradient in non_none_gradients])
                )
            )
        self.optimizer.apply_gradients(zip(gradients, self.student.trainable_variables, strict=False))
        self.loss_tracker.update_state(loss)
        self.ce_tracker.update_state(ce_loss)
        self.ce_unweighted_tracker.update_state(ce_loss_unweighted)
        self.kl_tracker.update_state(kl_loss)
        self.accuracy.update_state(labels, logits)
        self._update_count_metrics(labels, logits)
        return {
            **{metric.name: metric.result() for metric in self.metrics},
            self.grad_global_norm_tracker.name: self.grad_global_norm_tracker.result(),
            self.grad_max_norm_tracker.name: self.grad_max_norm_tracker.result(),
            self.image_lr_scale_tracker.name: self.image_lr_scale_tracker.result(),
        }

    def test_step(self, data: tuple[dict[str, tf.Tensor], dict[str, tf.Tensor]]) -> dict[str, tf.Tensor]:
        inputs, targets = data
        labels = targets["labels"]
        logits = self.student(inputs, training=False)
        loss, ce_loss, ce_loss_unweighted, kl_loss = self.compute_distillation_losses(
            logits,
            labels,
            targets["teacher_probs"],
        )
        self.loss_tracker.update_state(loss)
        self.ce_tracker.update_state(ce_loss)
        self.ce_unweighted_tracker.update_state(ce_loss_unweighted)
        self.kl_tracker.update_state(kl_loss)
        self.accuracy.update_state(labels, logits)
        self._update_count_metrics(labels, logits)
        return {metric.name: metric.result() for metric in self.metrics}


def compatibility_report(model: tf.keras.Model, cfg: DictConfig) -> dict[str, Any]:
    quantization_mode = str(cfg.export.quantization.mode)
    include_mobilenet_preprocessing = bool(
        cfg.keras_model.get("include_mobilenet_preprocessing", True)
    )
    pytorch_unsupported = [
        {
            "component": "MobileViTFusionBlock / nn.TransformerEncoderLayer",
            "reason": "Dynamic multi-head attention and token concat are not Edge-TPU-friendly.",
            "keras_prior": (
                "Use keras_model.fusion_mode=film_mlp for prompt-FiLM conditioning "
                "plus image-only Dense fusion, or fusion_mode=mlp for the older "
                "prompt-image concat baseline."
            ),
        },
        {
            "component": "LayerNorm",
            "reason": "TFLite can represent some normalization patterns, but Edge TPU mapping is poor.",
            "keras_prior": "Avoided; BatchNorm is used only in convolution blocks.",
        },
        {
            "component": "GELU",
            "reason": "Not a conservative Edge TPU op.",
            "keras_prior": "Replaced with ReLU.",
        },
        {
            "component": "spatial token flatten + positional embeddings",
            "reason": "Creates sequence-style tensor operations instead of conv/pool patterns.",
            "keras_prior": (
                "film_mlp keeps the projected image token map, applies prompt FiLM, "
                "then mean-pools visual tokens before the classifier."
            ),
        },
        {
            "component": "runtime prompt token embedding lookup",
            "reason": "TFLite supports Gather, but Edge TPU will likely leave it on CPU.",
            "keras_prior": "Kept for task parity; future deployment should freeze prompts or precompute prompt vectors.",
        },
    ]
    layers = [
        {
            "name": layer.name,
            "class": layer.__class__.__name__,
            "output_shape": str(getattr(layer, "output_shape", "unknown")),
        }
        for layer in model.layers
    ]
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
        "model_name": model.name,
        "keras_layers": layers,
        "tflite_prior_supported_patterns": [
            "Conv2D",
            "DepthwiseConv2D",
            "BatchNormalization folded at conversion",
            "ReLU",
            "GlobalAveragePooling2D",
            "prompt-conditioned Add/Mul FiLM on image-token maps",
            "Dense",
            "Concatenate",
        ],
        "known_non_edge_tpu_or_risky_patterns": [
            "Embedding/Gather for prompt tokens",
            "prompt embedding/gather and mask-aware pooling are intentionally not QAT-annotated",
            (
                "Keras MobileNetV3 preprocessing is inside the model; inspect the EdgeTPU "
                "compiler report for preprocessing CPU fallback."
                if include_mobilenet_preprocessing
                else "MobileNetV3 preprocessing is externalized; inspect exported TFLite "
                "image input quantization parameters before deployment."
            ),
            "Dropout exists only during training",
            "Keras training path currently does not mirror Lightning warmup scheduling.",
            "Keras W&B watch logging uses an explicit callback for weight and gradient histograms.",
        ],
        "quantization_mode": quantization_mode,
        "quantization_notes": {
            "none": "Float TFLite export only; useful for graph debugging, not Edge TPU.",
            "ptq": (
                "Train float Keras model, then run representative-dataset post-training "
                "quantization. Mixed inputs remain: token IDs are integer indices, image "
                "input can be quantized by TFLite."
            ),
            "qat": (
                "Wrap the Keras student with TensorFlow Model Optimization quantization "
                "before training. This is the right comparison point against PTQ, but "
                "depends on tfmot/Keras compatibility in the local environment."
            ),
        }[quantization_mode],
        "pytorch_student_steps_not_mirrored_for_tflite": pytorch_unsupported,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "compiler_report_layout": {
            "root": str(cfg.export.get("compiler_report_dir", "artifacts/reports/coral/edgetpu_compiler")),
            "per_run": (
                "<root>/<run_name>/{float,ptq,qat}/ containing compiler stdout/stderr, "
                "compiler_summary.json, model_operator_report.txt, and compiled .tflite when produced"
            ),
        },
    }


def maybe_write_visualkeras(model: tf.keras.Model, output: Path) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        import visualkeras
    except ImportError:
        return {
            "enabled": False,
            "status": "missing_dependency",
            "message": "Install visualkeras to generate model architecture PNGs.",
            "output": str(output),
        }
    try:
        visualkeras.layered_view(model, legend=True, to_file=str(output))
    except Exception as exc:
        return {
            "enabled": True,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "output": str(output),
        }
    return {"enabled": True, "status": "written", "output": str(output)}


class WarmupPlateauDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self,
        learning_rate: float,
        warmup_start_learning_rate: float,
        warmup_steps: int,
        decay_start_step: int,
        final_learning_rate: float,
        total_steps: int,
    ):
        super().__init__()
        self.learning_rate = float(learning_rate)
        self.warmup_start_learning_rate = float(warmup_start_learning_rate)
        self.warmup_steps = max(0, int(warmup_steps))
        self.decay_start_step = max(self.warmup_steps, int(decay_start_step))
        self.final_learning_rate = float(final_learning_rate)
        self.total_steps = max(1, int(total_steps))

    def __call__(self, step: tf.Tensor) -> tf.Tensor:
        step_f = tf.cast(step, tf.float32)
        learning_rate = tf.constant(self.learning_rate, dtype=tf.float32)
        warmup_start = tf.constant(self.warmup_start_learning_rate, dtype=tf.float32)
        final_learning_rate = tf.constant(self.final_learning_rate, dtype=tf.float32)
        if self.warmup_steps > 0:
            warmup_progress = tf.clip_by_value(step_f / float(self.warmup_steps), 0.0, 1.0)
            warmup_lr = warmup_start + (learning_rate - warmup_start) * warmup_progress
        else:
            warmup_lr = learning_rate
        decay_progress = tf.clip_by_value(
            (step_f - float(self.decay_start_step))
            / max(1.0, float(self.total_steps - self.decay_start_step)),
            0.0,
            1.0,
        )
        decay_lr = learning_rate + (final_learning_rate - learning_rate) * decay_progress
        return tf.where(
            step_f < float(self.warmup_steps),
            warmup_lr,
            tf.where(step_f < float(self.decay_start_step), learning_rate, decay_lr),
        )

    def get_config(self) -> dict[str, float | int]:
        return {
            "learning_rate": self.learning_rate,
            "warmup_start_learning_rate": self.warmup_start_learning_rate,
            "warmup_steps": self.warmup_steps,
            "decay_start_step": self.decay_start_step,
            "final_learning_rate": self.final_learning_rate,
            "total_steps": self.total_steps,
        }


def keras_learning_rate(cfg: DictConfig, total_steps: int) -> float | tf.keras.optimizers.schedules.LearningRateSchedule:
    schedule = str(cfg.optimizer.get("lr_schedule", "none"))
    learning_rate = float(cfg.optimizer.learning_rate)
    if schedule == "none":
        return learning_rate
    warmup_steps = int(cfg.optimizer.get("warmup_steps", 0) or 0)
    warmup_start = float(cfg.optimizer.get("warmup_start_learning_rate", learning_rate) or learning_rate)
    if schedule == "warmup":
        return WarmupPlateauDecaySchedule(
            learning_rate=learning_rate,
            warmup_start_learning_rate=warmup_start,
            warmup_steps=warmup_steps,
            decay_start_step=total_steps,
            final_learning_rate=learning_rate,
            total_steps=total_steps,
        )
    if schedule == "warmup_plateau_decay":
        if cfg.optimizer.get("lr_decay_start_step", None) is not None:
            decay_start_step = int(cfg.optimizer.lr_decay_start_step)
        else:
            decay_start_step = int(round(total_steps * float(cfg.optimizer.get("lr_decay_start_fraction", 0.5))))
        final_lr = float(cfg.optimizer.get("lr_final_learning_rate", warmup_start) or warmup_start)
        return WarmupPlateauDecaySchedule(
            learning_rate=learning_rate,
            warmup_start_learning_rate=warmup_start,
            warmup_steps=warmup_steps,
            decay_start_step=decay_start_step,
            final_learning_rate=final_lr,
            total_steps=total_steps,
        )
    raise ValueError("optimizer.lr_schedule must be one of {'none', 'warmup', 'warmup_plateau_decay'}.")


class WandbKerasLogger(tf.keras.callbacks.Callback):
    def __init__(self, prefix: str = "", log_every_n_steps: int = 50):
        super().__init__()
        self.prefix = prefix
        self.log_every_n_steps = max(1, int(log_every_n_steps))
        self.global_train_step = 0

    @staticmethod
    def _wandb_key(key: str) -> str:
        if key.startswith("val_"):
            return f"val/{key.removeprefix('val_')}"
        return f"train/{key}"

    @staticmethod
    def _scalar_logs(logs: dict[str, Any] | None) -> dict[str, float]:
        if logs is None:
            return {}
        return {key: float(value) for key, value in logs.items() if np.isscalar(value)}

    def _learning_rate(self) -> float | None:
        optimizer = getattr(self.model, "optimizer", None)
        if optimizer is None:
            return None
        learning_rate = optimizer.learning_rate
        if callable(learning_rate):
            learning_rate = learning_rate(optimizer.iterations)
        return float(tf.keras.backend.get_value(learning_rate))

    def _optimizer_step(self) -> int:
        optimizer = getattr(self.model, "optimizer", None)
        if optimizer is None:
            return self.global_train_step
        return int(tf.keras.backend.get_value(optimizer.iterations))

    def on_train_batch_end(self, batch: int, logs: dict[str, Any] | None = None) -> None:
        self.global_train_step = self._optimizer_step()
        if self.global_train_step % self.log_every_n_steps != 0:
            return
        payload = {
            f"{self.prefix}train/{key}_step": value
            for key, value in self._scalar_logs(logs).items()
        }
        learning_rate = self._learning_rate()
        if learning_rate is not None:
            payload[f"{self.prefix}train/lr"] = learning_rate
        payload["trainer/global_step"] = self.global_train_step
        if payload:
            wandb.log(payload)

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        payload = {
            f"{self.prefix}{self._wandb_key(key)}": float(value)
            for key, value in self._scalar_logs(logs).items()
        }
        learning_rate = self._learning_rate()
        if learning_rate is not None:
            payload[f"{self.prefix}train/lr_epoch"] = learning_rate
        payload["trainer/epoch"] = epoch
        payload["trainer/global_step"] = self.global_train_step
        wandb.log(payload)


class WandbKerasWeightsAndGradients(tf.keras.callbacks.Callback):
    def __init__(
        self,
        sample_dataset: tf.data.Dataset,
        log: str = "all",
        log_freq: int = 100,
    ):
        super().__init__()
        log = str(log)
        if log not in {"all", "parameters", "gradients"}:
            raise ValueError("wandb.watch.log for Keras must be one of {'all', 'parameters', 'gradients'}.")
        self.sample_iterator = iter(sample_dataset)
        self.log = log
        self.log_freq = max(1, int(log_freq))
        self.global_train_step = 0

    def _optimizer_step(self) -> int:
        optimizer = getattr(self.model, "optimizer", None)
        if optimizer is None:
            return self.global_train_step
        return int(tf.keras.backend.get_value(optimizer.iterations))

    @staticmethod
    def _variable_key(variable: tf.Variable) -> str:
        name = str(getattr(variable, "path", getattr(variable, "name", "variable")))
        name = name.replace(":0", "")
        return re.sub(r"[^A-Za-z0-9_.-]+", "/", name).strip("/") or "variable"

    @staticmethod
    def _histogram(tensor: tf.Tensor) -> wandb.Histogram:
        values = tf.reshape(tf.cast(tensor, tf.float32), [-1]).numpy()
        return wandb.Histogram(values)

    def on_train_batch_end(self, batch: int, logs: dict[str, Any] | None = None) -> None:
        self.global_train_step = self._optimizer_step()
        if self.global_train_step % self.log_freq != 0:
            return
        inputs, targets = next(self.sample_iterator)
        labels = targets["labels"]
        teacher_probs = targets["teacher_probs"]
        with tf.GradientTape() as tape:
            logits = self.model.student(inputs, training=True)
            loss, _ce_loss, _ce_loss_unweighted, _kl_loss = self.model.compute_distillation_losses(
                logits,
                labels,
                teacher_probs,
            )
        variables = list(self.model.student.trainable_variables)
        gradients = tape.gradient(loss, variables)
        payload: dict[str, Any] = {
            "trainer/global_step": int(self.global_train_step),
            "wandb_watch/loss": float(loss.numpy()),
        }
        if self.log in {"all", "parameters"}:
            for variable in variables:
                key = self._variable_key(variable)
                payload[f"weights/{key}"] = self._histogram(variable)
        non_none_gradients: list[tf.Tensor] = []
        if self.log in {"all", "gradients"}:
            for variable, gradient in zip(variables, gradients, strict=True):
                if gradient is None:
                    continue
                non_none_gradients.append(gradient)
                key = self._variable_key(variable)
                payload[f"gradients/{key}"] = self._histogram(gradient)
            if non_none_gradients:
                payload["gradients/global_norm"] = float(
                    tf.linalg.global_norm(non_none_gradients).numpy()
                )
                payload["gradients/max_abs"] = float(
                    tf.reduce_max(
                        tf.stack(
                            [
                                tf.reduce_max(tf.abs(tf.cast(gradient, tf.float32)))
                                for gradient in non_none_gradients
                            ]
                        )
                    ).numpy()
                )
        wandb.log(payload)


class WandbEvaluationLogger(tf.keras.callbacks.Callback):
    def __init__(
        self,
        train_dataset: tf.data.Dataset,
        train_steps: int,
        val_dataset: tf.data.Dataset,
        val_steps: int,
        num_classes: int,
        output_dir: Path,
        data: KerasTallyQAData,
        example_samples: int,
        example_every_n_epochs: int,
        log_train_metrics: bool,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.train_steps = train_steps
        self.val_dataset = val_dataset
        self.val_steps = val_steps
        self.num_classes = num_classes
        self.output_dir = output_dir
        self.data = data
        self.example_samples = max(0, int(example_samples))
        self.example_every_n_epochs = max(1, int(example_every_n_epochs))
        self.log_train_metrics = bool(log_train_metrics)

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        logs = logs if logs is not None else {}
        if self.log_train_metrics:
            train_accumulator = evaluate_split_metrics(
                self.model,
                self.train_dataset,
                self.train_steps,
                self.num_classes,
                self.data,
                description="train metrics",
            )
            for name, value in train_accumulator.metrics().items():
                logs[name] = float(value)
        accumulator = evaluate_split_metrics(
            self.model,
            self.val_dataset,
            self.val_steps,
            self.num_classes,
            self.data,
            description="val metrics",
        )
        metrics = accumulator.metrics()
        for name, value in metrics.items():
            logs[f"val_{name}"] = float(value)
        payload: dict[str, Any] = {
            **(
                {
                    f"train/{name}": float(value)
                    for name, value in train_accumulator.metrics().items()
                }
                if self.log_train_metrics
                else {}
            ),
            **{f"val/{name}": float(value) for name, value in metrics.items()},
            "trainer/epoch": epoch,
            "trainer/global_step": int(self.model.optimizer.iterations.numpy()),
        }
        if int(accumulator.confusion.sum()) > 0:
            figure_path = save_confusion_matrix_plot(
                "val",
                accumulator,
                self.output_dir
                / "validation_plots"
                / f"confusion_matrix_epoch_{epoch + 1:03d}.png",
            )
            payload["val_plots/confusion_matrix"] = wandb.Image(str(figure_path))
            save_wandb_file(Path(figure_path), policy="now")
        if (
            self.example_samples > 0
            and (epoch + 1) % self.example_every_n_epochs == 0
        ):
            activation_path = save_activation_examples_plot(
                "val",
                self.model,
                self.data,
                self.output_dir
                / "validation_plots"
                / f"image_encoding_epoch_{epoch + 1:03d}.png",
                self.example_samples,
            )
            if activation_path is not None:
                payload["validation_plots/image_encoding"] = wandb.Image(str(activation_path))
                payload["validation_plots/image_encoding_count"] = self.example_samples
                save_wandb_file(activation_path, policy="now")
            examples_path = save_prediction_examples_plot(
                "val",
                self.model,
                self.data,
                self.output_dir
                / "validation_plots"
                / f"prediction_examples_epoch_{epoch + 1:03d}.png",
                self.example_samples,
            )
            if examples_path is not None:
                payload["val_plots/example_predictions"] = wandb.Image(str(examples_path))
                save_wandb_file(examples_path, policy="now")
        wandb.log(payload)

class TqdmKerasProgress(tf.keras.callbacks.Callback):
    def __init__(self, train_steps: int, val_steps: int):
        super().__init__()
        self.train_steps = train_steps
        self.val_steps = val_steps
        self.epoch_bar: tqdm | None = None
        self.batch_bar: tqdm | None = None
        self._seen_train_batches = 0

    def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
        epochs = int(self.params.get("epochs", 0) or 0)
        self.epoch_bar = tqdm(total=epochs, desc="epochs", unit="epoch")

    def on_epoch_begin(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        self.batch_bar = tqdm(total=self.train_steps, desc=f"train {epoch + 1}", unit="batch", leave=False)
        self._seen_train_batches = 0

    def on_train_batch_end(self, batch: int, logs: dict[str, Any] | None = None) -> None:
        if self.batch_bar is None:
            return
        if logs:
            self.batch_bar.set_postfix(
                {
                    key: f"{float(value):.4g}"
                    for key, value in logs.items()
                    if key
                    in {
                        "loss",
                        "ce_loss",
                        "kl_loss",
                        "accuracy",
                        "mae",
                        "image_lr_scale",
                    }
                },
                refresh=False,
            )
        seen = min(self.train_steps, int(batch) + 1)
        delta = max(0, seen - self._seen_train_batches)
        self._seen_train_batches = seen
        self.batch_bar.update(delta)

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        if self.batch_bar is not None:
            self.batch_bar.close()
            self.batch_bar = None
        if self.epoch_bar is not None:
            if logs:
                self.epoch_bar.set_postfix(
                    {
                        key: f"{float(value):.4g}"
                        for key, value in logs.items()
                        if key
                        in {
                            "loss",
                            "val_loss",
                            "accuracy",
                            "val_accuracy",
                            "mae",
                            "val_mae",
                            "image_lr_scale",
                        }
                    },
                    refresh=False,
                )
            self.epoch_bar.update(1)

    def on_train_end(self, logs: dict[str, Any] | None = None) -> None:
        if self.batch_bar is not None:
            self.batch_bar.close()
            self.batch_bar = None
        if self.epoch_bar is not None:
            self.epoch_bar.close()
            self.epoch_bar = None


class KerasDataEpochCallback(tf.keras.callbacks.Callback):
    def __init__(self, data: KerasTallyQAData, train_steps: int):
        super().__init__()
        self.data = data
        self.train_steps = int(train_steps)

    def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
        self.data.set_train_steps_per_epoch(self.train_steps)

    def on_epoch_begin(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        self.data.set_train_epoch(epoch)
        if str(self.data.cfg.data.get("train_sampling", "natural")) == "prompt_class_tempered":
            print(
                json.dumps(
                    {
                        "event": "keras_prompt_sampling_epoch",
                        "epoch": int(epoch),
                        "prompt_class_sampling_temperature": self.data.prompt_sampling_temperature(),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )


class StudentWeightCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, student: tf.keras.Model, filepath: Path, monitor: str, mode: str):
        super().__init__()
        self.student = student
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.best = np.inf if mode == "min" else -np.inf
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        logs = logs or {}
        value = logs.get(self.monitor)
        if value is None:
            return
        improved = float(value) < self.best if self.mode == "min" else float(value) > self.best
        if improved:
            self.best = float(value)
            self.student.save_weights(self.filepath)


def export_float_tflite(model: tf.keras.Model, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite = converter.convert()
    output.write_bytes(tflite)


def representative_dataset(
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
        yield {
            "token_ids": token_ids.astype(np.int32),
            "images": image.astype(np.float32),
        }


def export_ptq_tflite(
    model: tf.keras.Model,
    output: Path,
    data: KerasTallyQAData,
    cfg: DictConfig,
    representative_samples: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(
        data,
        representative_samples,
        cfg,
    )
    if bool(cfg.export.quantization.get("full_integer", False)):
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        input_type = str(cfg.export.quantization.get("inference_input_type", "uint8"))
        output_type = str(cfg.export.quantization.get("inference_output_type", "int8"))
        converter.inference_input_type = getattr(tf, input_type)
        converter.inference_output_type = getattr(tf, output_type)
    tflite = converter.convert()
    output.write_bytes(tflite)


def quantize_tflite_input(value: np.ndarray, detail: dict[str, Any]) -> np.ndarray:
    dtype = detail["dtype"]
    if np.issubdtype(dtype, np.floating):
        return value.astype(dtype)
    if dtype == np.int32:
        return value.astype(np.int32)
    scale, zero_point = detail.get("quantization", (0.0, 0))
    if not scale:
        return value.astype(dtype)
    quantized = np.round(value / float(scale) + int(zero_point))
    info = np.iinfo(dtype)
    return np.clip(quantized, info.min, info.max).astype(dtype)


def dequantize_tflite_output(value: np.ndarray, detail: dict[str, Any]) -> np.ndarray:
    if np.issubdtype(value.dtype, np.floating):
        return value.astype(np.float32)
    scale, zero_point = detail.get("quantization", (0.0, 0))
    if not scale:
        return value.astype(np.float32)
    return (value.astype(np.float32) - int(zero_point)) * float(scale)


def serialize_tflite_tensor_detail(detail: dict[str, Any]) -> dict[str, Any]:
    scale, zero_point = detail.get("quantization", (0.0, 0))
    quantization_parameters = detail.get("quantization_parameters", {})
    return {
        "name": str(detail.get("name", "")),
        "index": int(detail["index"]),
        "shape": [int(value) for value in detail.get("shape", [])],
        "shape_signature": [
            int(value) for value in detail.get("shape_signature", detail.get("shape", []))
        ],
        "dtype": str(np.dtype(detail["dtype"])),
        "quantization": [float(scale), int(zero_point)],
        "quantization_parameters": {
            "scales": [
                float(value)
                for value in np.asarray(quantization_parameters.get("scales", []))
                .reshape(-1)
                .tolist()
            ],
            "zero_points": [
                int(value)
                for value in np.asarray(quantization_parameters.get("zero_points", []))
                .reshape(-1)
                .tolist()
            ],
            "quantized_dimension": int(
                quantization_parameters.get("quantized_dimension", 0) or 0
            ),
        },
    }


def inspect_tflite_model(path: Path) -> dict[str, Any]:
    interpreter = tf.lite.Interpreter(
        model_path=str(path),
        experimental_delegates=[],
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    )
    interpreter.allocate_tensors()
    try:
        ops = [
            {
                "index": int(index),
                "op_name": str(op.get("op_name", "")),
                "inputs": [int(value) for value in op.get("inputs", [])],
                "outputs": [int(value) for value in op.get("outputs", [])],
            }
            for index, op in enumerate(interpreter._get_ops_details())  # noqa: SLF001
        ]
    except AttributeError:
        ops = []
    return {
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "inputs": [
            serialize_tflite_tensor_detail(detail)
            for detail in interpreter.get_input_details()
        ],
        "outputs": [
            serialize_tflite_tensor_detail(detail)
            for detail in interpreter.get_output_details()
        ],
        "operators": ops,
        "operator_counts": {
            op_name: sum(1 for op in ops if op["op_name"] == op_name)
            for op_name in sorted({op["op_name"] for op in ops})
        },
    }


def map_tflite_inputs(
    details: list[dict[str, Any]],
    inputs: dict[str, tf.Tensor],
) -> dict[int, np.ndarray]:
    mapped: dict[int, np.ndarray] = {}
    token_ids = inputs["token_ids"].numpy()
    images = inputs["images"].numpy()
    for detail in details:
        name = str(detail.get("name", "")).lower()
        shape = list(detail.get("shape_signature", detail.get("shape", [])))
        rank = len(shape)
        if "token" in name or (rank == 2 and detail["dtype"] == np.int32):
            mapped[int(detail["index"])] = token_ids
        elif "image" in name or rank == 4:
            mapped[int(detail["index"])] = images
        else:
            raise ValueError(
                "Could not map TFLite input "
                f"{detail.get('name')} with shape {detail.get('shape')} and dtype {detail['dtype']}."
            )
    return mapped


def evaluate_tflite_split_metrics(
    tflite_path: Path,
    dataset: tf.data.Dataset,
    steps: int,
    num_classes: int,
    data: KerasTallyQAData,
    description: str | None = None,
) -> MulticlassAccumulator:
    interpreter = tf.lite.Interpreter(
        model_path=str(tflite_path),
        experimental_delegates=[],
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    )
    interpreter.allocate_tensors()
    accumulator = MulticlassAccumulator(num_classes)
    iterator = iter(dataset)
    progress = tqdm(
        range(steps),
        desc=description,
        unit="batch",
        leave=False,
        disable=description is None,
    )
    for _ in progress:
        inputs, targets = next(iterator)
        input_details = interpreter.get_input_details()
        raw_inputs = map_tflite_inputs(input_details, inputs)
        resized = False
        for detail in input_details:
            tensor = raw_inputs[int(detail["index"])]
            current_shape = list(detail["shape"])
            desired_shape = list(tensor.shape)
            if current_shape != desired_shape:
                interpreter.resize_tensor_input(int(detail["index"]), desired_shape, strict=False)
                resized = True
        if resized:
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            raw_inputs = map_tflite_inputs(input_details, inputs)
        for detail in input_details:
            tensor = raw_inputs[int(detail["index"])]
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
        labels = targets["labels"].numpy()
        dataset_indices = targets["dataset_index"].numpy().tolist()
        prompts = [str(data.rows[int(index)]["student_prompt"]) for index in dataset_indices]
        accumulator.update(labels, logits, prompts)
    return accumulator


def metric_comparison_plot(
    float_metrics: dict[str, float],
    quantized_metrics: dict[str, float],
    output: Path,
    baseline_label: str = "Keras float",
    quantized_label: str = "TFLite quantized",
    title: str = "Float vs Quantized Test Metrics",
) -> wandb.Image:
    output.parent.mkdir(parents=True, exist_ok=True)
    metric_keys = [
        "accuracy",
        "within_1_accuracy",
        "class_weighted_accuracy",
        "class_weighted_within_1_accuracy",
        "prompt_class_weighted_accuracy",
        "prompt_class_weighted_within_1_accuracy",
        "prompt_class_output_weighted_accuracy",
        "prompt_class_output_weighted_within_1_accuracy",
        "mae",
        "class_weighted_mae",
        "prompt_class_weighted_mae",
        "prompt_class_output_weighted_mae",
    ]
    labels = [key.replace("_", "\n") for key in metric_keys]
    x = np.arange(len(metric_keys))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    ax.bar(
        x - width / 2,
        [float(float_metrics.get(key, np.nan)) for key in metric_keys],
        width,
        label=baseline_label,
        color="#4c78a8",
    )
    ax.bar(
        x + width / 2,
        [float(quantized_metrics.get(key, np.nan)) for key in metric_keys],
        width,
        label=quantized_label,
        color="#f58518",
    )
    ax.set_xticks(x, labels=labels)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    payload = wandb.Image(fig)
    plt.close(fig)
    return payload


def metric_deltas(
    baseline: dict[str, float] | None,
    candidate: dict[str, float] | None,
) -> dict[str, float] | None:
    if baseline is None or candidate is None:
        return None
    return {
        key: float(candidate[key]) - float(baseline[key])
        for key in sorted(set(baseline).intersection(candidate))
        if np.isscalar(baseline[key]) and np.isscalar(candidate[key])
    }


def normalize_initial_weights_load_stage(value: str) -> str:
    normalized = value.replace("-", "_").lower()
    aliases = {
        "before_qat": "before_quantization",
        "pre_qat": "before_quantization",
        "float": "before_quantization",
        "after_qat": "after_quantization",
        "post_qat": "after_quantization",
        "qat": "after_quantization",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"before_quantization", "after_quantization"}:
        raise ValueError(
            "paths.initial_weights_load_stage must be one of "
            "before_quantization, after_quantization, before_qat, or after_qat."
        )
    return normalized


def load_initial_weights_if_configured(
    model: tf.keras.Model,
    initial_weights: Path | None,
    cfg: DictConfig,
) -> None:
    if initial_weights is None:
        return
    if not initial_weights.exists():
        raise FileNotFoundError(f"Initial Keras weights not found: {initial_weights}")
    model.load_weights(
        initial_weights,
        skip_mismatch=bool(cfg.paths.get("initial_weights_skip_mismatch", False)),
    )


@hydra.main(version_base=None, config_path="../conf", config_name="tallyqa_keras_student")
def main(cfg: DictConfig) -> None:
    quantization_mode = str(cfg.export.quantization.mode)
    if quantization_mode not in {"none", "ptq", "qat"}:
        raise ValueError("export.quantization.mode must be one of: none, ptq, qat.")
    tf.keras.utils.set_random_seed(int(cfg.seed))
    load_dotenv(absolute_path(cfg.paths.wandb_env_file), override=False)
    data = make_data(cfg)
    prompt_length = int(data.prompt_token_ids.shape[1])
    train_steps = inferred_steps(data, "train", cfg)
    val_steps = inferred_steps(data, "val", cfg)
    test_steps = inferred_steps(data, "test", cfg)
    data.set_train_steps_per_epoch(train_steps)
    total_train_steps = max(1, train_steps * int(cfg.trainer.max_epochs))

    initial_weights = (
        absolute_path(cfg.paths.initial_weights)
        if cfg.paths.get("initial_weights", None) is not None
        else None
    )
    initial_weights_load_stage = normalize_initial_weights_load_stage(
        str(cfg.paths.get("initial_weights_load_stage", "after_quantization"))
    )

    student = build_keras_student_model(cfg, data.embedding_rows, prompt_length)
    if initial_weights_load_stage == "before_quantization":
        load_initial_weights_if_configured(student, initial_weights, cfg)
    student = maybe_apply_qat(student, cfg)
    quantization_coverage = assert_qat_coverage(student, cfg)
    class_weights = class_weights_from_config(cfg, data)
    kl_weights = (
        np.asarray([float(weight) for weight in cfg.distillation.kl_class_weights], dtype=np.float32)
        if cfg.distillation.get("kl_class_weights", None) is not None
        else class_weights
    )
    model = DistilledStudent(
        student=student,
        alpha=float(cfg.distillation.alpha),
        beta=float(cfg.distillation.beta),
        temperature=float(cfg.distillation.temperature),
        target_distribution=str(cfg.distillation.target_distribution),
        local_soft_sigma=float(cfg.distillation.local_soft_sigma),
        local_soft_radius=int(cfg.distillation.local_soft_radius),
        class_weights=class_weights,
        kl_class_weights=kl_weights,
        image_learning_rate_scale=float(cfg.trainer.get("image_learning_rate_scale", 1.0)),
        image_learning_rate_scale_schedule=str(
            cfg.trainer.get("image_learning_rate_scale_schedule", "constant")
        ),
        image_learning_rate_scale_warmup_steps=int(
            cfg.trainer.get("image_learning_rate_scale_warmup_steps", 1500)
        ),
    )
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=keras_learning_rate(cfg, total_train_steps),
            weight_decay=float(cfg.optimizer.weight_decay),
        ),
        steps_per_execution=int(cfg.trainer.get("steps_per_execution", 1)),
    )
    if initial_weights_load_stage == "after_quantization":
        load_initial_weights_if_configured(student, initial_weights, cfg)

    run_name = str(cfg.experiment.run_name)
    report_dir = absolute_path(cfg.paths.report_dir)
    run_report_dir = report_dir / run_name
    ckpt_dir = absolute_path(cfg.paths.checkpoint_dir) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    run_report_dir.mkdir(parents=True, exist_ok=True)
    counts = parameter_counts(student)

    readable_model_reports = write_model_readable_reports(
        student,
        run_report_dir / f"{run_name}_model",
    )
    visualkeras_report = maybe_write_visualkeras(
        student,
        run_report_dir / f"{run_name}_visualkeras.png",
    )
    mobilenet_cut_shape = infer_mobilenet_cut_shape(cfg)
    fusion_head_model = (
        build_fusion_head_visualization_model(
            cfg,
            data.embedding_rows,
            prompt_length,
            mobilenet_cut_shape,
        )
        if mobilenet_cut_shape is not None
        else None
    )
    fusion_head_visualkeras_report = (
        maybe_write_visualkeras(
            fusion_head_model,
            run_report_dir / f"{run_name}_fusion_head_visualkeras.png",
        )
        if fusion_head_model is not None
        else {"enabled": False, "status": "not_available"}
    )
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
        "run_name": run_name,
        "dataset": {
            "root": str(absolute_path(cfg.paths.dataset_root)),
            "prompt_embeddings": str(absolute_path(cfg.paths.prompt_embeddings)),
            "teacher_cache": (
                str(absolute_path(cfg.paths.teacher_cache))
                if cfg.paths.teacher_cache
                else None
            ),
            "split_sizes": data.split_sizes(),
            "full_split_sizes": data.full_split_sizes(),
            "teacher_cache_coverage": data.cache_coverage(),
            "classes": int(cfg.model.num_outputs),
            "collapse_at": int(cfg.data.collapse_at),
            "prompt_embedding_rows": list(data.embedding_rows.shape),
            "prompt_token_shape": list(data.prompt_token_ids.shape),
        },
        "model": compatibility_report(student, cfg),
        "initial_weights": str(initial_weights) if initial_weights is not None else None,
        "initial_weights_skip_mismatch": bool(
            cfg.paths.get("initial_weights_skip_mismatch", False)
        ),
        "initial_weights_load_stage": initial_weights_load_stage,
        "visualization": {
            "visualkeras": visualkeras_report,
            "fusion_head_visualkeras": fusion_head_visualkeras_report,
            "readable_model_reports": readable_model_reports,
        },
        "quantization": {
            "mode": quantization_mode,
            "qat_coverage": quantization_coverage,
            "representative_indices": data.representative_indices(
                max_samples=int(cfg.export.quantization.representative_samples),
                strategy=str(cfg.export.quantization.get("representative_strategy", "prompt_tempered")),
                prompt_temperature=float(
                    cfg.export.quantization.get("representative_prompt_temperature", 0.5)
                ),
                min_per_prompt=int(cfg.export.quantization.get("representative_min_per_prompt", 4)),
                min_per_output_class=int(
                    cfg.export.quantization.get("representative_min_per_output_class", 32)
                ),
            ),
            "representative_sampling": OmegaConf.to_container(
                cfg.export.quantization,
                resolve=True,
            ),
        },
    }
    report["model"]["parameter_counts"] = counts
    report_path = run_report_dir / f"{run_name}_architecture.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    split_rows = [
        {
            "split": split,
            "active": report["dataset"]["split_sizes"][split],
            "full": report["dataset"]["full_split_sizes"][split],
        }
        for split in ["train", "val", "test"]
    ]
    count_rows = [{"scope": key, "parameters": value} for key, value in counts.items()]
    print("Dataset splits")
    print(format_table(split_rows, ["split", "active", "full"]))
    print("\nParameter counts")
    print(format_table(count_rows, ["scope", "parameters"]))
    print(f"\nModel summary: {readable_model_reports['summary_txt']}")
    print(f"Layer parameter table: {readable_model_reports['layer_table_txt']}")
    print(f"Full architecture report: {report_path}")
    if quantization_mode == "qat":
        print(
            "QAT coverage: "
            f"{quantization_coverage['wrapped_quantizable_leaf_layers']}/"
            f"{quantization_coverage['quantizable_leaf_layers']} quantizable leaf layers wrapped "
            f"({quantization_coverage['wrapped_fraction']:.3f})"
        )
    if bool(cfg.trainer.get("preflight_only", False)):
        print("Preflight-only run complete; exiting before W&B init, training, eval, and export.")
        return

    wandb.init(
        project=str(cfg.wandb.project),
        entity=str(cfg.wandb.entity) if cfg.wandb.entity else None,
        name=run_name,
        mode=str(cfg.wandb.mode),
        tags=list(cfg.wandb.tags),
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    wandb.config.update(
        {
            "parameter_counts": counts,
            "split_sizes": data.split_sizes(),
            "full_split_sizes": data.full_split_sizes(),
            "teacher_cache_coverage": data.cache_coverage(),
            "keras_parameter_count": student.count_params(),
            "initial_weights": str(initial_weights) if initial_weights is not None else None,
            "initial_weights_load_stage": initial_weights_load_stage,
            "qat_wrapped_fraction": float(quantization_coverage["wrapped_fraction"]),
            "qat_unwrapped_quantizable_leaf_layers": int(
                quantization_coverage["unwrapped_quantizable_leaf_layers"]
            ),
        },
        allow_val_change=True,
    )
    if bool(cfg.wandb.get("watch", {}).get("enabled", False)):
        try:
            wandb.watch(
                student,
                log=str(cfg.wandb.watch.get("log", "all")),
                log_freq=int(cfg.wandb.watch.get("log_freq", 100)),
                log_graph=bool(cfg.wandb.watch.get("log_graph", False)),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: wandb.watch failed for Keras model: {exc}")
    wandb.define_metric("trainer/global_step")
    for metric_pattern in [
        "train/*",
        "val/*",
        "test/*",
        "test_float_tflite/*",
        "test_quantized/*",
        "model/*",
        "weights/*",
        "gradients/*",
        "wandb_watch/*",
        "val_plots/*",
        "test_plots/*",
        "test_quantized_plots/*",
    ]:
        wandb.define_metric(metric_pattern, step_metric="trainer/global_step")
    summary_text = Path(readable_model_reports["summary_txt"]).read_text(encoding="utf-8")
    layer_table_text = Path(readable_model_reports["layer_table_txt"]).read_text(encoding="utf-8")
    model_log_payload: dict[str, Any] = {
        "trainer/global_step": 0,
        "model/architecture": wandb.Html(f"<pre>{html.escape(summary_text)}</pre>"),
        "model/layer_parameter_table": wandb.Html(
            f"<pre>{html.escape(layer_table_text)}</pre>"
        ),
    }
    parameter_plot_path = Path(readable_model_reports["parameter_plot_png"])
    if parameter_plot_path.exists():
        model_log_payload["model/layer_parameter_bars"] = wandb.Image(str(parameter_plot_path))
    visualkeras_path = Path(str(visualkeras_report.get("output", "")))
    fusion_head_visualkeras_path = Path(str(fusion_head_visualkeras_report.get("output", "")))
    if visualkeras_report.get("status") == "written" and visualkeras_path.exists():
        model_log_payload["model/visualkeras"] = wandb.Image(str(visualkeras_path))
    if (
        fusion_head_visualkeras_report.get("status") == "written"
        and fusion_head_visualkeras_path.exists()
    ):
        model_log_payload["model/fusion_head_visualkeras"] = wandb.Image(
            str(fusion_head_visualkeras_path)
        )
    wandb.log(model_log_payload)
    save_wandb_file(Path(report_path), policy="now")
    for artifact_path in readable_model_reports.values():
        if Path(artifact_path).exists():
            save_wandb_file(Path(artifact_path), policy="now")
    if visualkeras_path.exists():
        save_wandb_file(Path(visualkeras_path), policy="now")
    if fusion_head_visualkeras_path.exists():
        save_wandb_file(Path(fusion_head_visualkeras_path), policy="now")
    log_wandb_artifact(
        f"{run_name}-model-report",
        "model-report",
        [
            report_path,
            *(Path(path) for path in readable_model_reports.values()),
            visualkeras_path,
            fusion_head_visualkeras_path,
        ],
    )

    train_ds = make_tf_dataset(data, "train", cfg, prompt_length).repeat().prefetch(tf.data.AUTOTUNE)
    val_ds = make_tf_dataset(data, "val", cfg, prompt_length).repeat().prefetch(tf.data.AUTOTUNE)
    test_ds = make_tf_dataset(data, "test", cfg, prompt_length).repeat().prefetch(tf.data.AUTOTUNE)

    checkpoint_callback = StudentWeightCheckpoint(
        student=student,
        filepath=ckpt_dir / "best.weights.h5",
        monitor=str(cfg.trainer.early_stopping.get("monitor", "val_loss")).replace("/", "_"),
        mode=str(cfg.trainer.early_stopping.get("mode", "min")),
    )
    callbacks: list[tf.keras.callbacks.Callback] = [
        KerasDataEpochCallback(data=data, train_steps=train_steps),
        TqdmKerasProgress(train_steps=train_steps, val_steps=val_steps),
        WandbEvaluationLogger(
            train_dataset=train_ds,
            train_steps=train_steps,
            val_dataset=val_ds,
            val_steps=val_steps,
            num_classes=int(cfg.model.num_outputs),
            output_dir=run_report_dir,
            data=data,
            example_samples=(
                int(cfg.validation_plots.samples)
                if bool(cfg.validation_plots.enabled)
                else 0
            ),
            example_every_n_epochs=int(cfg.validation_plots.every_n_epochs),
            log_train_metrics=bool(cfg.trainer.get("log_train_eval_metrics", True)),
        ),
        WandbKerasLogger(log_every_n_steps=int(cfg.trainer.log_every_n_steps)),
        checkpoint_callback,
    ]
    if bool(cfg.wandb.get("watch", {}).get("enabled", False)):
        callbacks.append(
            WandbKerasWeightsAndGradients(
                sample_dataset=train_ds,
                log=str(cfg.wandb.watch.get("log", "all")),
                log_freq=int(cfg.wandb.watch.get("log_freq", 100)),
            )
        )
    if bool(cfg.trainer.get("early_stopping", {}).get("enabled", False)):
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=str(cfg.trainer.early_stopping.get("monitor", "val_loss")).replace("/", "_"),
                mode=str(cfg.trainer.early_stopping.get("mode", "min")),
                patience=int(cfg.trainer.early_stopping.get("patience", 3)),
                min_delta=float(cfg.trainer.early_stopping.get("min_delta", 0.0)),
                restore_best_weights=True,
            )
        )

    if bool(cfg.trainer.get("skip_fit", False)):
        history_dict: dict[str, list[float]] = {}
    else:
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=int(cfg.trainer.max_epochs),
            steps_per_epoch=train_steps,
            validation_steps=val_steps,
            callbacks=callbacks,
            verbose=0,
        )
        history_dict = {
            key: [float(value) for value in values] for key, values in history.history.items()
        }
    test_results = model.evaluate(
        test_ds,
        steps=test_steps,
        return_dict=True,
        verbose=0,
    )
    test_accumulator = evaluate_split_metrics(
        model,
        test_ds,
        test_steps,
        int(cfg.model.num_outputs),
        data,
        description="test metrics",
    )
    test_metric_results = test_accumulator.metrics()
    test_results.update(test_metric_results)
    if int(test_accumulator.confusion.sum()) > 0:
        test_confusion_path = save_confusion_matrix_plot(
            "test",
            test_accumulator,
            run_report_dir / "test_plots" / "confusion_matrix.png",
        )
        final_global_step = int(model.optimizer.iterations.numpy())
        test_plot_payload: dict[str, Any] = {
            "trainer/global_step": final_global_step,
            "test_plots/confusion_matrix": wandb.Image(str(test_confusion_path)),
            **{f"test/{key}": float(value) for key, value in test_metric_results.items()},
        }
        test_example_samples = (
            int(cfg.validation_plots.samples)
            if bool(cfg.validation_plots.enabled)
            else 0
        )
        if test_example_samples > 0:
            test_activation_path = save_activation_examples_plot(
                "test",
                model,
                data,
                run_report_dir / "test_plots" / "image_encoding.png",
                test_example_samples,
            )
            if test_activation_path is not None:
                test_plot_payload["test_plots/image_encoding"] = wandb.Image(
                    str(test_activation_path)
                )
                test_plot_payload["test_plots/image_encoding_count"] = test_example_samples
                save_wandb_file(test_activation_path, policy="now")
            test_examples_path = save_prediction_examples_plot(
                "test",
                model,
                data,
                run_report_dir / "test_plots" / "prediction_examples.png",
                test_example_samples,
            )
            if test_examples_path is not None:
                test_plot_payload["test_plots/example_predictions"] = wandb.Image(
                    str(test_examples_path)
                )
                save_wandb_file(test_examples_path, policy="now")
        wandb.log(test_plot_payload)
        save_wandb_file(Path(test_confusion_path), policy="now")
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": run_name,
        "best_model_monitor": checkpoint_callback.monitor,
        "best_model_mode": checkpoint_callback.mode,
        "best_model_path": str(checkpoint_callback.filepath),
        "best_model_score": (
            float(checkpoint_callback.best)
            if np.isfinite(float(checkpoint_callback.best))
            else None
        ),
        "split_sizes": data.split_sizes(),
        "full_split_sizes": data.full_split_sizes(),
        "teacher_cache_coverage": data.cache_coverage(),
        "initial_weights": str(initial_weights) if initial_weights is not None else None,
        "initial_weights_load_stage": initial_weights_load_stage,
        "fit_skipped": bool(cfg.trainer.get("skip_fit", False)),
        "history": history_dict,
        "test_results": {key: float(value) for key, value in test_results.items()},
        "float_tflite_test_results": None,
        "quantized_test_results": None,
        "metric_deltas": {},
        "checkpoint": str(ckpt_dir / "best.weights.h5"),
        "quantization_mode": quantization_mode,
        "qat_coverage": quantization_coverage,
        "tflite": {},
        "tflite_inspection": {},
    }
    if checkpoint_callback.filepath.exists():
        save_wandb_file(Path(checkpoint_callback.filepath), policy="now")

    if bool(cfg.export.export_tflite):
        float_tflite_path = export_path_for_run(str(cfg.export.tflite_float), run_name, cfg)
        quantized_tflite_path = export_path_for_run(
            str(cfg.export.tflite_quantized),
            run_name,
            cfg,
        )
        export_float_tflite(student, float_tflite_path)
        result["tflite"]["float"] = str(float_tflite_path)
        result["tflite_inspection"]["float"] = inspect_tflite_model(float_tflite_path)
        save_wandb_file(Path(float_tflite_path), policy="now")
        float_tflite_accumulator = evaluate_tflite_split_metrics(
            float_tflite_path,
            test_ds,
            test_steps,
            int(cfg.model.num_outputs),
            data,
            description="float TFLite test metrics",
        )
        float_tflite_metrics = float_tflite_accumulator.metrics()
        result["float_tflite_test_results"] = {
            key: float(value) for key, value in float_tflite_metrics.items()
        }
        result["metric_deltas"]["float_tflite_minus_keras"] = metric_deltas(
            test_metric_results,
            float_tflite_metrics,
        )
        wandb.log(
            {
                "trainer/global_step": int(model.optimizer.iterations.numpy()),
                **{
                    f"test_float_tflite/{key}": float(value)
                    for key, value in float_tflite_metrics.items()
                },
            }
        )
        if quantization_mode in {"ptq", "qat"}:
            export_ptq_tflite(
                student,
                quantized_tflite_path,
                data,
                cfg,
                int(cfg.export.quantization.representative_samples),
            )
            result["tflite"]["quantized"] = str(quantized_tflite_path)
            result["tflite_inspection"]["quantized"] = inspect_tflite_model(
                quantized_tflite_path
            )
            save_wandb_file(Path(quantized_tflite_path), policy="now")
            quantized_accumulator = evaluate_tflite_split_metrics(
                quantized_tflite_path,
                test_ds,
                test_steps,
                int(cfg.model.num_outputs),
                data,
                description="quantized test metrics",
            )
            quantized_metrics = quantized_accumulator.metrics()
            result["quantized_test_results"] = {
                key: float(value) for key, value in quantized_metrics.items()
            }
            result["metric_deltas"]["quantized_minus_keras"] = metric_deltas(
                test_metric_results,
                quantized_metrics,
            )
            result["metric_deltas"]["quantized_minus_float_tflite"] = metric_deltas(
                float_tflite_metrics,
                quantized_metrics,
            )
            comparison_plot_path = (
                run_report_dir / "test_quantized_plots" / "float_vs_quantized_metrics.png"
            )
            quantized_confusion_path = save_confusion_matrix_plot(
                "quantized test",
                quantized_accumulator,
                run_report_dir / "test_quantized_plots" / "confusion_matrix.png",
            )
            wandb.log(
                {
                    "trainer/global_step": int(model.optimizer.iterations.numpy()),
                    "test_quantized_plots/confusion_matrix": wandb.Image(str(quantized_confusion_path)),
                    "test_plots/float_vs_quantized_metrics": metric_comparison_plot(
                        test_metric_results,
                        quantized_metrics,
                        comparison_plot_path,
                        baseline_label=(
                            "Keras QAT simulated"
                            if quantization_mode == "qat"
                            else "Keras float"
                        ),
                        title=(
                            "QAT-Simulated vs TFLite Quantized Test Metrics"
                            if quantization_mode == "qat"
                            else "Float vs Quantized Test Metrics"
                        ),
                    ),
                    **{f"test_quantized/{key}": float(value) for key, value in quantized_metrics.items()},
                }
            )
            save_wandb_file(Path(comparison_plot_path), policy="now")
            save_wandb_file(Path(quantized_confusion_path), policy="now")
    result_path = run_report_dir / f"{run_name}_results.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    save_wandb_file(Path(result_path), policy="now")
    if checkpoint_callback.filepath.exists():
        log_wandb_artifact(
            f"{run_name}-chosen-test-checkpoint",
            "model-checkpoint",
            [
                Path(checkpoint_callback.filepath),
                result_path,
                report_path,
            ],
            aliases=["best", "test-evaluated", checkpoint_callback.monitor],
        )
    final_artifact_paths: list[Path] = [
        result_path,
        report_path,
        *(Path(path) for path in readable_model_reports.values()),
        visualkeras_path,
        fusion_head_visualkeras_path,
        *run_report_dir.rglob("*.png"),
    ]
    checkpoint_path = Path(str(checkpoint_callback.filepath))
    if checkpoint_path.exists():
        final_artifact_paths.append(checkpoint_path)
    for tflite_path in result.get("tflite", {}).values():
        final_artifact_paths.append(Path(str(tflite_path)))
    log_wandb_artifact(
        f"{run_name}-outputs",
        "training-output",
        final_artifact_paths,
        base_path=run_report_dir,
    )
    wandb.finish()


if __name__ == "__main__":
    main()
