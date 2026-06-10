#!/usr/bin/env python3
"""Train a Keras/TFLite-oriented TallyQA student with teacher distillation."""

from __future__ import annotations

import html
import io
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
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

    def batches(self, split: str) -> Iterable[tuple[dict[str, np.ndarray], dict[str, np.ndarray]]]:
        batch_size = int(self.cfg.data.batch_size)
        indices = self.train_epoch_indices() if split == "train" else list(self.indices[split])
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
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
            yield (
                {
                    "token_ids": self.prompt_token_ids[item_class_ids],
                    "images": images,
                },
                {"labels": labels, "teacher_probs": teacher_probs},
            )


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


class MulticlassAccumulator:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, labels: np.ndarray, logits: np.ndarray) -> None:
        predictions = np.argmax(logits, axis=1)
        for true_label, predicted_label in zip(labels.tolist(), predictions.tolist(), strict=True):
            self.confusion[int(true_label), int(predicted_label)] += 1

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
        }


def class_labels(num_classes: int) -> list[str]:
    if num_classes == 6:
        return ["0", "1", "2", "3", "4", "5+"]
    return [str(index) for index in range(num_classes)]


def confusion_matrix_plot(stage: str, accumulator: MulticlassAccumulator) -> wandb.Image:
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
    payload = wandb.Image(fig)
    plt.close(fig)
    return payload


def evaluate_split_metrics(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    steps: int,
    num_classes: int,
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
        accumulator.update(labels, logits)
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

    token_ids = tf.keras.Input(shape=(prompt_length,), dtype=tf.int32, name="token_ids")
    images = tf.keras.Input(
        shape=(int(cfg.keras_model.image_size), int(cfg.keras_model.image_size), 3),
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


def normformer_block(
    tokens: tf.Tensor,
    fusion_dim: int,
    heads: int,
    mlp_ratio: int,
    dropout: float,
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
    x = tf.keras.layers.Dense(fusion_dim * mlp_ratio, activation="gelu", name=f"fusion_{index}_mlp_up")(x)
    if dropout > 0:
        x = tf.keras.layers.Dropout(dropout, name=f"fusion_{index}_mlp_dropout")(x)
    x = tf.keras.layers.Dense(fusion_dim, name=f"fusion_{index}_mlp_down")(x)
    tokens = tf.keras.layers.Add(name=f"fusion_{index}_mlp_residual")([residual, x])
    return tokens


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
    fusion_depth = int(cfg.keras_model.get("fusion_depth", cfg.model.fusion_depth))
    mlp_ratio = int(cfg.keras_model.get("fusion_mlp_ratio", cfg.model.fusion_mlp_ratio))
    dropout = float(cfg.keras_model.get("dropout", cfg.model.dropout))
    fusion_mode = str(cfg.keras_model.get("fusion_mode", "normformer"))
    use_prompt_identity = bool(cfg.keras_model.get("use_prompt_identity", cfg.model.use_prompt_identity))
    use_image_positional_embeddings = bool(
        cfg.keras_model.get(
            "use_image_positional_embeddings",
            cfg.model.use_image_positional_embeddings,
        )
    )

    token_ids = tf.keras.Input(shape=(prompt_length,), dtype=tf.int32, name="token_ids")
    images = tf.keras.Input(
        shape=(int(cfg.keras_model.image_size), int(cfg.keras_model.image_size), 3),
        dtype=tf.float32,
        name="images",
    )

    pad = np.zeros((1, embedding_rows.shape[1]), dtype=np.float32)
    embedding_init = np.concatenate([pad, embedding_rows.astype(np.float32)], axis=0)
    embedded = tf.keras.layers.Embedding(
        input_dim=embedding_init.shape[0],
        output_dim=embedding_init.shape[1],
        embeddings_initializer=tf.keras.initializers.Constant(embedding_init),
        trainable=not bool(cfg.model.freeze_embeddings),
        mask_zero=True,
        name="compact_prompt_embedding",
    )(token_ids)
    query = tf.keras.layers.GlobalAveragePooling1D(name="mean_prompt_embedding")(embedded)
    query = tf.keras.layers.Dense(fusion_dim, activation="gelu", name="prompt_projection_dense")(query)
    query = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="prompt_projection_norm")(query)
    query_token = tf.keras.layers.Reshape((1, fusion_dim), name="prompt_token")(query)
    if use_prompt_identity:
        query_token = add_prompt_identity(query_token, fusion_dim)

    backbone = build_keras_mobilenet(cfg, images)
    backbone.trainable = not bool(cfg.model.freeze_image_features)
    image_features = backbone(images)
    image_tokens = tf.keras.layers.Conv2D(
        fusion_dim,
        kernel_size=1,
        padding="same",
        activation=None,
        name="image_token_projection",
    )(image_features)
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
        fused = tf.keras.layers.Dense(fusion_dim * mlp_ratio, activation="gelu", name="fusion_mlp_0")(fused)
        if dropout > 0:
            fused = tf.keras.layers.Dropout(dropout, name="fusion_mlp_dropout")(fused)
        fused = tf.keras.layers.Dense(fusion_dim, activation="gelu", name="fusion_mlp_1")(fused)
    elif fusion_mode == "normformer":
        tokens = tf.keras.layers.Concatenate(axis=1, name="prompt_image_tokens")(
            [query_token, image_tokens]
        )
        for index in range(fusion_depth):
            tokens = normformer_block(tokens, fusion_dim, heads, mlp_ratio, dropout, index)
        fused = tf.keras.layers.GlobalAveragePooling1D(name="fusion_token_mean")(tokens)
        fused = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="fusion_output_norm")(fused)
    else:
        raise ValueError("keras_model.fusion_mode must be one of {'normformer', 'mlp'}.")

    logits = tf.keras.layers.Dense(num_classes, name="logits")(fused)
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
        quantized_types = (
            tf.keras.layers.Conv2D,
            tf.keras.layers.DepthwiseConv2D,
            tf.keras.layers.Dense,
        )
        if isinstance(layer, quantized_types):
            return quantize.quantize_annotate_layer(layer)
        return layer

    annotated = tf.keras.models.clone_model(student, clone_function=annotate)
    return quantize.quantize_apply(annotated)


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
    ):
        super().__init__(name="distilled_tallyqa_keras_student")
        self.student = student
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.temperature = float(temperature)
        self.target_distribution = target_distribution
        self.local_soft_sigma = float(local_soft_sigma)
        self.local_soft_radius = int(local_soft_radius)
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
        self.optimizer.apply_gradients(zip(gradients, self.student.trainable_variables, strict=False))
        self.loss_tracker.update_state(loss)
        self.ce_tracker.update_state(ce_loss)
        self.ce_unweighted_tracker.update_state(ce_loss_unweighted)
        self.kl_tracker.update_state(kl_loss)
        self.accuracy.update_state(labels, logits)
        self._update_count_metrics(labels, logits)
        return {metric.name: metric.result() for metric in self.metrics}

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
            "keras_prior": "Replaced with prompt-image concat plus Dense fusion.",
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
            "keras_prior": "Replaced with GlobalAveragePooling2D image branch.",
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
            "Keras W&B logging mirrors scalar metrics, confusion matrices, reports, results, and weights; Lightning-specific wandb.watch graph/gradient logging is not mirrored.",
            "Keras validation image activation plots are not mirrored because the TFLite-prior model does not expose the same spatial token activations as the PyTorch student.",
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

    def on_train_batch_end(self, batch: int, logs: dict[str, Any] | None = None) -> None:
        self.global_train_step += 1
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
            wandb.log(payload, step=self.global_train_step)

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        payload = {
            f"{self.prefix}{self._wandb_key(key)}": float(value)
            for key, value in self._scalar_logs(logs).items()
        }
        learning_rate = self._learning_rate()
        if learning_rate is not None:
            payload[f"{self.prefix}train/lr_epoch"] = learning_rate
        payload["trainer/epoch"] = epoch
        wandb.log(payload)


class WandbEvaluationLogger(tf.keras.callbacks.Callback):
    def __init__(
        self,
        val_dataset: tf.data.Dataset,
        val_steps: int,
        num_classes: int,
    ):
        super().__init__()
        self.val_dataset = val_dataset
        self.val_steps = val_steps
        self.num_classes = num_classes

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        logs = logs if logs is not None else {}
        accumulator = evaluate_split_metrics(
            self.model,
            self.val_dataset,
            self.val_steps,
            self.num_classes,
            description="val metrics",
        )
        for name, value in accumulator.metrics().items():
            logs[f"val_{name}"] = float(value)
        if int(accumulator.confusion.sum()) > 0:
            wandb.log(
                {
                    "val_plots/confusion_matrix": confusion_matrix_plot("val", accumulator),
                    "trainer/epoch": epoch,
                },
                step=epoch + 1,
            )


class TqdmKerasProgress(tf.keras.callbacks.Callback):
    def __init__(self, train_steps: int, val_steps: int):
        super().__init__()
        self.train_steps = train_steps
        self.val_steps = val_steps
        self.epoch_bar: tqdm | None = None
        self.batch_bar: tqdm | None = None

    def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
        epochs = int(self.params.get("epochs", 0) or 0)
        self.epoch_bar = tqdm(total=epochs, desc="epochs", unit="epoch")

    def on_epoch_begin(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        self.batch_bar = tqdm(total=self.train_steps, desc=f"train {epoch + 1}", unit="batch", leave=False)

    def on_train_batch_end(self, batch: int, logs: dict[str, Any] | None = None) -> None:
        if self.batch_bar is None:
            return
        if logs:
            self.batch_bar.set_postfix(
                {
                    key: f"{float(value):.4g}"
                    for key, value in logs.items()
                    if key in {"loss", "ce_loss", "kl_loss", "accuracy", "mae"}
                },
                refresh=False,
            )
        self.batch_bar.update(1)

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
                        if key in {"loss", "val_loss", "accuracy", "val_accuracy", "mae", "val_mae"}
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
) -> Iterable[list[np.ndarray]]:
    quant_cfg = cfg.export.quantization
    for token_ids, image in data.representative_examples(
        max_samples=max_samples,
        strategy=str(quant_cfg.get("representative_strategy", "prompt_tempered")),
        prompt_temperature=float(quant_cfg.get("representative_prompt_temperature", 0.5)),
        min_per_prompt=int(quant_cfg.get("representative_min_per_prompt", 4)),
        min_per_output_class=int(quant_cfg.get("representative_min_per_output_class", 32)),
    ):
        yield [token_ids.astype(np.int32), image.astype(np.float32)]


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

    student = build_keras_student_model(cfg, data.embedding_rows, prompt_length)
    student = maybe_apply_qat(student, cfg)
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
    )
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=keras_learning_rate(cfg, total_train_steps),
            weight_decay=float(cfg.optimizer.weight_decay),
        )
    )

    run_name = str(cfg.experiment.run_name)
    report_dir = absolute_path(cfg.paths.report_dir)
    ckpt_dir = absolute_path(cfg.paths.checkpoint_dir) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    counts = parameter_counts(student)

    visualkeras_report = (
        maybe_write_visualkeras(
            student,
            report_dir / f"{run_name}_visualkeras.png",
        )
        if bool(cfg.keras_model.get("visualkeras", {}).get("enabled", False))
        else {"enabled": False, "status": "disabled"}
    )
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
        "run_name": run_name,
        "dataset": {
            "root": str(absolute_path(cfg.paths.dataset_root)),
            "prompt_embeddings": str(absolute_path(cfg.paths.prompt_embeddings)),
            "teacher_cache": str(absolute_path(cfg.paths.teacher_cache)),
            "split_sizes": data.split_sizes(),
            "full_split_sizes": data.full_split_sizes(),
            "teacher_cache_coverage": data.cache_coverage(),
            "classes": int(cfg.model.num_outputs),
            "collapse_at": int(cfg.data.collapse_at),
            "prompt_embedding_rows": list(data.embedding_rows.shape),
            "prompt_token_shape": list(data.prompt_token_ids.shape),
        },
        "model": compatibility_report(student, cfg),
        "visualization": {
            "visualkeras": visualkeras_report,
        },
        "quantization": {
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
    report_path = report_dir / f"{run_name}_architecture.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["dataset"], indent=2))
    print(json.dumps(counts, indent=2))
    print(f"Full architecture report: {report_path}")

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
        },
        allow_val_change=True,
    )
    summary_buffer = io.StringIO()
    student.summary(print_fn=lambda line: summary_buffer.write(line + "\n"))
    wandb.log({"model/architecture": wandb.Html(f"<pre>{html.escape(summary_buffer.getvalue())}</pre>")})
    wandb.save(str(report_path), policy="now")

    train_ds = make_tf_dataset(data, "train", cfg, prompt_length).prefetch(tf.data.AUTOTUNE)
    val_ds = make_tf_dataset(data, "val", cfg, prompt_length).prefetch(tf.data.AUTOTUNE)
    test_ds = make_tf_dataset(data, "test", cfg, prompt_length).prefetch(tf.data.AUTOTUNE)

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
            val_dataset=val_ds,
            val_steps=val_steps,
            num_classes=int(cfg.model.num_outputs),
        ),
        WandbKerasLogger(log_every_n_steps=int(cfg.trainer.log_every_n_steps)),
        checkpoint_callback,
    ]
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

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(cfg.trainer.max_epochs),
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=0,
    )
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
        description="test metrics",
    )
    test_metric_results = test_accumulator.metrics()
    test_results.update(test_metric_results)
    if int(test_accumulator.confusion.sum()) > 0:
        wandb.log(
            {
                "test_plots/confusion_matrix": confusion_matrix_plot("test", test_accumulator),
                **{f"test/{key}": float(value) for key, value in test_metric_results.items()},
            },
            step=int(cfg.trainer.max_epochs) + 1,
        )
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
        "history": {key: [float(value) for value in values] for key, values in history.history.items()},
        "test_results": {key: float(value) for key, value in test_results.items()},
        "checkpoint": str(ckpt_dir / "best.weights.h5"),
        "quantization_mode": quantization_mode,
    }
    result_path = report_dir / f"{run_name}_results.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    wandb.save(str(result_path), policy="now")
    if checkpoint_callback.filepath.exists():
        wandb.save(str(checkpoint_callback.filepath), policy="now")

    if bool(cfg.export.export_tflite):
        export_float_tflite(student, absolute_path(cfg.export.tflite_float))
        wandb.save(str(absolute_path(cfg.export.tflite_float)), policy="now")
        if quantization_mode in {"ptq", "qat"}:
            export_ptq_tflite(
                student,
                absolute_path(cfg.export.tflite_quantized),
                data,
                cfg,
                int(cfg.export.quantization.representative_samples),
            )
            wandb.save(str(absolute_path(cfg.export.tflite_quantized)), policy="now")
    wandb.finish()


if __name__ == "__main__":
    main()
