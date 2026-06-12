#!/usr/bin/env python3
"""Evaluate a MAX78000 TallyQA checkpoint and write W&B-ready plots/artifacts."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import textwrap
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode


CLASS_NAMES = ["0", "1", "2", "3", "4", "5+"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ai8x-training", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-file", type=Path, default=None)
    parser.add_argument("--factory", required=True)
    parser.add_argument("--input-channels", type=int, default=588)
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def import_max_modules(ai8x_training: Path, model_file: Path | None):
    sys.path.insert(0, str(ai8x_training.resolve()))
    sys.path.insert(0, str((ai8x_training / "models").resolve()))
    sys.path.insert(0, str((ai8x_training / "datasets").resolve()))
    import ai8x  # type: ignore

    ai8x.set_device(85, simulate=False, round_avg=False, verbose=False)
    model_path = model_file or ai8x_training / "models" / "ai85net-tallyqa-mbv3-small.py"
    dataset_path = ai8x_training / "datasets" / "tallyqa_count.py"
    model_module = load_module("edge_vlm_max_model_eval", model_path)
    dataset_module = load_module("edge_vlm_max_dataset_eval", dataset_path)
    return ai8x, model_module, dataset_module


def build_transform(ai8x, dataset_module):
    return transforms.Compose(
        [
            transforms.Resize(
                (dataset_module.RESIZE_SIZE, dataset_module.RESIZE_SIZE),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            dataset_module.Fold2x2(),
            ai8x.normalize(args=SimpleNamespace(act_mode_8bit=False)),
        ]
    )


def build_dataset(ai8x, dataset_module, data_dir: Path, split: str, prompt_channels: int):
    return dataset_module.TallyQACount(
        root_dir=data_dir,
        d_type=split,
        transform=build_transform(ai8x, dataset_module),
        seed=0,
        prompt_embedding_channels=prompt_channels,
    )


def hard_labels(target: torch.Tensor) -> torch.Tensor:
    if target.ndim == 2 and target.size(1) > 1 and torch.is_floating_point(target):
        return target[:, 0].long()
    return target.long()


def load_checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint payload in {path}")
    return {
        str(key).removeprefix("module."): value
        for key, value in state.items()
        if torch.is_tensor(value)
    }


def build_model(model_module, factory: str, input_channels: int, num_classes: int):
    model_factory = getattr(model_module, factory)
    return model_factory(
        pretrained=False,
        num_classes=num_classes,
        num_channels=input_channels,
        dimensions=(56, 56),
        bias=True,
    )


def display_image(dataset, index: int) -> np.ndarray:
    record = dataset.records[index]
    image_chw = np.asarray(dataset.images[int(record["image_index"])])
    image_hwc = np.transpose(image_chw, (1, 2, 0))
    return image_hwc.astype(np.uint8)


def normalize_map(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    minimum = float(np.nanmin(values))
    maximum = float(np.nanmax(values))
    if not np.isfinite(minimum) or not np.isfinite(maximum) or maximum <= minimum:
        return np.zeros_like(values, dtype=np.float32)
    return (values - minimum) / (maximum - minimum)


class Accumulator:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.absolute_errors: list[int] = []
        self.prompt_errors: dict[str, list[int]] = defaultdict(list)
        self.prompt_correct: dict[str, list[bool]] = defaultdict(list)

    def add(self, labels: np.ndarray, predictions: np.ndarray, prompts: list[str]) -> None:
        for label, prediction, prompt in zip(labels, predictions, prompts, strict=True):
            label_i = int(label)
            pred_i = int(prediction)
            self.confusion[label_i, pred_i] += 1
            error = abs(pred_i - label_i)
            self.absolute_errors.append(error)
            self.prompt_errors[prompt].append(error)
            self.prompt_correct[prompt].append(label_i == pred_i)

    def metrics(self) -> dict[str, float]:
        total = int(self.confusion.sum())
        correct = int(np.trace(self.confusion))
        labels = np.arange(self.num_classes)
        row_totals = self.confusion.sum(axis=1)
        class_acc = np.divide(
            np.diag(self.confusion),
            np.clip(row_totals, a_min=1, a_max=None),
        )
        within_1 = 0
        for row in range(self.num_classes):
            for col in range(self.num_classes):
                if abs(row - col) <= 1:
                    within_1 += int(self.confusion[row, col])
        class_mae = []
        for row in range(self.num_classes):
            if row_totals[row] <= 0:
                continue
            class_mae.append(
                float(
                    sum(abs(row - col) * self.confusion[row, col] for col in labels)
                    / row_totals[row]
                )
            )
        prompt_acc = [
            float(np.mean(values))
            for values in self.prompt_correct.values()
            if values
        ]
        prompt_mae = [
            float(np.mean(values))
            for values in self.prompt_errors.values()
            if values
        ]
        return {
            "accuracy": correct / total if total else 0.0,
            "within_1_accuracy": within_1 / total if total else 0.0,
            "class_weighted_accuracy": float(np.mean(class_acc)) if len(class_acc) else 0.0,
            "mae": float(np.mean(self.absolute_errors)) if self.absolute_errors else 0.0,
            "class_weighted_mae": float(np.mean(class_mae)) if class_mae else 0.0,
            "prompt_class_output_weighted_accuracy": float(np.mean(prompt_acc)) if prompt_acc else 0.0,
            "prompt_class_output_weighted_mae": float(np.mean(prompt_mae)) if prompt_mae else 0.0,
        }


def save_confusion(stage: str, accumulator: Accumulator, output: Path) -> Path:
    counts = accumulator.confusion
    row_totals = counts.sum(axis=1, keepdims=True)
    normalized = counts / np.clip(row_totals, a_min=1, a_max=None)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    cmap = plt.get_cmap("magma")
    image = ax.imshow(normalized, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASS_NAMES)), labels=CLASS_NAMES)
    ax.set_yticks(range(len(CLASS_NAMES)), labels=CLASS_NAMES)
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
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def unique_example_indices(dataset, max_samples: int) -> list[int]:
    seen_images: set[str] = set()
    selected: list[int] = []
    for index, record in enumerate(dataset.records):
        image_id = str(record.get("image_id") or record.get("image_index"))
        if image_id in seen_images:
            continue
        seen_images.add(image_id)
        selected.append(index)
        if len(selected) >= max_samples:
            break
    return selected


def save_examples(stage: str, model: torch.nn.Module, dataset, output: Path, max_samples: int, device: str):
    indices = unique_example_indices(dataset, max_samples)
    if not indices:
        return None, None
    rows = []
    images = []
    inputs = []
    labels = []
    prompts = []
    for index in indices:
        image, target = dataset[index]
        record = dataset.records[index]
        inputs.append(image)
        labels.append(int(hard_labels(torch.as_tensor(target).unsqueeze(0))[0]))
        prompts.append(str(record.get("student_prompt", "")))
        images.append(display_image(dataset, index))
        rows.append(record)
    batch = torch.stack(inputs).to(device)
    with torch.no_grad():
        features = model.forward_features(batch).detach().cpu().numpy()
        logits = model(batch).detach().cpu()
        probabilities = F.softmax(logits, dim=1).numpy()
    predictions = np.argmax(probabilities, axis=1)

    height = max(3.2, 2.2 * len(rows))
    fig, axes = plt.subplots(len(rows), 4, figsize=(14.4, height), squeeze=False)
    for row_index, row in enumerate(rows):
        image_ax, map_ax, raw_map_ax, prob_ax = axes[row_index]
        image_ax.imshow(images[row_index])
        title = (
            f"{stage} idx={indices[row_index]} image={row.get('image_id')} "
            f"true={labels[row_index]} pred={int(predictions[row_index])} "
            f"prompt={prompts[row_index]}"
        )
        image_ax.set_title("\n".join(textwrap.wrap(title, width=42)), fontsize=9)
        image_ax.axis("off")

        feature_map = np.squeeze(features[row_index])
        if feature_map.ndim == 3:
            feature_map = feature_map.mean(axis=0)
        map_ax.imshow(normalize_map(feature_map), cmap="magma")
        map_ax.set_title("14x14 head map normalized", fontsize=9)
        map_ax.axis("off")
        raw_map_ax.imshow(feature_map, cmap="magma")
        raw_map_ax.set_title("14x14 head map raw", fontsize=9)
        raw_map_ax.axis("off")

        colors = ["#8a8f98"] * len(CLASS_NAMES)
        colors[labels[row_index]] = "#2b8a3e"
        colors[int(predictions[row_index])] = (
            "#2f6fdd" if int(predictions[row_index]) == labels[row_index] else "#c92a2a"
        )
        prob_ax.bar(CLASS_NAMES, probabilities[row_index], color=colors)
        prob_ax.set_ylim(0, 1)
        prob_ax.set_ylabel("p")
        prob_ax.set_xlabel("count")
        prob_ax.set_title("predicted count distribution", fontsize=9)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)

    metadata = {
        "stage": stage,
        "indices": indices,
        "image_ids": [str(row.get("image_id")) for row in rows],
        "example_ids": [row.get("example_id") for row in rows],
        "labels": labels,
        "predictions": [int(value) for value in predictions],
        "prompts": prompts,
    }
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return output, metadata_path


def evaluate_split(
    stage: str,
    model: torch.nn.Module,
    dataset,
    output_dir: Path,
    batch_size: int,
    samples: int,
    device: str,
) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    accumulator = Accumulator(num_classes=len(CLASS_NAMES))
    model.eval()
    with torch.no_grad():
        offset = 0
        for inputs, target in loader:
            labels = hard_labels(target)
            logits = model(inputs.to(device)).cpu()
            predictions = torch.argmax(logits, dim=1)
            prompts = [
                str(record.get("student_prompt", ""))
                for record in dataset.records[offset : offset + int(labels.numel())]
            ]
            accumulator.add(labels.numpy(), predictions.numpy(), prompts)
            offset += int(labels.numel())
    split_dir = output_dir / f"{stage}_plots"
    confusion_path = save_confusion(stage, accumulator, split_dir / "confusion_matrix.png")
    examples_path, examples_meta = save_examples(
        stage,
        model,
        dataset,
        split_dir / "image_encoding.png",
        samples,
        device,
    )
    return {
        "metrics": accumulator.metrics(),
        "confusion_matrix": str(confusion_path),
        "image_encoding": str(examples_path) if examples_path else None,
        "image_encoding_metadata": str(examples_meta) if examples_meta else None,
        "samples": int(len(dataset)),
        "unique_plot_samples": int(samples),
    }


def main() -> None:
    args = parse_args()
    ai8x, model_module, dataset_module = import_max_modules(args.ai8x_training, args.model_file)
    prompt_channels = max(0, args.input_channels - 12)
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    model = build_model(model_module, args.factory, args.input_channels, args.num_classes)
    state = load_checkpoint_state(args.checkpoint)
    fused_bn_before_load = not any(".bn." in key for key in state)
    if fused_bn_before_load:
        ai8x.fuse_bn_layers(model)
    load_result = model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint),
        "data_dir": str(args.data_dir),
        "factory": args.factory,
        "input_channels": args.input_channels,
        "num_classes": args.num_classes,
        "load_state": {
            "fused_bn_before_load": fused_bn_before_load,
            "missing_keys": list(load_result.missing_keys),
            "unexpected_keys": list(load_result.unexpected_keys),
        },
        "splits": {},
    }
    for split in ("val", "test"):
        dataset = build_dataset(ai8x, dataset_module, args.data_dir, split, prompt_channels)
        results["splits"][split] = evaluate_split(
            split,
            model,
            dataset,
            args.output_dir,
            args.batch_size,
            args.samples,
            device,
        )

    result_path = args.output_dir / "max78000_eval_results.json"
    result_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(result_path)


if __name__ == "__main__":
    main()
