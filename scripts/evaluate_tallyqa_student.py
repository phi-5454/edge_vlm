from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import json
import os
from pathlib import Path
from typing import Any

os.environ["MPLBACKEND"] = "Agg"

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from vlm_micro.student.data import TallyQAStudentDataModule
from vlm_micro.student.lightning import TallyQAStudentModule
from vlm_micro.student.model import StudentBaseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a TallyQA student checkpoint and plot class diagnostics."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config-name", default="tallyqa_student_local")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit-batches", type=int, default=None)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def absolute_path(value: str) -> Path:
    return Path(to_absolute_path(value))


def load_config(config_name: str, overrides: list[str]) -> DictConfig:
    conf_dir = Path("conf").resolve()
    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        return compose(config_name=config_name, overrides=overrides)


def build_data(cfg: DictConfig) -> TallyQAStudentDataModule:
    teacher_cache = absolute_path(cfg.paths.teacher_cache) if cfg.paths.teacher_cache else None
    prompt_class_filter_csv = (
        absolute_path(cfg.data.prompt_class_filter_csv)
        if cfg.data.get("prompt_class_filter_csv", None)
        else None
    )
    prompt_class_names_file = (
        absolute_path(cfg.data.prompt_class_names_file)
        if cfg.data.get("prompt_class_names_file", None)
        else None
    )
    curriculum_schedule = (
        absolute_path(cfg.data.curriculum_schedule)
        if cfg.data.get("curriculum_schedule", None)
        else None
    )
    return TallyQAStudentDataModule(
        dataset_root=absolute_path(cfg.paths.dataset_root),
        prompt_embeddings=absolute_path(cfg.paths.prompt_embeddings),
        teacher_cache=teacher_cache,
        batch_size=int(cfg.data.batch_size),
        num_workers=int(cfg.data.num_workers),
        seed=int(cfg.seed),
        tensor_cache_size=int(cfg.data.tensor_cache_size),
        prefetch_factor=int(cfg.data.prefetch_factor),
        persistent_workers=bool(cfg.data.persistent_workers),
        pin_memory=bool(cfg.data.pin_memory),
        group_train_by_image=bool(cfg.data.group_train_by_image),
        shuffle_train=bool(cfg.data.get("shuffle_train", True)),
        train_sampling=str(cfg.data.get("train_sampling", "natural")),
        prompt_class_sampling_temperature=float(
            cfg.data.get("prompt_class_sampling_temperature", 0.5)
        ),
        train_epoch_size=(
            int(cfg.data.train_epoch_size)
            if cfg.data.get("train_epoch_size", None) is not None
            else None
        ),
        shuffle_block_size=int(cfg.data.shuffle_block_size),
        train_example_limit=(
            int(cfg.data.train_example_limit)
            if cfg.data.get("train_example_limit", None) is not None
            else None
        ),
        missing_teacher_policy=str(cfg.data.missing_teacher_policy),
        collapse_at=int(cfg.data.collapse_at),
        num_classes=int(cfg.model.num_outputs),
        prompt_class_filter_csv=prompt_class_filter_csv,
        min_prompt_accuracy=(
            float(cfg.data.min_prompt_accuracy)
            if cfg.data.get("min_prompt_accuracy", None) is not None
            else None
        ),
        prompt_class_names=(
            str(cfg.data.prompt_class_names)
            if cfg.data.get("prompt_class_names", None) is not None
            else None
        ),
        prompt_class_names_file=prompt_class_names_file,
        curriculum_schedule=curriculum_schedule,
    )


def class_weights_from_config(cfg: DictConfig, data: TallyQAStudentDataModule) -> list[float] | None:
    explicit_weights = cfg.distillation.get("class_weights", None)
    weight_mode = cfg.distillation.get("class_weight_mode", None)
    if explicit_weights is not None and weight_mode is not None:
        raise ValueError("Use either distillation.class_weights or class_weight_mode, not both.")
    if explicit_weights is not None:
        return [float(weight) for weight in explicit_weights]
    if weight_mode is None:
        return None
    if str(weight_mode) != "balanced":
        raise ValueError("distillation.class_weight_mode must be null or 'balanced'.")
    counts = data.label_counts("train")
    total = sum(counts.values())
    num_classes = int(cfg.model.num_outputs)
    if total <= 0 or any(counts[class_id] <= 0 for class_id in range(num_classes)):
        raise ValueError(f"Cannot compute balanced class weights from counts: {counts}")
    return [total / (num_classes * counts[class_id]) for class_id in range(num_classes)]


def build_module(cfg: DictConfig, data: TallyQAStudentDataModule) -> TallyQAStudentModule:
    model = StudentBaseline(
        embedding_rows=data.embedding_rows,
        freeze_embeddings=bool(cfg.model.freeze_embeddings),
        freeze_image_features=bool(cfg.model.get("freeze_image_features", False)),
        image_pretrained=bool(cfg.model.image_pretrained),
        query_dim=int(cfg.model.query_dim),
        image_dim=int(cfg.model.image_dim),
        fusion_dim=int(cfg.model.fusion_dim),
        fusion_depth=int(cfg.model.fusion_depth),
        fusion_heads=int(cfg.model.fusion_heads),
        fusion_mlp_ratio=int(cfg.model.fusion_mlp_ratio),
        dropout=float(cfg.model.dropout),
        image_backbone=str(cfg.model.image_backbone),
        image_feature_cutoff=cfg.model.get("image_feature_cutoff", "auto"),
        image_film_at=cfg.model.get("image_film_at", None),
        image_token_mode=str(cfg.model.get("image_token_mode", "spatial")),
        fusion_mode=str(cfg.model.get("fusion_mode", "transformer")),
        use_prompt_identity=bool(cfg.model.get("use_prompt_identity", True)),
        use_image_positional_embeddings=bool(
            cfg.model.get("use_image_positional_embeddings", True)
        ),
        image_position_tokens=int(cfg.model.get("image_position_tokens", 196)),
        num_outputs=int(cfg.model.num_outputs),
    )
    return TallyQAStudentModule(
        model=model,
        alpha=float(cfg.distillation.alpha),
        beta=float(cfg.distillation.beta),
        learning_rate=float(cfg.optimizer.learning_rate),
        warmup_start_learning_rate=float(cfg.optimizer.warmup_start_learning_rate),
        warmup_steps=int(cfg.optimizer.warmup_steps),
        weight_decay=float(cfg.optimizer.weight_decay),
        image_learning_rate_scale=float(cfg.optimizer.get("image_learning_rate_scale", 1.0)),
        temperature=float(cfg.distillation.temperature),
        class_weights=class_weights_from_config(cfg, data),
        kl_class_weights=(
            [float(weight) for weight in cfg.distillation.kl_class_weights]
            if cfg.distillation.get("kl_class_weights", None) is not None
            else None
        ),
        validation_plot_samples=0,
    )


def load_checkpoint(module: TallyQAStudentModule, checkpoint: Path) -> None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = payload["state_dict"] if "state_dict" in payload else payload
    module.load_state_dict(state_dict)


def prompt_lookup(data: TallyQAStudentDataModule) -> dict[int, str]:
    return {
        int(row["class_id"]): str(row["item"])
        for row in data.prompt_classes
        if "class_id" in row and "item" in row
    }


def evaluate(
    module: TallyQAStudentModule,
    data: TallyQAStudentDataModule,
    split: str,
    device: torch.device,
    limit_batches: int | None,
) -> dict[str, Any]:
    loader = {
        "train": data.train_dataloader,
        "val": data.val_dataloader,
        "test": data.test_dataloader,
    }[split]()
    module.to(device)
    module.eval()

    confusion: Counter[tuple[int, int]] = Counter()
    by_prompt: dict[int, Counter] = defaultdict(Counter)
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if limit_batches is not None and batch_index >= limit_batches:
                break
            inputs = {
                "token_ids": batch["token_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "images": batch["images"].to(device),
            }
            logits = module(**inputs)
            predictions = torch.argmax(logits.detach().cpu(), dim=1)
            labels = batch["labels"].detach().cpu()
            item_class_ids = batch["item_class_ids"].detach().cpu()
            dataset_indices = batch["dataset_index"].detach().cpu()
            for offset in range(labels.numel()):
                label = int(labels[offset])
                prediction = int(predictions[offset])
                item_class_id = int(item_class_ids[offset])
                correct = prediction == label
                confusion[(label, prediction)] += 1
                by_prompt[item_class_id]["total"] += 1
                by_prompt[item_class_id]["correct"] += int(correct)
                records.append(
                    {
                        "dataset_index": int(dataset_indices[offset]),
                        "item_class_id": item_class_id,
                        "label": label,
                        "prediction": prediction,
                        "correct": correct,
                    }
                )
    return {"confusion": confusion, "by_prompt": by_prompt, "records": records}


def build_confusion_matrices(
    confusion: Counter[tuple[int, int]],
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.zeros((num_classes, num_classes), dtype=np.int64)
    for (label, prediction), count in confusion.items():
        counts[label, prediction] = int(count)
    row_totals = counts.sum(axis=1, keepdims=True)
    normalized = np.divide(
        counts,
        row_totals,
        out=np.zeros_like(counts, dtype=np.float64),
        where=row_totals > 0,
    )
    return counts, normalized


def class_labels(num_classes: int, collapse_at: int) -> list[str]:
    return [str(index) for index in range(collapse_at)] + [f"{collapse_at}+"]


def plot_confusion(
    counts: np.ndarray,
    normalized: np.ndarray,
    labels: list[str],
    output: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = plt.get_cmap("Blues")
    image = ax.imshow(normalized, vmin=0, vmax=1, cmap=cmap)
    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.set_xlabel("Predicted output class")
    ax.set_ylabel("True output class")
    ax.set_title(title)
    cbar = fig.colorbar(image, ax=ax, shrink=0.85)
    cbar.set_label("Row-normalized fraction")
    for row in range(counts.shape[0]):
        for col in range(counts.shape[1]):
            count = int(counts[row, col])
            if count == 0:
                continue
            value = float(normalized[row, col])
            red, green, blue, _alpha = cmap(value)
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            ax.text(
                col,
                row,
                f"{value:.2f}\n{count}",
                ha="center",
                va="center",
                color="black" if luminance > 0.5 else "white",
                fontsize=9,
            )
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def write_confusion_csv(path: Path, counts: np.ndarray, normalized: np.ndarray, labels: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["true_label", "predicted_label", "count", "row_fraction"],
        )
        writer.writeheader()
        for row, true_label in enumerate(labels):
            for col, predicted_label in enumerate(labels):
                writer.writerow(
                    {
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "count": int(counts[row, col]),
                        "row_fraction": float(normalized[row, col]),
                    }
                )


def prompt_rows(by_prompt: dict[int, Counter], prompts: dict[int, str]) -> list[dict[str, Any]]:
    rows = []
    for item_class_id, counter in sorted(by_prompt.items()):
        total = int(counter["total"])
        correct = int(counter["correct"])
        rows.append(
            {
                "item_class_id": item_class_id,
                "student_prompt": prompts.get(item_class_id, str(item_class_id)),
                "count": total,
                "correct": correct,
                "accuracy": correct / total if total else None,
            }
        )
    return rows


def write_prompt_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["student_prompt", "item_class_id", "count", "correct", "accuracy"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_prompt_accuracy(rows: list[dict[str, Any]], output: Path, title: str) -> None:
    sorted_rows = sorted(rows, key=lambda row: float(row["accuracy"]), reverse=True)
    fig_height = max(10, len(sorted_rows) * 0.22)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    y = np.arange(len(sorted_rows))
    accuracies = [float(row["accuracy"]) for row in sorted_rows]
    labels = [str(row["student_prompt"]) for row in sorted_rows]
    counts = [int(row["count"]) for row in sorted_rows]
    ax.barh(y, accuracies, color="#4c78a8")
    ax.set_yticks(y, labels=labels, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Accuracy")
    ax.set_title(title)
    for index, (accuracy, count) in enumerate(zip(accuracies, counts, strict=True)):
        ax.text(accuracy + 0.005, index, f"{accuracy:.2f} n={count}", va="center", fontsize=5)
    fig.subplots_adjust(left=0.24, right=0.94, top=0.96, bottom=0.04)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    cfg = load_config(args.config_name, args.overrides)
    data = build_data(cfg)
    module = build_module(cfg, data)
    load_checkpoint(module, args.checkpoint)

    device = torch.device(args.device)
    evaluated = evaluate(module, data, args.split, device, args.limit_batches)
    labels = class_labels(int(cfg.model.num_outputs), int(cfg.data.collapse_at))
    counts, normalized = build_confusion_matrices(
        evaluated["confusion"],
        num_classes=int(cfg.model.num_outputs),
    )
    prompts = prompt_lookup(data)
    prompt_accuracy_rows = prompt_rows(evaluated["by_prompt"], prompts)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    confusion_png = args.output_dir / "student_output_confusion_matrix.png"
    confusion_csv = args.output_dir / "student_output_confusion_matrix.csv"
    prompt_png = args.output_dir / "student_accuracy_by_prompt_class_bar.png"
    prompt_csv = args.output_dir / "student_accuracy_by_prompt_class.csv"

    run_name = str(cfg.experiment.run_name)
    title_prefix = f"{run_name} {args.split}"
    plot_confusion(
        counts,
        normalized,
        labels,
        confusion_png,
        f"{title_prefix} Output Confusion Matrix",
    )
    write_confusion_csv(confusion_csv, counts, normalized, labels)
    write_prompt_csv(prompt_csv, prompt_accuracy_rows)
    plot_prompt_accuracy(
        prompt_accuracy_rows,
        prompt_png,
        f"{title_prefix} Accuracy by Prompt Class",
    )

    total = len(evaluated["records"])
    correct = sum(int(row["correct"]) for row in evaluated["records"])
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint),
        "config_name": args.config_name,
        "overrides": args.overrides,
        "split": args.split,
        "limit_batches": args.limit_batches,
        "records": total,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "split_sizes": data.split_sizes(),
        "full_split_sizes": data.full_split_sizes(),
        "teacher_cache_coverage": data.cache_coverage(),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "figures": {
            "confusion_matrix": str(confusion_png),
            "prompt_accuracy": str(prompt_png),
        },
        "tables": {
            "confusion_matrix": str(confusion_csv),
            "prompt_accuracy": str(prompt_csv),
        },
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
