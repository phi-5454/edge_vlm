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

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_SMOLVLM_CACHE = Path("artifacts/teacher_cache/smolvlm_tallyqa_target_mobilenet224.jsonl")
DEFAULT_FASTERRCNN_CACHE = Path(
    "artifacts/teacher_cache/torchvision_fasterrcnn_coco80_letterbox_full_score005_poibin.jsonl"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/reports/final_dataset/teacher_eda_full")
BAR_GUIDES = (0.4, 0.5, 0.6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run contained final-dataset teacher EDA for TallyQA caches, including output-class "
            "balanced prompt accuracies and confusion matrices."
        )
    )
    parser.add_argument("--smolvlm-cache", type=Path, default=DEFAULT_SMOLVLM_CACHE)
    parser.add_argument("--fasterrcnn-cache", type=Path, default=DEFAULT_FASTERRCNN_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--examples-jsonl",
        type=Path,
        default=None,
        help=(
            "Optional materialized split examples.jsonl. When provided, teacher caches are filtered "
            "to this split before computing EDA."
        ),
    )
    parser.add_argument(
        "--reference-examples-jsonl",
        type=Path,
        default=Path("data/tallyqa_cauldron_target_mobilenet224/examples.jsonl"),
        help=(
            "Original full target examples.jsonl used to map materialized split examples back to "
            "teacher-cache dataset_index/example_id values."
        ),
    )
    parser.add_argument("--answer-min", type=int, default=0)
    parser.add_argument("--answer-max", type=int, default=15)
    parser.add_argument("--collapse-at", type=int, default=5)
    parser.add_argument("--min-prompt-count", type=int, default=10)
    return parser.parse_args()


def output_class(answer: int, collapse_at: int | None) -> int | str:
    if collapse_at is not None and answer >= collapse_at:
        return f"{collapse_at}+"
    return answer


def output_classes(answer_min: int, answer_max: int, collapse_at: int | None) -> list[int | str]:
    if collapse_at is None:
        return list(range(answer_min, answer_max + 1))
    if collapse_at <= answer_min:
        return [f"{collapse_at}+"]
    return list(range(answer_min, min(answer_max, collapse_at - 1) + 1)) + [f"{collapse_at}+"]


def init_counter() -> Counter:
    return Counter({"total": 0, "correct": 0})


def update_counter(counter: Counter, correct: bool) -> None:
    counter["total"] += 1
    counter["correct"] += int(correct)


def accuracy(counter: Counter) -> float:
    total = int(counter["total"])
    return float(counter["correct"]) / total if total else float("nan")


def balanced_accuracy(counters: dict[Any, Counter]) -> float:
    values = [accuracy(counter) for counter in counters.values() if int(counter["total"]) > 0]
    return float(np.mean(values)) if values else float("nan")


def example_key(row: dict[str, Any]) -> tuple[int, int | None, str, int]:
    qa_index = row.get("qa_index")
    return (
        int(row["source_row_index"]),
        int(qa_index) if qa_index is not None else None,
        str(row["student_prompt"]),
        int(row["answer"]),
    )


def load_allowed_dataset_indices(
    examples_jsonl: Path | None,
    reference_examples_jsonl: Path,
) -> tuple[set[int] | None, dict[str, Any]]:
    if examples_jsonl is None:
        return None, {"enabled": False}

    requested_keys = set()
    requested_rows = 0
    with examples_jsonl.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {examples_jsonl}:{line_number}") from exc
            requested_keys.add(example_key(row))
            requested_rows += 1

    allowed_indices: set[int] = set()
    duplicate_reference_keys = 0
    reference_key_to_id: dict[tuple[int, int | None, str, int], int] = {}
    with reference_examples_jsonl.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {reference_examples_jsonl}:{line_number}") from exc
            key = example_key(row)
            if key in reference_key_to_id:
                duplicate_reference_keys += 1
                continue
            reference_key_to_id[key] = int(row["example_id"])

    missing_keys = requested_keys - set(reference_key_to_id)
    for key in requested_keys & set(reference_key_to_id):
        allowed_indices.add(reference_key_to_id[key])

    metadata = {
        "enabled": True,
        "examples_jsonl": str(examples_jsonl),
        "reference_examples_jsonl": str(reference_examples_jsonl),
        "requested_rows": requested_rows,
        "requested_unique_keys": len(requested_keys),
        "allowed_dataset_indices": len(allowed_indices),
        "missing_reference_keys": len(missing_keys),
        "duplicate_reference_keys_ignored": duplicate_reference_keys,
    }
    if missing_keys:
        sample = sorted(missing_keys)[:5]
        metadata["missing_reference_key_sample"] = [
            {
                "source_row_index": key[0],
                "qa_index": key[1],
                "student_prompt": key[2],
                "answer": key[3],
            }
            for key in sample
        ]
    return allowed_indices, metadata


def stream_teacher(
    cache: Path,
    name: str,
    labels: list[int | str],
    collapse_at: int | None,
    allowed_dataset_indices: set[int] | None = None,
) -> dict[str, Any]:
    overall = init_counter()
    by_prompt: dict[str, Counter] = defaultdict(init_counter)
    by_output: dict[int | str, Counter] = defaultdict(init_counter)
    by_prompt_output: dict[tuple[str, int | str], Counter] = defaultdict(init_counter)
    confusion: Counter[tuple[int | str, int | str]] = Counter()
    dataset_indices: set[int] = set()

    with cache.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {cache}:{line_number}") from exc
            dataset_index = int(row["dataset_index"])
            if allowed_dataset_indices is not None and dataset_index not in allowed_dataset_indices:
                continue
            prompt = str(row["student_prompt"])
            true_label = output_class(int(row["answer"]), collapse_at)
            pred_label = output_class(
                int(row["teacher_metrics"]["numeric_answer"]["prediction"]),
                collapse_at,
            )
            correct = true_label == pred_label
            dataset_indices.add(dataset_index)
            update_counter(overall, correct)
            update_counter(by_prompt[prompt], correct)
            update_counter(by_output[true_label], correct)
            update_counter(by_prompt_output[(prompt, true_label)], correct)
            confusion[(true_label, pred_label)] += 1

    output_counters = {label: by_output[label] for label in labels}
    prompt_rows = []
    for prompt, counter in sorted(by_prompt.items()):
        prompt_output_counters = {
            label: by_prompt_output[(prompt, label)]
            for label in labels
            if int(by_prompt_output[(prompt, label)]["total"]) > 0
        }
        prompt_rows.append(
            {
                "teacher": name,
                "student_prompt": prompt,
                "count": int(counter["total"]),
                "correct": int(counter["correct"]),
                "accuracy": accuracy(counter),
                "output_class_weighted_accuracy": balanced_accuracy(prompt_output_counters),
                "output_classes_present": len(prompt_output_counters),
            }
        )
    output_rows = [
        {
            "teacher": name,
            "answer": str(label),
            "count": int(output_counters[label]["total"]),
            "correct": int(output_counters[label]["correct"]),
            "accuracy": accuracy(output_counters[label]),
            "recall": accuracy(output_counters[label]),
        }
        for label in labels
    ]
    return {
        "name": name,
        "cache": str(cache),
        "records": int(overall["total"]),
        "dataset_indices": dataset_indices,
        "overall_accuracy": accuracy(overall),
        "overall_output_class_weighted_accuracy": balanced_accuracy(output_counters),
        "by_prompt": prompt_rows,
        "by_output": output_rows,
        "confusion": confusion,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_prompt_bar(
    rows: list[dict[str, Any]],
    title: str,
    output: Path,
    min_count: int,
    sort_by: str = "accuracy",
) -> None:
    filtered = [row for row in rows if int(row["count"]) >= min_count]
    if sort_by == "frequency":
        filtered.sort(key=lambda row: int(row["count"]), reverse=True)
    elif sort_by == "accuracy":
        filtered.sort(key=lambda row: float(row["output_class_weighted_accuracy"]))
    else:
        raise ValueError(f"Unsupported prompt bar sort: {sort_by}")
    height = max(8, 0.24 * len(filtered))
    fig, ax = plt.subplots(figsize=(12, height))
    y = np.arange(len(filtered))
    values = [float(row["output_class_weighted_accuracy"]) for row in filtered]
    ax.barh(y, values, color="#4c78a8")
    ax.set_yticks(y, labels=[str(row["student_prompt"]) for row in filtered], fontsize=6)
    if sort_by == "frequency":
        ax.invert_yaxis()
    ax.set_xlabel("Output-class-weighted accuracy within prompt")
    ax.set_xlim(0, 1)
    ax.set_title(title)
    for guide in BAR_GUIDES:
        ax.axvline(guide, color="#666666", linewidth=0.8, linestyle="--", alpha=0.45, zorder=0)
    for index, row in enumerate(filtered):
        value = float(row["output_class_weighted_accuracy"])
        ax.text(
            value + 0.005,
            index,
            f"{value:.2f} n={row['count']} k={row['output_classes_present']}",
            va="center",
            fontsize=5,
        )
    fig.subplots_adjust(left=0.24, right=0.96, top=0.96, bottom=0.04)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_prompt_pair_bar(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    left_name: str,
    right_name: str,
    title: str,
    output: Path,
    min_count: int,
    sort_by: str = "delta",
) -> None:
    left_by_prompt = {str(row["student_prompt"]): row for row in left_rows}
    right_by_prompt = {str(row["student_prompt"]): row for row in right_rows}
    prompts = sorted(set(left_by_prompt) | set(right_by_prompt))
    rows = []
    for prompt in prompts:
        left = left_by_prompt.get(prompt)
        right = right_by_prompt.get(prompt)
        max_count = max(int(left["count"]) if left else 0, int(right["count"]) if right else 0)
        if max_count >= min_count:
            rows.append((prompt, left, right, max_count))
    if sort_by == "frequency":
        rows.sort(key=lambda item: item[3], reverse=True)
    elif sort_by == "delta":
        rows.sort(
            key=lambda item: (
                float(item[2]["output_class_weighted_accuracy"]) if item[2] else -1.0
            )
            - (float(item[1]["output_class_weighted_accuracy"]) if item[1] else -1.0)
        )
    else:
        raise ValueError(f"Unsupported prompt pair bar sort: {sort_by}")
    height = max(8, 0.28 * len(rows))
    fig, ax = plt.subplots(figsize=(13, height))
    y = np.arange(len(rows))
    bar_height = 0.38
    left_values = [
        float(row["output_class_weighted_accuracy"]) if row is not None else np.nan
        for _prompt, row, _right, _count in rows
    ]
    right_values = [
        float(row["output_class_weighted_accuracy"]) if row is not None else np.nan
        for _prompt, _left, row, _count in rows
    ]
    ax.barh(y - bar_height / 2, left_values, height=bar_height, color="#4c78a8", label=left_name)
    ax.barh(y + bar_height / 2, right_values, height=bar_height, color="#f58518", label=right_name)
    ax.set_yticks(y, labels=[prompt for prompt, _left, _right, _count in rows], fontsize=6)
    if sort_by == "frequency":
        ax.invert_yaxis()
    ax.set_xlabel("Output-class-weighted accuracy within prompt")
    ax.set_xlim(0, 1)
    ax.set_title(title)
    for guide in BAR_GUIDES:
        ax.axvline(guide, color="#666666", linewidth=0.8, linestyle="--", alpha=0.45, zorder=0)
    ax.legend(loc="lower right")
    fig.subplots_adjust(left=0.24, right=0.96, top=0.96, bottom=0.04)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_output_bar(
    rows: list[dict[str, Any]],
    title: str,
    output: Path,
    metric: str = "accuracy",
    ylabel: str = "Accuracy",
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(rows))
    values = [float(row[metric]) for row in rows]
    ax.bar(x, values, color="#4c78a8")
    ax.set_xticks(x, labels=[str(row["answer"]) for row in rows])
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    for guide in BAR_GUIDES:
        ax.axhline(guide, color="#666666", linewidth=0.8, linestyle="--", alpha=0.45, zorder=0)
    for index, row in enumerate(rows):
        value = float(row[metric])
        ax.text(index, value + 0.015, f"{value:.2f}\nn={row['count']}", ha="center", fontsize=7)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_output_pair_bar(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    left_name: str,
    right_name: str,
    title: str,
    output: Path,
    metric: str = "accuracy",
    ylabel: str = "Accuracy",
) -> None:
    labels = [str(row["answer"]) for row in left_rows]
    right_by_answer = {str(row["answer"]): row for row in right_rows}
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        x - width / 2,
        [float(row[metric]) for row in left_rows],
        width=width,
        color="#4c78a8",
        label=left_name,
    )
    ax.bar(
        x + width / 2,
        [float(right_by_answer[label][metric]) for label in labels],
        width=width,
        color="#f58518",
        label=right_name,
    )
    ax.set_xticks(x, labels=labels)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    for guide in BAR_GUIDES:
        ax.axhline(guide, color="#666666", linewidth=0.8, linestyle="--", alpha=0.45, zorder=0)
    ax.legend(loc="lower right")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def confusion_matrices(
    confusion: Counter[tuple[int | str, int | str]],
    labels: list[int | str],
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.zeros((len(labels), len(labels)), dtype=np.int64)
    index_by_label = {label: index for index, label in enumerate(labels)}
    for (true_label, pred_label), count in confusion.items():
        if true_label in index_by_label and pred_label in index_by_label:
            counts[index_by_label[true_label], index_by_label[pred_label]] = int(count)
    row_totals = counts.sum(axis=1, keepdims=True)
    normalized = np.divide(
        counts,
        row_totals,
        out=np.zeros_like(counts, dtype=np.float64),
        where=row_totals > 0,
    )
    return counts, normalized


def draw_confusion_axis(
    ax: plt.Axes,
    counts: np.ndarray,
    normalized: np.ndarray,
    labels: list[int | str],
    title: str,
) -> None:
    cmap = plt.get_cmap("Blues")
    image = ax.imshow(normalized, vmin=0, vmax=1, cmap=cmap)
    ax.set_xticks(np.arange(len(labels)), labels=[str(label) for label in labels])
    ax.set_yticks(np.arange(len(labels)), labels=[str(label) for label in labels])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
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
                fontsize=8,
            )
    return image


def plot_confusion(
    counts: np.ndarray,
    normalized: np.ndarray,
    labels: list[int | str],
    title: str,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    image = draw_confusion_axis(ax, counts, normalized, labels, title)
    cbar = fig.colorbar(image, ax=ax, shrink=0.85)
    cbar.set_label("Row-normalized fraction")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_confusion_pair(
    left: tuple[np.ndarray, np.ndarray],
    right: tuple[np.ndarray, np.ndarray],
    labels: list[int | str],
    left_name: str,
    right_name: str,
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), constrained_layout=True)
    image = draw_confusion_axis(axes[0], left[0], left[1], labels, left_name)
    draw_confusion_axis(axes[1], right[0], right[1], labels, right_name)
    cbar = fig.colorbar(image, ax=axes, shrink=0.85)
    cbar.set_label("Row-normalized fraction")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def write_confusion_csv(
    path: Path,
    counts: np.ndarray,
    normalized: np.ndarray,
    labels: list[int | str],
) -> None:
    rows = []
    for row_index, true_label in enumerate(labels):
        for col_index, pred_label in enumerate(labels):
            rows.append(
                {
                    "true_label": str(true_label),
                    "predicted_label": str(pred_label),
                    "count": int(counts[row_index, col_index]),
                    "row_fraction": float(normalized[row_index, col_index]),
                }
            )
    write_csv(path, rows)


def run_teacher_plots(
    result: dict[str, Any],
    labels: list[int | str],
    teacher_dir: Path,
    min_prompt_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    figures = teacher_dir / "figures"
    tables = teacher_dir / "tables"
    write_csv(tables / "prompt_output_class_weighted_accuracy.csv", result["by_prompt"])
    write_csv(tables / "output_class_accuracy.csv", result["by_output"])
    write_csv(tables / "output_class_recall.csv", result["by_output"])
    plot_prompt_bar(
        result["by_prompt"],
        title=f"{result['name']}: Prompt Accuracy Balanced Across Output Classes",
        output=figures / "prompt_output_class_weighted_accuracy.png",
        min_count=min_prompt_count,
    )
    plot_prompt_bar(
        result["by_prompt"],
        title=f"{result['name']}: Prompt Accuracy Balanced Across Output Classes, Ordered by Frequency",
        output=figures / "prompt_output_class_weighted_accuracy_by_frequency.png",
        min_count=min_prompt_count,
        sort_by="frequency",
    )
    plot_output_bar(
        result["by_output"],
        title=f"{result['name']}: Accuracy by True Output Class",
        output=figures / "output_class_accuracy.png",
    )
    plot_output_bar(
        result["by_output"],
        title=f"{result['name']}: Recall by True Output Class",
        output=figures / "output_class_recall.png",
        metric="recall",
        ylabel="Recall",
    )
    counts, normalized = confusion_matrices(result["confusion"], labels)
    plot_confusion(
        counts,
        normalized,
        labels,
        title=f"{result['name']}: Output Confusion Matrix",
        output=figures / "output_confusion_matrix.png",
    )
    write_confusion_csv(tables / "output_confusion_matrix.csv", counts, normalized, labels)
    return counts, normalized


def main() -> None:
    args = parse_args()
    if args.collapse_at is not None and args.collapse_at < args.answer_min:
        raise ValueError("--collapse-at must be >= --answer-min.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = output_classes(args.answer_min, args.answer_max, args.collapse_at)
    allowed_dataset_indices, filter_metadata = load_allowed_dataset_indices(
        args.examples_jsonl,
        args.reference_examples_jsonl,
    )
    teachers = [
        ("smolvlm256_stretched", "SmolVLM-256M", args.smolvlm_cache),
        ("fasterrcnn_coco80_letterbox_score005_poibin", "Faster R-CNN COCO80", args.fasterrcnn_cache),
    ]
    results = {
        key: stream_teacher(cache, display_name, labels, args.collapse_at, allowed_dataset_indices)
        for key, display_name, cache in teachers
    }

    smol_key = "smolvlm256_stretched"
    detector_key = "fasterrcnn_coco80_letterbox_score005_poibin"
    smol_confusion = run_teacher_plots(
        results[smol_key],
        labels,
        args.output_dir / smol_key,
        args.min_prompt_count,
    )
    detector_confusion = run_teacher_plots(
        results[detector_key],
        labels,
        args.output_dir / detector_key,
        args.min_prompt_count,
    )

    combined_figures = args.output_dir / "combined" / "figures"
    combined_tables = args.output_dir / "combined" / "tables"
    plot_prompt_pair_bar(
        results[smol_key]["by_prompt"],
        results[detector_key]["by_prompt"],
        results[smol_key]["name"],
        results[detector_key]["name"],
        "Prompt Accuracy Balanced Across Output Classes",
        combined_figures / "prompt_output_class_weighted_accuracy_side_by_side.png",
        args.min_prompt_count,
    )
    plot_prompt_pair_bar(
        results[smol_key]["by_prompt"],
        results[detector_key]["by_prompt"],
        results[smol_key]["name"],
        results[detector_key]["name"],
        "Prompt Accuracy Balanced Across Output Classes, Ordered by Frequency",
        combined_figures / "prompt_output_class_weighted_accuracy_by_frequency_side_by_side.png",
        args.min_prompt_count,
        sort_by="frequency",
    )
    plot_output_pair_bar(
        results[smol_key]["by_output"],
        results[detector_key]["by_output"],
        results[smol_key]["name"],
        results[detector_key]["name"],
        "Accuracy by True Output Class",
        combined_figures / "output_class_accuracy_side_by_side.png",
    )
    plot_output_pair_bar(
        results[smol_key]["by_output"],
        results[detector_key]["by_output"],
        results[smol_key]["name"],
        results[detector_key]["name"],
        "Recall by True Output Class",
        combined_figures / "output_class_recall_side_by_side.png",
        metric="recall",
        ylabel="Recall",
    )
    plot_confusion_pair(
        smol_confusion,
        detector_confusion,
        labels,
        results[smol_key]["name"],
        results[detector_key]["name"],
        combined_figures / "output_confusion_matrices_side_by_side.png",
    )
    write_csv(
        combined_tables / "prompt_output_class_weighted_accuracy_all_teachers.csv",
        results[smol_key]["by_prompt"] + results[detector_key]["by_prompt"],
    )
    write_csv(
        combined_tables / "output_class_accuracy_all_teachers.csv",
        results[smol_key]["by_output"] + results[detector_key]["by_output"],
    )
    write_csv(
        combined_tables / "output_class_recall_all_teachers.csv",
        results[smol_key]["by_output"] + results[detector_key]["by_output"],
    )

    common_indices = results[smol_key]["dataset_indices"] & results[detector_key]["dataset_indices"]
    union_indices = results[smol_key]["dataset_indices"] | results[detector_key]["dataset_indices"]
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(args.output_dir),
        "answer_min": args.answer_min,
        "answer_max": args.answer_max,
        "collapse_at": args.collapse_at,
        "min_prompt_count": args.min_prompt_count,
        "filter": filter_metadata,
        "metric_definition": {
            "prompt_output_class_weighted_accuracy": (
                "For each prompt class, compute accuracy separately for each true output class "
                "present in that prompt class, then average those output-class accuracies equally."
            ),
            "k_in_prompt_bar_labels": (
                "Number of true output classes with at least one example for that prompt class."
            ),
            "overall_output_class_weighted_accuracy": (
                "Mean of true-output-class accuracies across collapsed output classes present "
                "in the teacher cache."
            ),
            "output_class_recall": (
                "For these count labels this is the same denominator as accuracy by true output "
                "class: correct examples divided by all examples with that true output class."
            ),
        },
        "teacher_summaries": {
            key: {
                "name": result["name"],
                "cache": result["cache"],
                "records": result["records"],
                "prompt_classes": len(result["by_prompt"]),
                "overall_accuracy": result["overall_accuracy"],
                "overall_output_class_weighted_accuracy": result[
                    "overall_output_class_weighted_accuracy"
                ],
            }
            for key, result in results.items()
        },
        "dataset_index_coverage": {
            "intersection": len(common_indices),
            "union": len(union_indices),
            "smolvlm_only": len(results[smol_key]["dataset_indices"] - results[detector_key]["dataset_indices"]),
            "fasterrcnn_only": len(results[detector_key]["dataset_indices"] - results[smol_key]["dataset_indices"]),
        },
        "artifacts": {
            smol_key: {
                "figures": str(args.output_dir / smol_key / "figures"),
                "tables": str(args.output_dir / smol_key / "tables"),
            },
            detector_key: {
                "figures": str(args.output_dir / detector_key / "figures"),
                "tables": str(args.output_dir / detector_key / "tables"),
            },
            "combined": {
                "figures": str(combined_figures),
                "tables": str(combined_tables),
            },
        },
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
