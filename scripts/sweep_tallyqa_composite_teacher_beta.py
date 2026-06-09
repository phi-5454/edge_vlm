from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ["MPLBACKEND"] = "Agg"

import matplotlib.pyplot as plt
import numpy as np

from scripts.compare_tallyqa_composite_vs_argmax_teacher import teacher_key, write_csv
from scripts.run_tallyqa_final_dataset_teacher_eda import (
    accuracy,
    balanced_accuracy,
    init_counter,
    load_allowed_dataset_indices,
    output_class,
    output_classes,
    update_counter,
)


DEFAULT_OUTPUT_DIR = Path(
    "artifacts/reports/final_dataset/post_pruning_teacher_eda/composite_teacher_beta_sweep"
)
TEACHER_LABELS = {
    "smolvlm": "SmolVLM-256M",
    "fasterrcnn": "Faster R-CNN COCO80",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep the beta temperature for the prompt-accuracy-weighted composite teacher. "
            "Prompt weights are fit from train/val prompt output-class-weighted accuracy."
        )
    )
    parser.add_argument(
        "--selection-csv",
        type=Path,
        default=Path(
            "artifacts/reports/final_dataset/post_pruning_teacher_eda/trainval/combined/tables/"
            "prompt_output_class_weighted_accuracy_all_teachers.csv"
        ),
    )
    parser.add_argument(
        "--smolvlm-cache",
        type=Path,
        default=Path("artifacts/teacher_cache/smolvlm_tallyqa_target_mobilenet224.jsonl"),
    )
    parser.add_argument(
        "--fasterrcnn-cache",
        type=Path,
        default=Path(
            "artifacts/teacher_cache/torchvision_fasterrcnn_coco80_letterbox_full_score005_poibin.jsonl"
        ),
    )
    parser.add_argument(
        "--trainval-examples",
        type=Path,
        default=Path("data/final_dataset/tallyqa_trainval_mobilenet224_letterbox/examples.jsonl"),
    )
    parser.add_argument(
        "--test-examples",
        type=Path,
        default=Path("data/final_dataset/tallyqa_test_mobilenet224_letterbox/examples.jsonl"),
    )
    parser.add_argument(
        "--reference-examples",
        type=Path,
        default=Path("data/tallyqa_cauldron_target_mobilenet224/examples.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--beta-min", type=float, default=0.1)
    parser.add_argument("--beta-max", type=float, default=10.0)
    parser.add_argument("--num-betas", type=int, default=21)
    parser.add_argument(
        "--extra-beta",
        type=float,
        action="append",
        default=[5.0],
        help="Additional beta value to include exactly. Can be passed multiple times.",
    )
    parser.add_argument("--chance-baseline", type=float, default=1.0 / 6.0)
    parser.add_argument(
        "--smolvlm-temperature",
        type=float,
        default=1.0,
        help="Probability temperature applied to SmolVLM numeric candidate probabilities before fusion.",
    )
    parser.add_argument(
        "--fasterrcnn-temperature",
        type=float,
        default=1.0,
        help=(
            "Probability temperature applied to Faster R-CNN numeric candidate probabilities before fusion."
        ),
    )
    parser.add_argument("--collapse-at", type=int, default=5)
    parser.add_argument("--answer-min", type=int, default=0)
    parser.add_argument("--answer-max", type=int, default=15)
    return parser.parse_args()


def beta_values(beta_min: float, beta_max: float, num_betas: int, extra: list[float]) -> list[float]:
    if beta_min <= 0 or beta_max <= 0:
        raise ValueError("Log-spaced beta sweep requires positive beta bounds.")
    if num_betas < 2:
        raise ValueError("--num-betas must be at least 2.")
    values = list(np.geomspace(beta_min, beta_max, num_betas))
    values.extend(extra)
    return sorted({round(float(value), 10) for value in values})


def load_prompt_accuracies(selection_csv: Path) -> dict[str, dict[str, float]]:
    prompt_accuracies: dict[str, dict[str, float]] = {}
    with selection_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            prompt = row["student_prompt"]
            prompt_accuracies.setdefault(prompt, {})[teacher_key(row["teacher"])] = float(
                row["output_class_weighted_accuracy"]
            )
    return prompt_accuracies


def weights_for_beta(
    prompt_accuracies: dict[str, dict[str, float]],
    beta: float,
    chance_baseline: float,
) -> dict[str, dict[str, float]]:
    weights: dict[str, dict[str, float]] = {}
    for prompt, accuracies in prompt_accuracies.items():
        raw = {
            teacher: math.exp(beta * (accuracy_value - chance_baseline))
            for teacher, accuracy_value in accuracies.items()
        }
        total = sum(raw.values())
        weights[prompt] = {teacher: value / total for teacher, value in raw.items()}
    return weights


def apply_probability_temperature(
    probs: np.ndarray,
    temperature: float,
    eps: float = 1e-12,
) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("Probability temperature must be positive.")
    if temperature == 1.0:
        return probs
    scaled = np.where(probs > 0, np.power(np.maximum(probs, eps), 1.0 / temperature), 0.0)
    total = float(scaled.sum())
    if total <= 0:
        return probs
    return scaled / total


def candidate_probs(row: dict[str, Any], answer_max: int, temperature: float) -> np.ndarray:
    probs = np.zeros(answer_max + 1, dtype=np.float64)
    candidates = row.get("teacher_logits", {}).get("numeric_answer_candidates", [])
    for candidate in candidates:
        answer = int(candidate["answer"])
        if 0 <= answer <= answer_max:
            probs[answer] = float(candidate["candidate_probability"])
    total = float(probs.sum())
    if total > 0:
        return apply_probability_temperature(probs / total, temperature)
    prediction = int(row["teacher_metrics"]["numeric_answer"]["prediction"])
    probs[min(max(prediction, 0), answer_max)] = 1.0
    return probs


def load_distribution_rows(
    cache: Path,
    allowed_dataset_indices: set[int],
    collapse_at: int,
    answer_max: int,
    temperature: float,
) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with cache.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {cache}:{line_number}") from exc
            dataset_index = int(row["dataset_index"])
            if dataset_index not in allowed_dataset_indices:
                continue
            answer = int(row["answer"])
            probs = candidate_probs(row, answer_max, temperature)
            prediction = int(np.argmax(probs))
            rows[dataset_index] = {
                "dataset_index": dataset_index,
                "student_prompt": str(row["student_prompt"]),
                "answer": answer,
                "true_label": output_class(answer, collapse_at),
                "probs": probs,
                "prediction": prediction,
                "pred_label": output_class(prediction, collapse_at),
            }
    return rows


def evaluate_hard_teacher(
    rows: dict[int, dict[str, Any]],
    labels: list[int | str],
) -> dict[str, Any]:
    overall = init_counter()
    by_output = {label: init_counter() for label in labels}
    by_prompt_output: dict[str, dict[int | str, Counter]] = {}
    for row in rows.values():
        correct = row["pred_label"] == row["true_label"]
        update_counter(overall, correct)
        update_counter(by_output[row["true_label"]], correct)
        prompt_outputs = by_prompt_output.setdefault(
            row["student_prompt"],
            {label: init_counter() for label in labels},
        )
        update_counter(prompt_outputs[row["true_label"]], correct)

    prompt_values = [
        balanced_accuracy(
            {label: counter for label, counter in counters.items() if int(counter["total"]) > 0}
        )
        for counters in by_prompt_output.values()
    ]
    return {
        "records": int(overall["total"]),
        "overall_accuracy": accuracy(overall),
        "overall_output_class_weighted_accuracy": balanced_accuracy(by_output),
        "mean_prompt_output_class_weighted_accuracy": float(np.mean(prompt_values)),
        "prompt_classes": len(prompt_values),
    }


def evaluate_composite(
    weights: dict[str, dict[str, float]],
    smol: dict[int, dict[str, Any]],
    faster: dict[int, dict[str, Any]],
    labels: list[int | str],
    collapse_at: int,
) -> dict[str, Any]:
    overall = init_counter()
    by_output = {label: init_counter() for label in labels}
    by_prompt_output: dict[str, dict[int | str, Counter]] = {}
    source_counts = {
        "mixed_both_available": 0,
        "smol_only_missing_fasterrcnn": 0,
        "smol_only_no_fasterrcnn_weight": 0,
    }

    for dataset_index, smol_row in smol.items():
        prompt = smol_row["student_prompt"]
        prompt_weights = weights.get(prompt, {"smolvlm": 1.0})
        smol_weight = prompt_weights.get("smolvlm", 0.0)
        faster_weight = prompt_weights.get("fasterrcnn", 0.0)
        if faster_weight > 0 and dataset_index in faster:
            probs = smol_weight * smol_row["probs"] + faster_weight * faster[dataset_index]["probs"]
            source_counts["mixed_both_available"] += 1
        elif faster_weight > 0:
            probs = smol_row["probs"]
            source_counts["smol_only_missing_fasterrcnn"] += 1
        else:
            probs = smol_row["probs"]
            source_counts["smol_only_no_fasterrcnn_weight"] += 1
        prediction = int(np.argmax(probs))
        pred_label = output_class(prediction, collapse_at)
        correct = pred_label == smol_row["true_label"]

        update_counter(overall, correct)
        update_counter(by_output[smol_row["true_label"]], correct)
        prompt_outputs = by_prompt_output.setdefault(
            prompt,
            {label: init_counter() for label in labels},
        )
        update_counter(prompt_outputs[smol_row["true_label"]], correct)

    prompt_values = [
        balanced_accuracy(
            {label: counter for label, counter in counters.items() if int(counter["total"]) > 0}
        )
        for counters in by_prompt_output.values()
    ]
    return {
        "records": int(overall["total"]),
        "overall_accuracy": accuracy(overall),
        "overall_output_class_weighted_accuracy": balanced_accuracy(by_output),
        "mean_prompt_output_class_weighted_accuracy": float(np.mean(prompt_values)),
        "prompt_classes": len(prompt_values),
        "source_counts": source_counts,
    }


def flatten_metrics(split: str, beta: float, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "split": split,
        "beta": beta,
        "records": metrics["records"],
        "overall_accuracy": metrics["overall_accuracy"],
        "overall_output_class_weighted_accuracy": metrics["overall_output_class_weighted_accuracy"],
        "mean_prompt_output_class_weighted_accuracy": metrics[
            "mean_prompt_output_class_weighted_accuracy"
        ],
        "prompt_classes": metrics["prompt_classes"],
    }


def plot_sweep(rows: list[dict[str, Any]], baselines: dict[str, dict[str, Any]], output_path: Path) -> None:
    metrics = [
        ("overall_accuracy", "Overall accuracy"),
        ("overall_output_class_weighted_accuracy", "Output-class weighted accuracy"),
        ("mean_prompt_output_class_weighted_accuracy", "Mean prompt weighted accuracy"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True)
    for ax, (metric_key, title) in zip(axes, metrics, strict=True):
        for split, color in (("trainval", "tab:blue"), ("test", "tab:orange")):
            split_rows = [row for row in rows if row["split"] == split]
            ax.plot(
                [row["beta"] for row in split_rows],
                [row[metric_key] for row in split_rows],
                marker="o",
                markersize=3,
                linewidth=1.5,
                label=f"{split} composite",
                color=color,
            )
            for teacher, linestyle in (("smolvlm", ":"), ("fasterrcnn", "--")):
                baseline = baselines[split][teacher]
                ax.axhline(
                    baseline[metric_key],
                    color=color,
                    linestyle=linestyle,
                    linewidth=0.8,
                    alpha=0.6,
                    label=f"{split} {TEACHER_LABELS[teacher]}",
                )
        ax.set_xscale("log")
        ax.set_title(title)
        ax.set_xlabel("beta")
        ax.grid(axis="y", color="0.85", linewidth=0.8)
    axes[0].set_ylabel("Accuracy")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    betas = beta_values(args.beta_min, args.beta_max, args.num_betas, args.extra_beta)
    prompt_accuracies = load_prompt_accuracies(args.selection_csv)
    labels = output_classes(args.answer_min, args.answer_max, args.collapse_at)
    split_examples = {"trainval": args.trainval_examples, "test": args.test_examples}

    all_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    baselines: dict[str, dict[str, Any]] = {}
    filter_metadata: dict[str, Any] = {}

    for split, examples_path in split_examples.items():
        allowed_indices, metadata = load_allowed_dataset_indices(examples_path, args.reference_examples)
        assert allowed_indices is not None
        filter_metadata[split] = metadata
        smol = load_distribution_rows(
            args.smolvlm_cache,
            allowed_indices,
            args.collapse_at,
            args.answer_max,
            args.smolvlm_temperature,
        )
        faster = load_distribution_rows(
            args.fasterrcnn_cache,
            allowed_indices,
            args.collapse_at,
            args.answer_max,
            args.fasterrcnn_temperature,
        )
        baselines[split] = {
            "smolvlm": evaluate_hard_teacher(smol, labels),
            "fasterrcnn": evaluate_hard_teacher(faster, labels),
        }
        for teacher_name, metrics in baselines[split].items():
            row = flatten_metrics(split, float("nan"), metrics)
            row["teacher"] = teacher_name
            baseline_rows.append(row)

        for beta in betas:
            weights = weights_for_beta(prompt_accuracies, beta, args.chance_baseline)
            metrics = evaluate_composite(weights, smol, faster, labels, args.collapse_at)
            all_rows.append(flatten_metrics(split, beta, metrics))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "tables" / "beta_sweep_metrics.csv", all_rows)
    write_csv(args.output_dir / "tables" / "teacher_baseline_metrics.csv", baseline_rows)

    weight_rows: list[dict[str, Any]] = []
    for beta in betas:
        weights = weights_for_beta(prompt_accuracies, beta, args.chance_baseline)
        for prompt, prompt_weights in sorted(weights.items()):
            weight_rows.append(
                {
                    "beta": beta,
                    "student_prompt": prompt,
                    "smolvlm_weight": prompt_weights.get("smolvlm", 0.0),
                    "fasterrcnn_weight": prompt_weights.get("fasterrcnn", 0.0),
                }
            )
    write_csv(args.output_dir / "tables" / "prompt_teacher_weights_by_beta.csv", weight_rows)
    plot_sweep(all_rows, baselines, args.output_dir / "figures" / "beta_sweep_metrics.png")

    best_by_split = {}
    for split in split_examples:
        split_rows = [row for row in all_rows if row["split"] == split]
        best_by_split[split] = {
            metric: max(split_rows, key=lambda row, key=metric: row[key])
            for metric in (
                "overall_accuracy",
                "overall_output_class_weighted_accuracy",
                "mean_prompt_output_class_weighted_accuracy",
            )
        }

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "formula": "w_t = exp(beta * (prompt_output_class_weighted_accuracy_t - 1/6)) / Z",
        "note": "The chance-baseline term is common to available teachers for a prompt, so it does not change two-teacher relative weights.",
        "beta_values": betas,
        "selection_source": str(args.selection_csv),
        "teacher_probability_temperatures": {
            "smolvlm": args.smolvlm_temperature,
            "fasterrcnn": args.fasterrcnn_temperature,
        },
        "temperature_note": (
            "Temperatures are applied to the uncollapsed numeric candidate probabilities before "
            "teacher fusion; evaluation metrics still collapse counts at the configured threshold."
        ),
        "filter_metadata": filter_metadata,
        "artifacts": {
            "metrics": str(args.output_dir / "tables" / "beta_sweep_metrics.csv"),
            "baselines": str(args.output_dir / "tables" / "teacher_baseline_metrics.csv"),
            "weights": str(args.output_dir / "tables" / "prompt_teacher_weights_by_beta.csv"),
            "plot": str(args.output_dir / "figures" / "beta_sweep_metrics.png"),
        },
        "best_by_split": best_by_split,
    }
    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    for split, best_rows in best_by_split.items():
        best_balanced = best_rows["overall_output_class_weighted_accuracy"]
        print(
            f"{split}: best output-class weighted accuracy "
            f"{best_balanced['overall_output_class_weighted_accuracy']:.4f} "
            f"at beta={best_balanced['beta']:.4g}"
        )


if __name__ == "__main__":
    main()
