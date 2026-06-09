from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

os.environ["MPLBACKEND"] = "Agg"

import matplotlib.pyplot as plt
import numpy as np

from scripts.run_tallyqa_final_dataset_teacher_eda import (
    accuracy,
    balanced_accuracy,
    init_counter,
    load_allowed_dataset_indices,
    output_class,
    output_classes,
    plot_confusion_pair,
    plot_output_pair_bar,
    plot_prompt_pair_bar,
    run_teacher_plots,
    stream_teacher,
    update_counter,
    write_csv,
)
from scripts.sweep_tallyqa_composite_teacher_beta import (
    load_distribution_rows,
    load_prompt_accuracies,
    weights_for_beta,
)


DEFAULT_OUTPUT_DIR = Path(
    "artifacts/reports/final_dataset/post_pruning_teacher_eda/"
    "composite_teacher_ece_temp_smol1p1_frcnn2p2_beta12p968"
)
DEFAULT_BETA = 12.9683955465


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run final-dataset EDA for the selected composite TallyQA teacher and compare it "
            "against SmolVLM-256M and Faster R-CNN."
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
    parser.add_argument("--composite-beta", type=float, default=DEFAULT_BETA)
    parser.add_argument("--smolvlm-temperature", type=float, default=1.1)
    parser.add_argument("--fasterrcnn-temperature", type=float, default=2.2)
    parser.add_argument("--chance-baseline", type=float, default=1.0 / 6.0)
    parser.add_argument("--answer-min", type=int, default=0)
    parser.add_argument("--answer-max", type=int, default=15)
    parser.add_argument("--collapse-at", type=int, default=5)
    parser.add_argument("--min-prompt-count", type=int, default=10)
    return parser.parse_args()


def evaluate_composite_teacher(
    smol: dict[int, dict[str, Any]],
    faster: dict[int, dict[str, Any]],
    weights: dict[str, dict[str, float]],
    labels: list[int | str],
    collapse_at: int,
    display_name: str,
) -> dict[str, Any]:
    overall = init_counter()
    by_prompt: dict[str, Counter] = defaultdict(init_counter)
    by_output: dict[int | str, Counter] = defaultdict(init_counter)
    by_prompt_output: dict[tuple[str, int | str], Counter] = defaultdict(init_counter)
    confusion: Counter[tuple[int | str, int | str]] = Counter()
    dataset_indices: set[int] = set()
    prediction_rows: list[dict[str, Any]] = []
    source_counts = Counter(
        {
            "mixed_both_available": 0,
            "smol_only_missing_fasterrcnn": 0,
            "smol_only_no_fasterrcnn_weight": 0,
        }
    )

    for dataset_index, smol_row in sorted(smol.items()):
        prompt = smol_row["student_prompt"]
        prompt_weights = weights.get(prompt, {"smolvlm": 1.0})
        smol_weight = float(prompt_weights.get("smolvlm", 0.0))
        faster_weight = float(prompt_weights.get("fasterrcnn", 0.0))
        if faster_weight > 0 and dataset_index in faster:
            probs = smol_weight * smol_row["probs"] + faster_weight * faster[dataset_index]["probs"]
            source = "mixed_both_available"
        elif faster_weight > 0:
            probs = smol_row["probs"]
            source = "smol_only_missing_fasterrcnn"
        else:
            probs = smol_row["probs"]
            source = "smol_only_no_fasterrcnn_weight"
        source_counts[source] += 1

        prediction = int(np.argmax(probs))
        pred_label = output_class(prediction, collapse_at)
        true_label = smol_row["true_label"]
        correct = pred_label == true_label

        dataset_indices.add(dataset_index)
        update_counter(overall, correct)
        update_counter(by_prompt[prompt], correct)
        update_counter(by_output[true_label], correct)
        update_counter(by_prompt_output[(prompt, true_label)], correct)
        confusion[(true_label, pred_label)] += 1
        prediction_rows.append(
            {
                "dataset_index": dataset_index,
                "student_prompt": prompt,
                "answer": smol_row["answer"],
                "prediction": prediction,
                "true_label": str(true_label),
                "predicted_label": str(pred_label),
                "correct": int(correct),
                "smolvlm_weight": smol_weight,
                "fasterrcnn_weight": faster_weight,
                "source": source,
            }
        )

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
                "teacher": display_name,
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
            "teacher": display_name,
            "answer": str(label),
            "count": int(output_counters[label]["total"]),
            "correct": int(output_counters[label]["correct"]),
            "accuracy": accuracy(output_counters[label]),
            "recall": accuracy(output_counters[label]),
        }
        for label in labels
    ]
    return {
        "name": display_name,
        "cache": "composite",
        "records": int(overall["total"]),
        "dataset_indices": dataset_indices,
        "overall_accuracy": accuracy(overall),
        "overall_output_class_weighted_accuracy": balanced_accuracy(output_counters),
        "by_prompt": prompt_rows,
        "by_output": output_rows,
        "confusion": confusion,
        "prediction_rows": prediction_rows,
        "source_counts": dict(source_counts),
    }


def plot_three_way_prompt_bar(
    teacher_rows: list[tuple[str, list[dict[str, Any]]]],
    title: str,
    output: Path,
    min_count: int,
    sort_teacher: str,
    sort_by: str = "accuracy",
) -> None:
    by_teacher = {
        name: {str(row["student_prompt"]): row for row in rows} for name, rows in teacher_rows
    }
    prompts = sorted({prompt for rows in by_teacher.values() for prompt in rows})
    filtered = []
    for prompt in prompts:
        max_count = max(
            int(rows[prompt]["count"]) if prompt in rows else 0 for rows in by_teacher.values()
        )
        if max_count >= min_count:
            filtered.append(prompt)
    if sort_by == "frequency":
        filtered.sort(
            key=lambda prompt: max(
                int(rows[prompt]["count"]) if prompt in rows else 0 for rows in by_teacher.values()
            ),
            reverse=True,
        )
    elif sort_by == "accuracy":
        sort_rows = by_teacher[sort_teacher]
        filtered.sort(
            key=lambda prompt: float(
                sort_rows.get(prompt, {"output_class_weighted_accuracy": -1.0})[
                    "output_class_weighted_accuracy"
                ]
            )
        )
    else:
        raise ValueError(f"Unsupported sort_by: {sort_by}")

    height = max(8, 0.31 * len(filtered))
    fig, ax = plt.subplots(figsize=(13, height))
    y = np.arange(len(filtered))
    bar_height = 0.25
    offsets = np.linspace(-bar_height, bar_height, len(teacher_rows))
    colors = ["#4c78a8", "#f58518", "#54a24b"]
    for (offset, color, (name, _rows)) in zip(offsets, colors, teacher_rows, strict=True):
        values = [
            float(by_teacher[name][prompt]["output_class_weighted_accuracy"])
            if prompt in by_teacher[name]
            else np.nan
            for prompt in filtered
        ]
        ax.barh(y + offset, values, height=bar_height, color=color, label=name)
    ax.set_yticks(y, labels=filtered, fontsize=6)
    if sort_by == "frequency":
        ax.invert_yaxis()
    ax.set_xlabel("Output-class-weighted accuracy within prompt")
    ax.set_xlim(0, 1)
    ax.set_title(title)
    for guide in (0.4, 0.5, 0.6):
        ax.axvline(guide, color="#666666", linewidth=0.8, linestyle="--", alpha=0.45, zorder=0)
    ax.legend(loc="lower right")
    fig.subplots_adjust(left=0.24, right=0.96, top=0.96, bottom=0.04)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_three_way_output_bar(
    teacher_rows: list[tuple[str, list[dict[str, Any]]]],
    title: str,
    output: Path,
    metric: str = "accuracy",
) -> None:
    labels = [str(row["answer"]) for row in teacher_rows[0][1]]
    by_teacher = {
        name: {str(row["answer"]): row for row in rows} for name, rows in teacher_rows
    }
    x = np.arange(len(labels))
    width = 0.24
    offsets = np.linspace(-width, width, len(teacher_rows))
    colors = ["#4c78a8", "#f58518", "#54a24b"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for offset, color, (name, _rows) in zip(offsets, colors, teacher_rows, strict=True):
        ax.bar(
            x + offset,
            [float(by_teacher[name][label][metric]) for label in labels],
            width=width,
            color=color,
            label=name,
        )
    ax.set_xticks(x, labels=labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy" if metric == "accuracy" else "Recall")
    ax.set_title(title)
    for guide in (0.4, 0.5, 0.6):
        ax.axhline(guide, color="#666666", linewidth=0.8, linestyle="--", alpha=0.45, zorder=0)
    ax.legend(loc="lower right")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def write_prompt_weights(path: Path, weights: dict[str, dict[str, float]]) -> None:
    rows = [
        {
            "student_prompt": prompt,
            "smolvlm_weight": values.get("smolvlm", 0.0),
            "fasterrcnn_weight": values.get("fasterrcnn", 0.0),
        }
        for prompt, values in sorted(weights.items())
    ]
    write_csv(path, rows)


def run_split(
    split: str,
    examples_path: Path,
    args: argparse.Namespace,
    labels: list[int | str],
    weights: dict[str, dict[str, float]],
    composite_name: str,
) -> dict[str, Any]:
    split_dir = args.output_dir / split
    allowed_indices, filter_metadata = load_allowed_dataset_indices(
        examples_path,
        args.reference_examples,
    )
    smol_base = stream_teacher(
        args.smolvlm_cache,
        "SmolVLM-256M",
        labels,
        args.collapse_at,
        allowed_indices,
    )
    faster_base = stream_teacher(
        args.fasterrcnn_cache,
        "Faster R-CNN COCO80",
        labels,
        args.collapse_at,
        allowed_indices,
    )
    assert allowed_indices is not None
    smol_dist = load_distribution_rows(
        args.smolvlm_cache,
        allowed_indices,
        args.collapse_at,
        args.answer_max,
        args.smolvlm_temperature,
    )
    faster_dist = load_distribution_rows(
        args.fasterrcnn_cache,
        allowed_indices,
        args.collapse_at,
        args.answer_max,
        args.fasterrcnn_temperature,
    )
    composite = evaluate_composite_teacher(
        smol_dist,
        faster_dist,
        weights,
        labels,
        args.collapse_at,
        composite_name,
    )

    teacher_results = [
        ("smolvlm256_stretched", smol_base),
        ("fasterrcnn_coco80_letterbox_score005_poibin", faster_base),
        ("composite_ece_temperature_scaled", composite),
    ]
    confusion_by_key = {}
    for key, result in teacher_results:
        confusion_by_key[key] = run_teacher_plots(
            result,
            labels,
            split_dir / key,
            args.min_prompt_count,
        )
    write_csv(
        split_dir / "composite_ece_temperature_scaled" / "tables" / "composite_predictions.csv",
        composite["prediction_rows"],
    )

    combined_figures = split_dir / "combined" / "figures"
    combined_tables = split_dir / "combined" / "tables"
    rows_for_combined = [
        (smol_base["name"], smol_base["by_prompt"]),
        (faster_base["name"], faster_base["by_prompt"]),
        (composite["name"], composite["by_prompt"]),
    ]
    output_rows_for_combined = [
        (smol_base["name"], smol_base["by_output"]),
        (faster_base["name"], faster_base["by_output"]),
        (composite["name"], composite["by_output"]),
    ]
    plot_three_way_prompt_bar(
        rows_for_combined,
        "Prompt Accuracy Balanced Across Output Classes",
        combined_figures / "prompt_output_class_weighted_accuracy_three_way.png",
        args.min_prompt_count,
        sort_teacher=composite["name"],
    )
    plot_three_way_prompt_bar(
        rows_for_combined,
        "Prompt Accuracy Balanced Across Output Classes, Ordered by Frequency",
        combined_figures / "prompt_output_class_weighted_accuracy_by_frequency_three_way.png",
        args.min_prompt_count,
        sort_teacher=composite["name"],
        sort_by="frequency",
    )
    plot_three_way_output_bar(
        output_rows_for_combined,
        "Accuracy by True Output Class",
        combined_figures / "output_class_accuracy_three_way.png",
    )
    plot_three_way_output_bar(
        output_rows_for_combined,
        "Recall by True Output Class",
        combined_figures / "output_class_recall_three_way.png",
        metric="recall",
    )
    plot_prompt_pair_bar(
        smol_base["by_prompt"],
        composite["by_prompt"],
        smol_base["name"],
        composite["name"],
        "Prompt Accuracy: SmolVLM vs Composite",
        combined_figures / "prompt_output_class_weighted_accuracy_smolvlm_vs_composite.png",
        args.min_prompt_count,
    )
    plot_prompt_pair_bar(
        faster_base["by_prompt"],
        composite["by_prompt"],
        faster_base["name"],
        composite["name"],
        "Prompt Accuracy: Faster R-CNN vs Composite",
        combined_figures / "prompt_output_class_weighted_accuracy_fasterrcnn_vs_composite.png",
        args.min_prompt_count,
    )
    plot_output_pair_bar(
        smol_base["by_output"],
        composite["by_output"],
        smol_base["name"],
        composite["name"],
        "Accuracy by True Output Class: SmolVLM vs Composite",
        combined_figures / "output_class_accuracy_smolvlm_vs_composite.png",
    )
    plot_output_pair_bar(
        faster_base["by_output"],
        composite["by_output"],
        faster_base["name"],
        composite["name"],
        "Accuracy by True Output Class: Faster R-CNN vs Composite",
        combined_figures / "output_class_accuracy_fasterrcnn_vs_composite.png",
    )
    plot_confusion_pair(
        confusion_by_key["smolvlm256_stretched"],
        confusion_by_key["composite_ece_temperature_scaled"],
        labels,
        smol_base["name"],
        composite["name"],
        combined_figures / "output_confusion_matrices_smolvlm_vs_composite.png",
    )
    plot_confusion_pair(
        confusion_by_key["fasterrcnn_coco80_letterbox_score005_poibin"],
        confusion_by_key["composite_ece_temperature_scaled"],
        labels,
        faster_base["name"],
        composite["name"],
        combined_figures / "output_confusion_matrices_fasterrcnn_vs_composite.png",
    )

    write_csv(
        combined_tables / "prompt_output_class_weighted_accuracy_all_teachers.csv",
        smol_base["by_prompt"] + faster_base["by_prompt"] + composite["by_prompt"],
    )
    write_csv(
        combined_tables / "output_class_accuracy_all_teachers.csv",
        smol_base["by_output"] + faster_base["by_output"] + composite["by_output"],
    )
    write_csv(
        combined_tables / "output_class_recall_all_teachers.csv",
        smol_base["by_output"] + faster_base["by_output"] + composite["by_output"],
    )

    return {
        "filter": filter_metadata,
        "teacher_summaries": {
            key: {
                "name": result["name"],
                "records": result["records"],
                "prompt_classes": len(result["by_prompt"]),
                "overall_accuracy": result["overall_accuracy"],
                "overall_output_class_weighted_accuracy": result[
                    "overall_output_class_weighted_accuracy"
                ],
            }
            for key, result in teacher_results
        },
        "composite_source_counts": composite["source_counts"],
        "artifacts": {
            "combined_figures": str(combined_figures),
            "combined_tables": str(combined_tables),
            "composite_figures": str(split_dir / "composite_ece_temperature_scaled" / "figures"),
            "composite_tables": str(split_dir / "composite_ece_temperature_scaled" / "tables"),
        },
    }


def main() -> None:
    args = parse_args()
    if args.collapse_at is not None and args.collapse_at < args.answer_min:
        raise ValueError("--collapse-at must be >= --answer-min.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = output_classes(args.answer_min, args.answer_max, args.collapse_at)
    prompt_accuracies = load_prompt_accuracies(args.selection_csv)
    weights = weights_for_beta(prompt_accuracies, args.composite_beta, args.chance_baseline)
    write_prompt_weights(args.output_dir / "tables" / "prompt_teacher_weights.csv", weights)

    composite_name = (
        "Composite ECE-temp "
        f"(Smol T={args.smolvlm_temperature:g}, FRCNN T={args.fasterrcnn_temperature:g}, "
        f"beta={args.composite_beta:g})"
    )
    split_summaries = {
        "trainval": run_split(
            "trainval",
            args.trainval_examples,
            args,
            labels,
            weights,
            composite_name,
        ),
        "test": run_split(
            "test",
            args.test_examples,
            args,
            labels,
            weights,
            composite_name,
        ),
    }
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "purpose": (
            "Final EDA for the selected composite teacher, compared against the two base "
            "teachers on the locked final train/val and test splits."
        ),
        "selected_composite_parameters": {
            "smolvlm_temperature": args.smolvlm_temperature,
            "fasterrcnn_temperature": args.fasterrcnn_temperature,
            "composite_beta": args.composite_beta,
            "chance_baseline": args.chance_baseline,
            "weight_formula": (
                "w_t = exp(beta * (prompt_output_class_weighted_accuracy_t - 1/6)) / Z"
            ),
            "temperature_application": (
                "Per-teacher temperatures are applied to uncollapsed numeric candidate "
                "probabilities before teacher fusion."
            ),
            "selection_source": str(args.selection_csv),
        },
        "output_classes": [str(label) for label in labels],
        "splits": split_summaries,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
