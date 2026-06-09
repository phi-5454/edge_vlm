from __future__ import annotations

import argparse
import csv
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
    update_counter,
)


DEFAULT_OUTPUT_DIR = Path(
    "artifacts/reports/final_dataset/post_pruning_teacher_eda/"
    "composite_teacher_argmax_prompt_accuracy_comparison"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the soft composite teacher against a hard argmax selector that chooses "
            "the better teacher per prompt class from train/val output-class-weighted accuracy."
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
        "--composite-manifest",
        type=Path,
        default=Path(
            "artifacts/reports/final_dataset/post_pruning_teacher_eda/"
            "composite_teacher_softmax_prompt_accuracy_beta5/manifest.json"
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
    parser.add_argument("--collapse-at", type=int, default=5)
    parser.add_argument("--answer-min", type=int, default=0)
    parser.add_argument("--answer-max", type=int, default=15)
    return parser.parse_args()


def teacher_key(name: str) -> str:
    lowered = name.lower()
    if "smol" in lowered:
        return "smolvlm"
    if "faster" in lowered or "rcnn" in lowered:
        return "fasterrcnn"
    raise ValueError(f"Unrecognized teacher name: {name}")


def load_prompt_selector(selection_csv: Path) -> tuple[dict[str, str], list[dict[str, Any]]]:
    per_prompt: dict[str, dict[str, float]] = {}
    with selection_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            prompt = row["student_prompt"]
            per_prompt.setdefault(prompt, {})[teacher_key(row["teacher"])] = float(
                row["output_class_weighted_accuracy"]
            )

    selector: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    for prompt, values in sorted(per_prompt.items()):
        selected = max(values.items(), key=lambda item: (item[1], item[0]))[0]
        selector[prompt] = selected
        rows.append(
            {
                "student_prompt": prompt,
                "selected_teacher": selected,
                "smolvlm_output_class_weighted_accuracy": values.get("smolvlm", float("nan")),
                "fasterrcnn_output_class_weighted_accuracy": values.get("fasterrcnn", float("nan")),
            }
        )
    return selector, rows


def load_predictions(
    cache: Path,
    allowed_dataset_indices: set[int],
    collapse_at: int,
) -> dict[int, dict[str, Any]]:
    predictions: dict[int, dict[str, Any]] = {}
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
            predictions[dataset_index] = {
                "dataset_index": dataset_index,
                "student_prompt": str(row["student_prompt"]),
                "answer": int(row["answer"]),
                "true_label": output_class(int(row["answer"]), collapse_at),
                "prediction": int(row["teacher_metrics"]["numeric_answer"]["prediction"]),
                "pred_label": output_class(
                    int(row["teacher_metrics"]["numeric_answer"]["prediction"]),
                    collapse_at,
                ),
            }
    return predictions


def evaluate_argmax_selector(
    selector: dict[str, str],
    smol: dict[int, dict[str, Any]],
    faster: dict[int, dict[str, Any]],
    labels: list[int | str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    overall = init_counter()
    by_output = {label: init_counter() for label in labels}
    by_prompt: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    source_counts = {
        "selected_smolvlm": 0,
        "selected_fasterrcnn": 0,
        "fallback_smolvlm_missing_fasterrcnn_row": 0,
    }

    for dataset_index, smol_row in sorted(smol.items()):
        prompt = smol_row["student_prompt"]
        selected = selector.get(prompt, "smolvlm")
        if selected == "fasterrcnn" and dataset_index in faster:
            row = faster[dataset_index]
            source = "fasterrcnn"
            source_counts["selected_fasterrcnn"] += 1
        elif selected == "fasterrcnn":
            row = smol_row
            source = "smolvlm_fallback"
            source_counts["fallback_smolvlm_missing_fasterrcnn_row"] += 1
        else:
            row = smol_row
            source = "smolvlm"
            source_counts["selected_smolvlm"] += 1

        correct = row["pred_label"] == smol_row["true_label"]
        update_counter(overall, correct)
        update_counter(by_output[smol_row["true_label"]], correct)
        prompt_state = by_prompt.setdefault(
            prompt,
            {
                "overall": init_counter(),
                "by_output": {label: init_counter() for label in labels},
            },
        )
        update_counter(prompt_state["overall"], correct)
        update_counter(prompt_state["by_output"][smol_row["true_label"]], correct)
        rows.append(
            {
                "dataset_index": dataset_index,
                "student_prompt": prompt,
                "answer": smol_row["answer"],
                "prediction": row["prediction"],
                "selected_teacher": source,
                "correct": int(correct),
            }
        )

    prompt_rows = []
    for prompt, state in sorted(by_prompt.items()):
        present_outputs = {
            label: counter
            for label, counter in state["by_output"].items()
            if int(counter["total"]) > 0
        }
        prompt_rows.append(
            {
                "student_prompt": prompt,
                "count": int(state["overall"]["total"]),
                "correct": int(state["overall"]["correct"]),
                "accuracy": accuracy(state["overall"]),
                "output_class_weighted_accuracy": balanced_accuracy(present_outputs),
                "output_classes_present": len(present_outputs),
            }
        )

    metrics = {
        "records": int(overall["total"]),
        "overall_accuracy": accuracy(overall),
        "overall_output_class_weighted_accuracy": balanced_accuracy(by_output),
        "mean_prompt_output_class_weighted_accuracy": float(
            np.mean([row["output_class_weighted_accuracy"] for row in prompt_rows])
        ),
        "prompt_classes": len(prompt_rows),
        "source_counts": source_counts,
        "prompt_rows": prompt_rows,
        "prediction_rows": rows,
    }
    return metrics, rows


def read_composite_metrics(composite_manifest: Path) -> dict[str, Any]:
    with composite_manifest.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    return manifest["splits"]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_comparison(rows: list[dict[str, Any]], output_path: Path) -> None:
    labels = [row["split"] for row in rows]
    metrics = [
        ("overall_accuracy", "Overall accuracy"),
        ("overall_output_class_weighted_accuracy", "Output-class weighted"),
        ("mean_prompt_output_class_weighted_accuracy", "Mean prompt weighted"),
    ]
    x = np.arange(len(metrics))
    width = 0.22
    fig, axes = plt.subplots(1, len(rows), figsize=(10, 4), sharey=True)
    if len(rows) == 1:
        axes = [axes]
    for ax, row, label in zip(axes, rows, labels, strict=True):
        argmax_values = [row[f"argmax_{key}"] for key, _ in metrics]
        composite_values = [row[f"composite_{key}"] for key, _ in metrics]
        smol_values = [row[f"smol_{key}"] for key, _ in metrics]
        faster_values = [row[f"faster_{key}"] for key, _ in metrics]
        ax.bar(x - 1.5 * width, smol_values, width, label="SmolVLM")
        ax.bar(x - 0.5 * width, faster_values, width, label="Faster R-CNN")
        ax.bar(x + 0.5 * width, argmax_values, width, label="Argmax selector")
        ax.bar(x + 1.5 * width, composite_values, width, label="Soft composite beta=5")
        ax.set_title(label)
        ax.set_xticks(x)
        ax.set_xticklabels([title for _, title in metrics], rotation=25, ha="right")
        ax.set_ylim(0, 0.8)
        for guide in (0.5, 0.6, 0.7):
            ax.axhline(guide, color="0.75", linestyle="--", linewidth=0.8, zorder=0)
    axes[0].set_ylabel("Accuracy")
    axes[-1].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    labels = output_classes(args.answer_min, args.answer_max, args.collapse_at)
    selector, selector_rows = load_prompt_selector(args.selection_csv)
    composite_splits = read_composite_metrics(args.composite_manifest)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "tables" / "prompt_argmax_teacher_selection.csv", selector_rows)

    split_inputs = {
        "trainval": args.trainval_examples,
        "test": args.test_examples,
    }
    summary_rows = []
    manifest: dict[str, Any] = {
        "selection_source": str(args.selection_csv),
        "selection_metric": "trainval prompt output-class-weighted accuracy",
        "argmax_rule": (
            "For each prompt class, choose the teacher with higher trainval "
            "prompt output-class-weighted accuracy; if Faster R-CNN was selected but the row "
            "is unavailable, fall back to SmolVLM."
        ),
        "composite_manifest": str(args.composite_manifest),
        "splits": {},
    }

    for split, examples_path in split_inputs.items():
        allowed_indices, filter_meta = load_allowed_dataset_indices(
            examples_path,
            args.reference_examples,
        )
        assert allowed_indices is not None
        smol = load_predictions(args.smolvlm_cache, allowed_indices, args.collapse_at)
        faster = load_predictions(args.fasterrcnn_cache, allowed_indices, args.collapse_at)
        argmax_metrics, prediction_rows = evaluate_argmax_selector(selector, smol, faster, labels)

        split_dir = args.output_dir / split
        write_csv(split_dir / "tables" / "argmax_predictions.csv", prediction_rows)
        write_csv(
            split_dir / "tables" / "prompt_output_class_weighted_accuracy.csv",
            argmax_metrics["prompt_rows"],
        )

        composite_overall = composite_splits[split]["overall"]["Composite softmax beta=5"]
        smol_overall = composite_splits[split]["overall"]["SmolVLM-256M"]
        faster_overall = composite_splits[split]["overall"]["Faster R-CNN COCO80"]
        composite_prompt_rows_path = Path(
            composite_splits[split]["artifacts"]["three_way_prompt_table"]
        )
        composite_prompt_values = []
        with composite_prompt_rows_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["teacher"] == "Composite softmax beta=5":
                    composite_prompt_values.append(float(row["output_class_weighted_accuracy"]))
        smol_prompt_values = []
        faster_prompt_values = []
        with composite_prompt_rows_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["teacher"] == "SmolVLM-256M":
                    smol_prompt_values.append(float(row["output_class_weighted_accuracy"]))
                elif row["teacher"] == "Faster R-CNN COCO80":
                    faster_prompt_values.append(float(row["output_class_weighted_accuracy"]))

        summary_row = {
            "split": split,
            "smol_overall_accuracy": smol_overall["overall_accuracy"],
            "smol_overall_output_class_weighted_accuracy": smol_overall[
                "overall_output_class_weighted_accuracy"
            ],
            "smol_mean_prompt_output_class_weighted_accuracy": float(np.mean(smol_prompt_values)),
            "faster_overall_accuracy": faster_overall["overall_accuracy"],
            "faster_overall_output_class_weighted_accuracy": faster_overall[
                "overall_output_class_weighted_accuracy"
            ],
            "faster_mean_prompt_output_class_weighted_accuracy": float(
                np.mean(faster_prompt_values)
            ),
            "argmax_overall_accuracy": argmax_metrics["overall_accuracy"],
            "argmax_overall_output_class_weighted_accuracy": argmax_metrics[
                "overall_output_class_weighted_accuracy"
            ],
            "argmax_mean_prompt_output_class_weighted_accuracy": argmax_metrics[
                "mean_prompt_output_class_weighted_accuracy"
            ],
            "composite_overall_accuracy": composite_overall["overall_accuracy"],
            "composite_overall_output_class_weighted_accuracy": composite_overall[
                "overall_output_class_weighted_accuracy"
            ],
            "composite_mean_prompt_output_class_weighted_accuracy": float(
                np.mean(composite_prompt_values)
            ),
        }
        summary_rows.append(summary_row)
        manifest["splits"][split] = {
            "filter": filter_meta,
            "smolvlm_records": len(smol),
            "fasterrcnn_records": len(faster),
            "argmax_records": argmax_metrics["records"],
            "argmax_metrics": {
                key: value
                for key, value in argmax_metrics.items()
                if key not in {"prompt_rows", "prediction_rows"}
            },
        }

    write_csv(args.output_dir / "tables" / "metric_comparison.csv", summary_rows)
    plot_comparison(summary_rows, args.output_dir / "figures" / "argmax_vs_soft_composite.png")
    manifest["summary_table"] = str(args.output_dir / "tables" / "metric_comparison.csv")
    manifest["comparison_plot"] = str(args.output_dir / "figures" / "argmax_vs_soft_composite.png")
    manifest["created_at_utc"] = __import__("datetime").datetime.now(
        __import__("datetime").timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


if __name__ == "__main__":
    main()
