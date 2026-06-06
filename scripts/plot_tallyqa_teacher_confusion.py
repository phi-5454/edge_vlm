from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import json
import os
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_CACHE = Path(
    "artifacts/teacher_cache/smolvlm2_2p2b_tallyqa_target_mobilenet224_letterbox.calibration4096.jsonl"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/reports/tallyqa_teacher_confusion_smolvlm2_2p2b_letterbox")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a TallyQA teacher output-class confusion matrix."
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--answer-min", type=int, default=0)
    parser.add_argument("--answer-max", type=int, default=15)
    parser.add_argument(
        "--collapse-at",
        type=int,
        default=5,
        help="Collapse true/predicted output classes >= this value into a '<n>+' class.",
    )
    parser.add_argument(
        "--min-prompt-accuracy",
        type=float,
        default=None,
        help="Only include records from prompt classes with overall prompt accuracy at least this value.",
    )
    parser.add_argument(
        "--prompt-class-names-file",
        type=Path,
        default=None,
        help="Optional newline-delimited prompt class names to include.",
    )
    parser.add_argument("--title", default="SmolVLM2-2.2B TallyQA Output Confusion Matrix")
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


def stream_records(cache: Path, collapse_at: int | None) -> list[tuple[str, int | str, int | str]]:
    records: list[tuple[str, int | str, int | str]] = []
    with cache.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {cache}:{line_number}") from exc
            records.append(
                (
                    str(row["student_prompt"]),
                    output_class(int(row["answer"]), collapse_at),
                    output_class(
                        int(row["teacher_metrics"]["numeric_answer"]["prediction"]),
                        collapse_at,
                    ),
                )
            )
    return records


def eligible_prompts(
    records: list[tuple[str, int | str, int | str]],
    min_accuracy: float | None,
) -> set[str] | None:
    if min_accuracy is None:
        return None
    by_prompt: dict[str, Counter] = {}
    for prompt, true_label, pred_label in records:
        counter = by_prompt.setdefault(prompt, Counter())
        counter["total"] += 1
        counter["correct"] += int(true_label == pred_label)
    return {
        prompt
        for prompt, counter in by_prompt.items()
        if counter["total"] and counter["correct"] / counter["total"] >= min_accuracy
    }


def load_prompt_names(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def build_confusion(
    records: list[tuple[str, int | str, int | str]],
    prompts: set[str] | None,
) -> tuple[Counter, int, int]:
    confusion: Counter[tuple[int | str, int | str]] = Counter()
    record_count = 0
    correct = 0
    for prompt, true_label, pred_label in records:
        if prompts is not None and prompt not in prompts:
            continue
        confusion[(true_label, pred_label)] += 1
        record_count += 1
        correct += int(true_label == pred_label)
    return confusion, record_count, correct


def build_matrices(
    confusion: Counter,
    labels: list[int | str],
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.zeros((len(labels), len(labels)), dtype=np.int64)
    index_by_label = {label: index for index, label in enumerate(labels)}
    for (true_label, pred_label), count in confusion.items():
        if true_label not in index_by_label or pred_label not in index_by_label:
            continue
        counts[index_by_label[true_label], index_by_label[pred_label]] = int(count)
    row_totals = counts.sum(axis=1, keepdims=True)
    normalized = np.divide(
        counts,
        row_totals,
        out=np.zeros_like(counts, dtype=np.float64),
        where=row_totals > 0,
    )
    return counts, normalized


def plot_confusion(
    counts: np.ndarray,
    normalized: np.ndarray,
    labels: list[int | str],
    title: str,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = plt.get_cmap("Blues")
    image = ax.imshow(normalized, vmin=0, vmax=1, cmap=cmap)
    label_text = [str(label) for label in labels]
    ax.set_xticks(np.arange(len(labels)), labels=label_text)
    ax.set_yticks(np.arange(len(labels)), labels=label_text)
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
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def write_matrix_csv(
    path: Path,
    counts: np.ndarray,
    normalized: np.ndarray,
    labels: list[int | str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["true_label", "predicted_label", "count", "row_fraction"],
        )
        writer.writeheader()
        for row_index, true_label in enumerate(labels):
            for col_index, pred_label in enumerate(labels):
                writer.writerow(
                    {
                        "true_label": true_label,
                        "predicted_label": pred_label,
                        "count": int(counts[row_index, col_index]),
                        "row_fraction": float(normalized[row_index, col_index]),
                    }
                )


def main() -> None:
    args = parse_args()
    if args.collapse_at is not None and args.collapse_at < args.answer_min:
        raise ValueError("--collapse-at must be >= --answer-min")
    labels = output_classes(args.answer_min, args.answer_max, args.collapse_at)
    all_records = stream_records(args.cache, args.collapse_at)
    prompts = eligible_prompts(all_records, args.min_prompt_accuracy)
    if args.prompt_class_names_file is not None:
        file_prompts = load_prompt_names(args.prompt_class_names_file)
        prompts = file_prompts if prompts is None else prompts & file_prompts
    confusion, records, correct = build_confusion(all_records, prompts)
    counts, normalized = build_matrices(confusion, labels)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    figure_path = args.output_dir / "teacher_output_confusion_matrix.png"
    table_path = args.output_dir / "teacher_output_confusion_matrix.csv"
    plot_confusion(counts, normalized, labels, args.title, figure_path)
    write_matrix_csv(table_path, counts, normalized, labels)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cache": str(args.cache),
        "records": records,
        "correct": correct,
        "accuracy": correct / records if records else None,
        "answer_min": args.answer_min,
        "answer_max": args.answer_max,
        "collapse_at": args.collapse_at,
        "min_prompt_accuracy": args.min_prompt_accuracy,
        "prompt_class_names_file": (
            str(args.prompt_class_names_file) if args.prompt_class_names_file is not None else None
        ),
        "prompt_classes_after_filter": len(prompts) if prompts is not None else None,
        "output_classes": [str(label) for label in labels],
        "figure": str(figure_path),
        "table": str(table_path),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
