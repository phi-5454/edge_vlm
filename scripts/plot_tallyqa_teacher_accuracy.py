from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_CACHE = Path("artifacts/teacher_cache/smolvlm_tallyqa_target_mobilenet224.jsonl")
DEFAULT_CLASSES = Path("data/tallyqa_cauldron_target_mobilenet224/classes.json")
DEFAULT_OUTPUT_DIR = Path("artifacts/reports/tallyqa_teacher_accuracy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot TallyQA teacher accuracy by class.")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--classes", type=Path, default=DEFAULT_CLASSES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--answer-min", type=int, default=0)
    parser.add_argument("--answer-max", type=int, default=15)
    parser.add_argument(
        "--collapse-at",
        type=int,
        default=None,
        help="Collapse true/predicted output classes >= this value into a '<n>+' class.",
    )
    parser.add_argument("--min-heatmap-count", type=int, default=1)
    return parser.parse_args()


def load_class_rows(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return sorted(rows, key=lambda row: int(row["class_id"]))


def output_class(answer: int, collapse_at: int | None) -> int | str:
    if collapse_at is not None and answer >= collapse_at:
        return f"{collapse_at}+"
    return answer


def stream_accuracy(cache_path: Path, collapse_at: int | None) -> dict[str, Any]:
    overall = Counter()
    by_prompt: dict[str, Counter] = defaultdict(Counter)
    by_answer: dict[int | str, Counter] = defaultdict(Counter)
    by_prompt_answer: dict[tuple[str, int | str], Counter] = defaultdict(Counter)

    with cache_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {cache_path}:{line_number}") from exc
            prompt = str(row["student_prompt"])
            answer = output_class(int(row["answer"]), collapse_at)
            prediction = output_class(
                int(row["teacher_metrics"]["numeric_answer"]["prediction"]),
                collapse_at,
            )
            correct = prediction == answer
            increment = {"total": 1, "correct": int(correct)}
            overall.update(increment)
            by_prompt[prompt].update(increment)
            by_answer[answer].update(increment)
            by_prompt_answer[(prompt, answer)].update(increment)

    return {
        "overall": overall,
        "by_prompt": by_prompt,
        "by_answer": by_answer,
        "by_prompt_answer": by_prompt_answer,
    }


def accuracy(counter: Counter) -> float:
    total = int(counter["total"])
    return float(counter["correct"]) / total if total else float("nan")


def answer_classes(answer_min: int, answer_max: int, collapse_at: int | None) -> list[int | str]:
    if collapse_at is None:
        return list(range(answer_min, answer_max + 1))
    if collapse_at <= answer_min:
        return [f"{collapse_at}+"]
    return list(range(answer_min, min(answer_max, collapse_at - 1) + 1)) + [f"{collapse_at}+"]


def build_heatmap(
    class_rows: list[dict[str, Any]],
    answers: list[int | str],
    stats: dict[str, Any],
    min_count: int,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    prompts = [str(row["item"]) for row in class_rows]
    row_labels = prompts + ["Overall"]
    col_labels = [str(answer) for answer in answers] + ["Overall"]
    values = np.full((len(row_labels), len(col_labels)), np.nan, dtype=np.float32)
    counts = np.zeros((len(row_labels), len(col_labels)), dtype=np.int64)

    for row_index, prompt in enumerate(prompts):
        for col_index, answer in enumerate(answers):
            counter = stats["by_prompt_answer"].get((prompt, answer), Counter())
            counts[row_index, col_index] = int(counter["total"])
            if counter["total"] >= min_count:
                values[row_index, col_index] = accuracy(counter)
        prompt_counter = stats["by_prompt"].get(prompt, Counter())
        counts[row_index, -1] = int(prompt_counter["total"])
        values[row_index, -1] = accuracy(prompt_counter)

    for col_index, answer in enumerate(answers):
        answer_counter = stats["by_answer"].get(answer, Counter())
        counts[-1, col_index] = int(answer_counter["total"])
        values[-1, col_index] = accuracy(answer_counter)
    counts[-1, -1] = int(stats["overall"]["total"])
    values[-1, -1] = accuracy(stats["overall"])
    return values, counts, row_labels, col_labels


def plot_heatmap(
    values: np.ndarray,
    counts: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    output: Path,
) -> None:
    height = max(14, len(row_labels) * 0.24)
    fig, ax = plt.subplots(figsize=(13, height))
    masked = np.ma.masked_invalid(values)
    image = ax.imshow(masked, vmin=0, vmax=1, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(col_labels)), labels=col_labels)
    ax.set_yticks(np.arange(len(row_labels)), labels=row_labels, fontsize=6)
    ax.tick_params(axis="x", labelrotation=0)
    ax.set_xlabel("True output count class")
    ax.set_ylabel("Input prompt class")
    ax.set_title("SmolVLM TallyQA Accuracy by Prompt Class and Output Count")
    cbar = fig.colorbar(image, ax=ax, shrink=0.7)
    cbar.set_label("Accuracy")
    fig.subplots_adjust(left=0.24, right=0.94, top=0.96, bottom=0.04)

    for row_index in range(values.shape[0]):
        for col_index in range(values.shape[1]):
            if counts[row_index, col_index] == 0:
                continue
            is_margin = row_index == values.shape[0] - 1 or col_index == values.shape[1] - 1
            if is_margin:
                ax.text(
                    col_index,
                    row_index,
                    f"{values[row_index, col_index]:.2f}\n{counts[row_index, col_index]}",
                    ha="center",
                    va="center",
                    fontsize=4.8,
                    color="white" if values[row_index, col_index] < 0.65 else "black",
                )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_bar(
    labels: list[str],
    accuracies: list[float],
    counts: list[int],
    overall_accuracy: float,
    title: str,
    xlabel: str,
    output: Path,
    horizontal: bool,
) -> None:
    if horizontal:
        fig_height = max(10, len(labels) * 0.22)
        fig, ax = plt.subplots(figsize=(12, fig_height))
        y = np.arange(len(labels))
        colors = ["#4c78a8" if label != "Overall" else "#f58518" for label in labels]
        ax.barh(y, accuracies, color=colors)
        ax.set_yticks(y, labels=labels, fontsize=6)
        ax.invert_yaxis()
        ax.axvline(overall_accuracy, color="black", linewidth=1, linestyle="--", label="Overall")
        ax.set_xlabel(xlabel)
        ax.set_xlim(0, 1)
        fig.subplots_adjust(left=0.24, right=0.94, top=0.96, bottom=0.04)
    else:
        fig, ax = plt.subplots(figsize=(13, 5))
        x = np.arange(len(labels))
        colors = ["#4c78a8" if label != "Overall" else "#f58518" for label in labels]
        ax.bar(x, accuracies, color=colors)
        ax.set_xticks(x, labels=labels)
        ax.axhline(overall_accuracy, color="black", linewidth=1, linestyle="--", label="Overall")
        ax.set_ylabel(xlabel)
        ax.set_ylim(0, 1)
        fig.subplots_adjust(left=0.06, right=0.98, top=0.9, bottom=0.12)
    ax.set_title(title)
    ax.legend(loc="lower right")
    for index, (acc, count) in enumerate(zip(accuracies, counts, strict=True)):
        if horizontal:
            ax.text(acc + 0.005, index, f"{acc:.2f} n={count}", va="center", fontsize=5)
        else:
            ax.text(index, acc + 0.015, f"{acc:.2f}\nn={count}", ha="center", fontsize=6)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0])
    lines = [",".join(columns)]
    for row in rows:
        lines.append(",".join(str(row[column]) for column in columns))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.collapse_at is not None and args.collapse_at < args.answer_min:
        raise ValueError("--collapse-at must be >= --answer-min")
    class_rows = load_class_rows(args.classes)
    answers = answer_classes(args.answer_min, args.answer_max, args.collapse_at)
    stats = stream_accuracy(args.cache, args.collapse_at)
    overall_accuracy = accuracy(stats["overall"])

    heatmap, heatmap_counts, row_labels, col_labels = build_heatmap(
        class_rows,
        answers,
        stats,
        args.min_heatmap_count,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_heatmap(
        heatmap,
        heatmap_counts,
        row_labels,
        col_labels,
        args.output_dir / "teacher_accuracy_prompt_by_output_heatmap.png",
    )

    prompt_rows = [
        {
            "student_prompt": str(row["item"]),
            "item_class_id": int(row["class_id"]),
            "count": int(stats["by_prompt"][str(row["item"])]["total"]),
            "correct": int(stats["by_prompt"][str(row["item"])]["correct"]),
            "accuracy": accuracy(stats["by_prompt"][str(row["item"])]),
        }
        for row in class_rows
    ]
    answer_rows = [
        {
            "answer": answer,
            "count": int(stats["by_answer"][answer]["total"]),
            "correct": int(stats["by_answer"][answer]["correct"]),
            "accuracy": accuracy(stats["by_answer"][answer]),
        }
        for answer in answers
    ]
    prompt_plot_rows = prompt_rows + [
        {
            "student_prompt": "Overall",
            "item_class_id": "",
            "count": int(stats["overall"]["total"]),
            "correct": int(stats["overall"]["correct"]),
            "accuracy": overall_accuracy,
        }
    ]
    answer_plot_rows = answer_rows + [
        {
            "answer": "Overall",
            "count": int(stats["overall"]["total"]),
            "correct": int(stats["overall"]["correct"]),
            "accuracy": overall_accuracy,
        }
    ]

    plot_bar(
        labels=[str(row["student_prompt"]) for row in prompt_plot_rows],
        accuracies=[float(row["accuracy"]) for row in prompt_plot_rows],
        counts=[int(row["count"]) for row in prompt_plot_rows],
        overall_accuracy=overall_accuracy,
        title="SmolVLM TallyQA Accuracy by Input Prompt Class",
        xlabel="Accuracy",
        output=args.output_dir / "teacher_accuracy_by_prompt_class_bar.png",
        horizontal=True,
    )
    plot_bar(
        labels=[str(row["answer"]) for row in answer_plot_rows],
        accuracies=[float(row["accuracy"]) for row in answer_plot_rows],
        counts=[int(row["count"]) for row in answer_plot_rows],
        overall_accuracy=overall_accuracy,
        title="SmolVLM TallyQA Accuracy by Output Count Class",
        xlabel="Accuracy",
        output=args.output_dir / "teacher_accuracy_by_output_class_bar.png",
        horizontal=False,
    )

    write_csv(args.output_dir / "teacher_accuracy_by_prompt_class.csv", prompt_rows)
    write_csv(args.output_dir / "teacher_accuracy_by_output_class.csv", answer_rows)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cache": str(args.cache),
        "classes": str(args.classes),
        "records": int(stats["overall"]["total"]),
        "correct": int(stats["overall"]["correct"]),
        "overall_accuracy": overall_accuracy,
        "answer_min": args.answer_min,
        "answer_max": args.answer_max,
        "collapse_at": args.collapse_at,
        "output_classes": [str(answer) for answer in answers],
        "min_heatmap_count": args.min_heatmap_count,
        "figures": {
            "prompt_by_output_heatmap": str(
                args.output_dir / "teacher_accuracy_prompt_by_output_heatmap.png"
            ),
            "by_prompt_class_bar": str(args.output_dir / "teacher_accuracy_by_prompt_class_bar.png"),
            "by_output_class_bar": str(args.output_dir / "teacher_accuracy_by_output_class_bar.png"),
        },
        "tables": {
            "by_prompt_class": str(args.output_dir / "teacher_accuracy_by_prompt_class.csv"),
            "by_output_class": str(args.output_dir / "teacher_accuracy_by_output_class.csv"),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
