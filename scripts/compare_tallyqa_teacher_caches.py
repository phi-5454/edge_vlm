from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ["MPLBACKEND"] = "Agg"

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_BASELINE = Path("artifacts/teacher_cache/smolvlm_tallyqa_target_mobilenet224.jsonl")
DEFAULT_OUTPUT_DIR = Path("artifacts/reports/tallyqa_teacher_cache_comparison")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two TallyQA teacher caches on their intersecting dataset indices."
    )
    parser.add_argument("--baseline-cache", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--candidate-cache", type=Path, required=True)
    parser.add_argument("--baseline-name", default="SmolVLM-256M")
    parser.add_argument("--candidate-name", default="SmolVLM-2.2B")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--collapse-at",
        type=int,
        default=5,
        help="Collapse true/predicted output classes >= this value into a '<n>+' class.",
    )
    parser.add_argument("--min-group-count", type=int, default=10)
    return parser.parse_args()


def output_class(answer: int, collapse_at: int | None) -> int | str:
    if collapse_at is not None and answer >= collapse_at:
        return f"{collapse_at}+"
    return answer


def read_cache(
    path: Path,
    keep_indices: set[int] | None = None,
) -> tuple[dict[int, dict[str, Any]], int]:
    rows: dict[int, dict[str, Any]] = {}
    total_records = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            total_records += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            dataset_index = int(row["dataset_index"])
            if keep_indices is not None and dataset_index not in keep_indices:
                continue
            rows[dataset_index] = row
    return rows, total_records


def init_counter() -> Counter:
    return Counter(
        {
            "total": 0,
            "baseline_correct": 0,
            "candidate_correct": 0,
            "both_correct": 0,
            "both_wrong": 0,
            "baseline_only": 0,
            "candidate_only": 0,
        }
    )


def update_counter(counter: Counter, baseline_correct: bool, candidate_correct: bool) -> None:
    counter["total"] += 1
    counter["baseline_correct"] += int(baseline_correct)
    counter["candidate_correct"] += int(candidate_correct)
    counter["both_correct"] += int(baseline_correct and candidate_correct)
    counter["both_wrong"] += int((not baseline_correct) and (not candidate_correct))
    counter["baseline_only"] += int(baseline_correct and not candidate_correct)
    counter["candidate_only"] += int(candidate_correct and not baseline_correct)


def summarize(counter: Counter) -> dict[str, Any]:
    total = int(counter["total"])
    baseline_accuracy = counter["baseline_correct"] / total if total else None
    candidate_accuracy = counter["candidate_correct"] / total if total else None
    delta = (
        candidate_accuracy - baseline_accuracy
        if baseline_accuracy is not None and candidate_accuracy is not None
        else None
    )
    discordant = int(counter["baseline_only"] + counter["candidate_only"])
    if discordant:
        statistic = (abs(counter["candidate_only"] - counter["baseline_only"]) - 1) ** 2 / discordant
        p_value = math.erfc(math.sqrt(statistic / 2))
    else:
        statistic = None
        p_value = None
    return {
        "count": total,
        "baseline_correct": int(counter["baseline_correct"]),
        "candidate_correct": int(counter["candidate_correct"]),
        "baseline_accuracy": baseline_accuracy,
        "candidate_accuracy": candidate_accuracy,
        "accuracy_delta": delta,
        "both_correct": int(counter["both_correct"]),
        "both_wrong": int(counter["both_wrong"]),
        "baseline_only": int(counter["baseline_only"]),
        "candidate_only": int(counter["candidate_only"]),
        "mcnemar_chi2_cc": statistic,
        "mcnemar_p_value_approx": p_value,
    }


def compare(
    baseline_rows: dict[int, dict[str, Any]],
    candidate_rows: dict[int, dict[str, Any]],
    collapse_at: int | None,
) -> dict[str, Any]:
    common_indices = sorted(set(baseline_rows) & set(candidate_rows))
    overall = init_counter()
    by_prompt: dict[str, Counter] = defaultdict(init_counter)
    by_output: dict[int | str, Counter] = defaultdict(init_counter)
    mismatches: list[dict[str, Any]] = []

    for index in common_indices:
        baseline = baseline_rows[index]
        candidate = candidate_rows[index]
        true_answer = output_class(int(baseline["answer"]), collapse_at)
        baseline_prediction = output_class(
            int(baseline["teacher_metrics"]["numeric_answer"]["prediction"]),
            collapse_at,
        )
        candidate_prediction = output_class(
            int(candidate["teacher_metrics"]["numeric_answer"]["prediction"]),
            collapse_at,
        )
        baseline_correct = baseline_prediction == true_answer
        candidate_correct = candidate_prediction == true_answer
        update_counter(overall, baseline_correct, candidate_correct)
        update_counter(by_prompt[str(baseline["student_prompt"])], baseline_correct, candidate_correct)
        update_counter(by_output[true_answer], baseline_correct, candidate_correct)
        if baseline_correct != candidate_correct:
            mismatches.append(
                {
                    "dataset_index": index,
                    "student_prompt": baseline["student_prompt"],
                    "answer": true_answer,
                    "baseline_prediction": baseline_prediction,
                    "candidate_prediction": candidate_prediction,
                    "baseline_correct": baseline_correct,
                    "candidate_correct": candidate_correct,
                }
            )

    return {
        "overall": summarize(overall),
        "by_prompt": {
            key: summarize(counter)
            for key, counter in sorted(by_prompt.items(), key=lambda item: (-item[1]["total"], item[0]))
        },
        "by_output": {
            str(key): summarize(counter)
            for key, counter in sorted(by_output.items(), key=lambda item: str(item[0]))
        },
        "mismatches": mismatches,
        "common_indices": common_indices,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_delta_bar(
    rows: list[dict[str, Any]],
    label_key: str,
    title: str,
    output: Path,
    min_count: int,
    horizontal: bool,
) -> None:
    filtered = [row for row in rows if int(row["count"]) >= min_count]
    if not filtered:
        return
    if horizontal:
        filtered = sorted(filtered, key=lambda row: float(row["accuracy_delta"]))
        fig_height = max(8, len(filtered) * 0.22)
        fig, ax = plt.subplots(figsize=(12, fig_height))
        y = np.arange(len(filtered))
        deltas = [float(row["accuracy_delta"]) for row in filtered]
        colors = ["#54a24b" if delta >= 0 else "#e45756" for delta in deltas]
        ax.barh(y, deltas, color=colors)
        ax.set_yticks(y, labels=[str(row[label_key]) for row in filtered], fontsize=6)
        ax.set_xlabel("Candidate accuracy minus baseline accuracy")
        ax.axvline(0, color="black", linewidth=1)
        fig.subplots_adjust(left=0.24, right=0.96, top=0.95, bottom=0.05)
        for index, row in enumerate(filtered):
            delta = float(row["accuracy_delta"])
            ax.text(delta, index, f" {delta:+.3f} n={row['count']}", va="center", fontsize=5)
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(filtered))
        deltas = [float(row["accuracy_delta"]) for row in filtered]
        colors = ["#54a24b" if delta >= 0 else "#e45756" for delta in deltas]
        ax.bar(x, deltas, color=colors)
        ax.set_xticks(x, labels=[str(row[label_key]) for row in filtered])
        ax.set_ylabel("Candidate accuracy minus baseline accuracy")
        ax.axhline(0, color="black", linewidth=1)
        fig.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.14)
        for index, row in enumerate(filtered):
            delta = float(row["accuracy_delta"])
            ax.text(index, delta, f"{delta:+.3f}\nn={row['count']}", ha="center", fontsize=6)
    ax.set_title(title)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_accuracy_pair_bar(
    rows: list[dict[str, Any]],
    label_key: str,
    baseline_name: str,
    candidate_name: str,
    title: str,
    output: Path,
    min_count: int,
    horizontal: bool,
) -> None:
    filtered = [row for row in rows if int(row["count"]) >= min_count]
    if not filtered:
        return
    filtered = sorted(
        filtered,
        key=lambda row: float(row["candidate_accuracy"]) - float(row["baseline_accuracy"]),
    )
    if horizontal:
        fig_height = max(8, len(filtered) * 0.25)
        fig, ax = plt.subplots(figsize=(13, fig_height))
        y = np.arange(len(filtered))
        bar_height = 0.38
        ax.barh(
            y - bar_height / 2,
            [float(row["baseline_accuracy"]) for row in filtered],
            height=bar_height,
            color="#4c78a8",
            label=baseline_name,
        )
        ax.barh(
            y + bar_height / 2,
            [float(row["candidate_accuracy"]) for row in filtered],
            height=bar_height,
            color="#f58518",
            label=candidate_name,
        )
        ax.set_yticks(y, labels=[str(row[label_key]) for row in filtered], fontsize=6)
        ax.set_xlabel("Accuracy")
        ax.set_xlim(0, 1)
        fig.subplots_adjust(left=0.24, right=0.96, top=0.95, bottom=0.05)
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(filtered))
        bar_width = 0.38
        ax.bar(
            x - bar_width / 2,
            [float(row["baseline_accuracy"]) for row in filtered],
            width=bar_width,
            color="#4c78a8",
            label=baseline_name,
        )
        ax.bar(
            x + bar_width / 2,
            [float(row["candidate_accuracy"]) for row in filtered],
            width=bar_width,
            color="#f58518",
            label=candidate_name,
        )
        ax.set_xticks(x, labels=[str(row[label_key]) for row in filtered])
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        fig.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.14)
    ax.set_title(title)
    ax.legend(loc="lower right")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def table_rows(grouped: dict[str, dict[str, Any]], label_key: str) -> list[dict[str, Any]]:
    rows = []
    for label, summary in grouped.items():
        rows.append({label_key: label, **summary})
    return rows


def main() -> None:
    args = parse_args()
    candidate, candidate_total_records = read_cache(args.candidate_cache)
    baseline, baseline_total_records = read_cache(args.baseline_cache, keep_indices=set(candidate))
    comparison = compare(baseline, candidate, args.collapse_at)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prompt_rows = table_rows(comparison["by_prompt"], "student_prompt")
    output_rows = table_rows(comparison["by_output"], "answer")
    write_csv(args.output_dir / "teacher_cache_comparison_by_prompt.csv", prompt_rows)
    write_csv(args.output_dir / "teacher_cache_comparison_by_output.csv", output_rows)
    write_csv(
        args.output_dir / "teacher_cache_comparison_mismatches.csv",
        comparison["mismatches"],
    )

    plot_delta_bar(
        prompt_rows,
        label_key="student_prompt",
        title=f"{args.candidate_name} vs {args.baseline_name}: Accuracy Delta by Prompt",
        output=args.output_dir / "teacher_cache_comparison_by_prompt_delta.png",
        min_count=args.min_group_count,
        horizontal=True,
    )
    plot_delta_bar(
        output_rows,
        label_key="answer",
        title=f"{args.candidate_name} vs {args.baseline_name}: Accuracy Delta by Output",
        output=args.output_dir / "teacher_cache_comparison_by_output_delta.png",
        min_count=args.min_group_count,
        horizontal=False,
    )
    plot_accuracy_pair_bar(
        prompt_rows,
        label_key="student_prompt",
        baseline_name=args.baseline_name,
        candidate_name=args.candidate_name,
        title=f"{args.candidate_name} vs {args.baseline_name}: Accuracy by Prompt",
        output=args.output_dir / "teacher_cache_comparison_by_prompt_side_by_side.png",
        min_count=args.min_group_count,
        horizontal=True,
    )
    plot_accuracy_pair_bar(
        output_rows,
        label_key="answer",
        baseline_name=args.baseline_name,
        candidate_name=args.candidate_name,
        title=f"{args.candidate_name} vs {args.baseline_name}: Accuracy by Output",
        output=args.output_dir / "teacher_cache_comparison_by_output_side_by_side.png",
        min_count=args.min_group_count,
        horizontal=False,
    )

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_cache": str(args.baseline_cache),
        "candidate_cache": str(args.candidate_cache),
        "baseline_name": args.baseline_name,
        "candidate_name": args.candidate_name,
        "baseline_records": baseline_total_records,
        "candidate_records": candidate_total_records,
        "baseline_records_loaded": len(baseline),
        "candidate_records_loaded": len(candidate),
        "intersecting_records": len(comparison["common_indices"]),
        "collapse_at": args.collapse_at,
        "min_group_count": args.min_group_count,
        "overall": comparison["overall"],
        "figures": {
            "by_prompt_delta": str(args.output_dir / "teacher_cache_comparison_by_prompt_delta.png"),
            "by_output_delta": str(args.output_dir / "teacher_cache_comparison_by_output_delta.png"),
            "by_prompt_side_by_side": str(
                args.output_dir / "teacher_cache_comparison_by_prompt_side_by_side.png"
            ),
            "by_output_side_by_side": str(
                args.output_dir / "teacher_cache_comparison_by_output_side_by_side.png"
            ),
        },
        "tables": {
            "by_prompt": str(args.output_dir / "teacher_cache_comparison_by_prompt.csv"),
            "by_output": str(args.output_dir / "teacher_cache_comparison_by_output.csv"),
            "mismatches": str(args.output_dir / "teacher_cache_comparison_mismatches.csv"),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
