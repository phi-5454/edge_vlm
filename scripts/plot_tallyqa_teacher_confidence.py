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


DEFAULT_CACHE = Path("artifacts/teacher_cache/smolvlm_tallyqa_target_mobilenet224.jsonl")
DEFAULT_OUTPUT_DIR = Path("artifacts/reports/tallyqa_teacher_confidence")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot teacher confidence versus empirical accuracy for TallyQA caches."
    )
    parser.add_argument(
        "--cache",
        action="append",
        type=Path,
        default=None,
        help="Teacher cache JSONL. Repeat to overlay multiple caches.",
    )
    parser.add_argument(
        "--name",
        action="append",
        default=None,
        help="Display name for each --cache. Must be repeated the same number of times.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--answer-min", type=int, default=0)
    parser.add_argument("--answer-max", type=int, default=15)
    parser.add_argument(
        "--collapse-at",
        type=int,
        default=5,
        help="Collapse true/predicted output classes >= this value into a '<n>+' class.",
    )
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--min-prompt-count", type=int, default=25)
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


def row_distribution(
    row: dict[str, Any],
    classes: list[int | str],
    collapse_at: int | None,
) -> dict[int | str, float]:
    probabilities = {label: 0.0 for label in classes}
    candidates = row.get("teacher_logits", {}).get("numeric_answer_candidates", [])
    for candidate in candidates:
        answer = int(candidate["answer"])
        label = output_class(answer, collapse_at)
        if label in probabilities:
            probabilities[label] += float(candidate.get("candidate_probability", 0.0))

    total = sum(probabilities.values())
    if total > 0:
        return {label: probability / total for label, probability in probabilities.items()}

    prediction = output_class(
        int(row["teacher_metrics"]["numeric_answer"]["prediction"]),
        collapse_at,
    )
    probabilities[prediction] = 1.0
    return probabilities


def bin_index(confidence: float, bins: int) -> int:
    return min(bins - 1, max(0, int(confidence * bins)))


def cache_display_name(path: Path, name: str | None) -> str:
    return name if name is not None else path.stem


def stream_cache(
    path: Path,
    name: str,
    classes: list[int | str],
    collapse_at: int | None,
    bins: int,
) -> dict[str, Any]:
    bin_stats: list[Counter] = [Counter() for _ in range(bins)]
    prompt_stats: dict[str, Counter] = defaultdict(Counter)
    total = 0

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            probabilities = row_distribution(row, classes, collapse_at)
            prediction = max(classes, key=lambda label: probabilities[label])
            confidence = float(probabilities[prediction])
            true_label = output_class(int(row["answer"]), collapse_at)
            correct = int(prediction == true_label)
            index = bin_index(confidence, bins)

            bin_stats[index]["total"] += 1
            bin_stats[index]["correct"] += correct
            bin_stats[index]["confidence_sum"] += confidence

            prompt = str(row["student_prompt"])
            prompt_stats[prompt]["total"] += 1
            prompt_stats[prompt]["correct"] += correct
            prompt_stats[prompt]["confidence_sum"] += confidence
            total += 1

    bin_rows = []
    ece = 0.0
    for index, stats in enumerate(bin_stats):
        count = int(stats["total"])
        lower = index / bins
        upper = (index + 1) / bins
        accuracy = stats["correct"] / count if count else float("nan")
        mean_confidence = stats["confidence_sum"] / count if count else float("nan")
        if count:
            ece += (count / total) * abs(accuracy - mean_confidence)
        bin_rows.append(
            {
                "cache": name,
                "bin": index,
                "confidence_lower": lower,
                "confidence_upper": upper,
                "count": count,
                "accuracy": accuracy,
                "mean_confidence": mean_confidence,
            }
        )

    prompt_rows = []
    for prompt, stats in sorted(prompt_stats.items()):
        count = int(stats["total"])
        accuracy = stats["correct"] / count if count else float("nan")
        mean_confidence = stats["confidence_sum"] / count if count else float("nan")
        prompt_rows.append(
            {
                "cache": name,
                "student_prompt": prompt,
                "count": count,
                "accuracy": accuracy,
                "mean_confidence": mean_confidence,
                "confidence_minus_accuracy": mean_confidence - accuracy,
            }
        )

    overall_correct = sum(int(stats["correct"]) for stats in bin_stats)
    overall_confidence = sum(float(stats["confidence_sum"]) for stats in bin_stats)
    return {
        "name": name,
        "path": str(path),
        "total": total,
        "accuracy": overall_correct / total if total else float("nan"),
        "mean_confidence": overall_confidence / total if total else float("nan"),
        "ece": ece,
        "bin_rows": bin_rows,
        "prompt_rows": prompt_rows,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_reliability(results: list[dict[str, Any]], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    ax.plot([0, 1], [0, 1], color="0.35", linestyle="--", linewidth=1.1, label="calibrated")
    for result in results:
        rows = [row for row in result["bin_rows"] if row["count"] > 0]
        xs = [float(row["mean_confidence"]) for row in rows]
        ys = [float(row["accuracy"]) for row in rows]
        sizes = [max(24.0, np.sqrt(float(row["count"])) * 2.0) for row in rows]
        ax.scatter(xs, ys, s=sizes, alpha=0.75)
        ax.plot(xs, ys, linewidth=1.4, label=f"{result['name']} (ECE {result['ece']:.3f})")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Teacher Confidence vs Accuracy")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_confidence_histogram(results: list[dict[str, Any]], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    width = 0.8 / max(1, len(results))
    for result_index, result in enumerate(results):
        rows = result["bin_rows"]
        centers = np.array(
            [
                (float(row["confidence_lower"]) + float(row["confidence_upper"])) / 2.0
                for row in rows
            ]
        )
        counts = np.array([int(row["count"]) for row in rows], dtype=float)
        offset = (result_index - (len(results) - 1) / 2.0) * width
        ax.bar(centers + offset, counts, width=width, alpha=0.75, label=result["name"])
    ax.set_xlabel("Predicted confidence bin")
    ax.set_ylabel("Records")
    ax.set_title("Teacher Confidence Histogram")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    caches = args.cache or [DEFAULT_CACHE]
    names = args.name or [None] * len(caches)
    if len(names) != len(caches):
        raise ValueError("--name must be repeated the same number of times as --cache.")
    if args.bins <= 0:
        raise ValueError("--bins must be positive.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    classes = output_classes(args.answer_min, args.answer_max, args.collapse_at)
    results = [
        stream_cache(
            path=cache,
            name=cache_display_name(cache, name),
            classes=classes,
            collapse_at=args.collapse_at,
            bins=args.bins,
        )
        for cache, name in zip(caches, names, strict=True)
    ]

    bin_rows = [row for result in results for row in result["bin_rows"]]
    prompt_rows = [
        row
        for result in results
        for row in result["prompt_rows"]
        if int(row["count"]) >= args.min_prompt_count
    ]
    write_csv(args.output_dir / "teacher_confidence_bins.csv", bin_rows)
    write_csv(args.output_dir / "teacher_confidence_by_prompt_class.csv", prompt_rows)
    plot_reliability(results, args.output_dir / "teacher_confidence_reliability.png")
    plot_confidence_histogram(results, args.output_dir / "teacher_confidence_histogram.png")

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "caches": [{"name": result["name"], "path": result["path"]} for result in results],
        "answer_min": args.answer_min,
        "answer_max": args.answer_max,
        "collapse_at": args.collapse_at,
        "bins": args.bins,
        "min_prompt_count": args.min_prompt_count,
        "overall": [
            {
                "name": result["name"],
                "records": result["total"],
                "accuracy": result["accuracy"],
                "mean_confidence": result["mean_confidence"],
                "ece": result["ece"],
            }
            for result in results
        ],
        "outputs": {
            "reliability": str(args.output_dir / "teacher_confidence_reliability.png"),
            "histogram": str(args.output_dir / "teacher_confidence_histogram.png"),
            "bins": str(args.output_dir / "teacher_confidence_bins.csv"),
            "by_prompt": str(args.output_dir / "teacher_confidence_by_prompt_class.csv"),
        },
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
