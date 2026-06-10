from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vlm_micro.student.data import collapse_count, load_tallyqa_rows


def load_prompt_names(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    prompts = {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if not prompts:
        raise ValueError(f"{path} does not contain any prompt names.")
    return prompts


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prompt_stats(
    rows: list[dict[str, Any]],
    prompt_names: set[str] | None,
    collapse_at: int,
) -> list[dict[str, Any]]:
    answers_by_prompt: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        prompt = str(row["student_prompt"])
        if prompt_names is not None and prompt not in prompt_names:
            continue
        answers_by_prompt[prompt].append(int(row["answer"]))

    stats: list[dict[str, Any]] = []
    normal = NormalDist()
    for prompt, answers in sorted(answers_by_prompt.items()):
        raw = np.asarray(answers, dtype=float)
        collapsed = np.asarray([collapse_count(answer, collapse_at) for answer in answers], dtype=float)
        std = float(raw.std(ddof=1)) if len(raw) > 1 else 0.0
        collapsed_std = float(collapsed.std(ddof=1)) if len(collapsed) > 1 else 0.0
        se = std / math.sqrt(len(raw)) if raw.size else 0.0
        collapsed_se = collapsed_std / math.sqrt(len(collapsed)) if collapsed.size else 0.0
        stats.append(
            {
                "student_prompt": prompt,
                "n": int(len(raw)),
                "mean_answer": float(raw.mean()),
                "std_answer": std,
                "se_answer": se,
                "ci95_low_answer": float(raw.mean() - normal.inv_cdf(0.975) * se),
                "ci95_high_answer": float(raw.mean() + normal.inv_cdf(0.975) * se),
                "median_answer": float(np.median(raw)),
                "mean_collapsed": float(collapsed.mean()),
                "std_collapsed": collapsed_std,
                "se_collapsed": collapsed_se,
                "ci95_low_collapsed": float(collapsed.mean() - normal.inv_cdf(0.975) * collapsed_se),
                "ci95_high_collapsed": float(collapsed.mean() + normal.inv_cdf(0.975) * collapsed_se),
                "median_collapsed": float(np.median(collapsed)),
                "min_answer": int(raw.min()),
                "max_answer": int(raw.max()),
            }
        )
    return stats


def plot_histogram(rows: list[dict[str, Any]], key: str, title: str, output: Path) -> None:
    values = [float(row[key]) for row in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=min(24, max(6, len(values) // 3)), color="#4c78a8", edgecolor="white")
    ax.axvline(np.mean(values), color="#f58518", linewidth=2, label=f"mean={np.mean(values):.2f}")
    ax.axvline(np.median(values), color="#54a24b", linewidth=2, label=f"median={np.median(values):.2f}")
    ax.set_title(title)
    ax.set_xlabel(key)
    ax.set_ylabel("Prompt classes")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_sorted_means(rows: list[dict[str, Any]], key: str, ci_low: str, ci_high: str, title: str, output: Path) -> None:
    ordered = sorted(rows, key=lambda row: float(row[key]))
    x = np.arange(len(ordered))
    means = np.asarray([float(row[key]) for row in ordered])
    low = np.asarray([float(row[ci_low]) for row in ordered])
    high = np.asarray([float(row[ci_high]) for row in ordered])
    fig, ax = plt.subplots(figsize=(max(9, len(ordered) * 0.16), 5.5))
    ax.errorbar(
        x,
        means,
        yerr=[means - low, high - means],
        fmt="o",
        markersize=3,
        linewidth=0.8,
        color="#4c78a8",
        ecolor="#9ecae9",
    )
    ax.set_title(title)
    ax.set_xlabel("Prompt classes sorted by mean")
    ax.set_ylabel(key)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_mean_vs_frequency(rows: list[dict[str, Any]], key: str, title: str, output: Path) -> None:
    counts = np.asarray([int(row["n"]) for row in rows])
    means = np.asarray([float(row[key]) for row in rows])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(counts, means, s=24, alpha=0.75, color="#4c78a8")
    ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel("Prompt class examples (log)")
    ax.set_ylabel(key)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--prompt-class-names-file", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--collapse-at", type=int, default=5)
    args = parser.parse_args()

    rows = load_tallyqa_rows(args.dataset)
    prompt_names = load_prompt_names(args.prompt_class_names_file)
    stats = prompt_stats(rows, prompt_names, args.collapse_at)
    if not stats:
        raise ValueError("No prompt classes matched the requested filters.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "prompt_class_mean_answers.csv", stats)
    plot_histogram(
        stats,
        "mean_answer",
        "Distribution of per-prompt mean true counts",
        args.output_dir / "figures/prompt_mean_answer_histogram.png",
    )
    plot_histogram(
        stats,
        "mean_collapsed",
        f"Distribution of per-prompt mean collapsed counts ({args.collapse_at}+)",
        args.output_dir / "figures/prompt_mean_collapsed_histogram.png",
    )
    plot_sorted_means(
        stats,
        "mean_answer",
        "ci95_low_answer",
        "ci95_high_answer",
        "Per-prompt mean true counts with 95% CI",
        args.output_dir / "figures/prompt_mean_answer_sorted_ci.png",
    )
    plot_sorted_means(
        stats,
        "mean_collapsed",
        "ci95_low_collapsed",
        "ci95_high_collapsed",
        f"Per-prompt mean collapsed counts ({args.collapse_at}+) with 95% CI",
        args.output_dir / "figures/prompt_mean_collapsed_sorted_ci.png",
    )
    plot_mean_vs_frequency(
        stats,
        "mean_answer",
        "Per-prompt mean true count vs frequency",
        args.output_dir / "figures/prompt_mean_answer_vs_frequency.png",
    )

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "prompt_class_names_file": str(args.prompt_class_names_file)
        if args.prompt_class_names_file is not None
        else None,
        "collapse_at": args.collapse_at,
        "records": sum(int(row["n"]) for row in stats),
        "prompt_classes": len(stats),
        "mean_of_prompt_means_answer": float(np.mean([float(row["mean_answer"]) for row in stats])),
        "std_of_prompt_means_answer": float(np.std([float(row["mean_answer"]) for row in stats], ddof=1))
        if len(stats) > 1
        else 0.0,
        "min_prompt_mean_answer": float(min(float(row["mean_answer"]) for row in stats)),
        "max_prompt_mean_answer": float(max(float(row["mean_answer"]) for row in stats)),
        "tables": {
            "prompt_class_mean_answers": str(args.output_dir / "prompt_class_mean_answers.csv"),
        },
        "figures": {
            "prompt_mean_answer_histogram": str(args.output_dir / "figures/prompt_mean_answer_histogram.png"),
            "prompt_mean_collapsed_histogram": str(
                args.output_dir / "figures/prompt_mean_collapsed_histogram.png"
            ),
            "prompt_mean_answer_sorted_ci": str(args.output_dir / "figures/prompt_mean_answer_sorted_ci.png"),
            "prompt_mean_collapsed_sorted_ci": str(
                args.output_dir / "figures/prompt_mean_collapsed_sorted_ci.png"
            ),
            "prompt_mean_answer_vs_frequency": str(args.output_dir / "figures/prompt_mean_answer_vs_frequency.png"),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
