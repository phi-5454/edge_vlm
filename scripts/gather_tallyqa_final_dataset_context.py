from __future__ import annotations

import argparse
from datetime import datetime, timezone
import csv
import json
import os
from pathlib import Path
import shutil
from typing import Any

os.environ["MPLBACKEND"] = "Agg"

import matplotlib.pyplot as plt


DEFAULT_SOURCE_DIR = Path("artifacts/reports/tallyqa_cauldron_eda")
DEFAULT_TARGET_DIR = Path("artifacts/reports/final_dataset/teacher_eda_full/context")
DEFAULT_CLASSES = Path("data/tallyqa_cauldron_target_mobilenet224/classes.txt")


COPY_TARGETS = [
    (
        "figures/output_class_histogram_raw_answer_distribution.png",
        "figures/combined/answer_distribution.png",
        "Raw answer/output-class histogram from Cauldron TallyQA EDA.",
    ),
    (
        "figures/suffix_trie_top_branches.png",
        "figures/combined/suffix_trie_top_branches.png",
        "Suffix trie used to identify reusable prompt suffixes for item extraction.",
    ),
    (
        "figures/prefix_2_word_prevalence.png",
        "figures/combined/prefix_2_word_prevalence.png",
        "Two-word prefix prevalence showing that nearly all prompts begin with 'how many'.",
    ),
    (
        "figures/prefix_3_word_prevalence.png",
        "figures/combined/prefix_3_word_prevalence.png",
        "Three-word prefix prevalence for prompt regularity context.",
    ),
    (
        "figures/prefix_4_word_prevalence.png",
        "figures/combined/prefix_4_word_prevalence.png",
        "Four-word prefix prevalence for prompt regularity context.",
    ),
    (
        "figures/question_coverage.png",
        "figures/combined/question_coverage.png",
        "Question coverage plot from the original TallyQA prompt EDA.",
    ),
    (
        "figures/template_items_rank_001_100.png",
        "figures/combined/template_items_rank_001_100.png",
        "Top extracted prompt item classes, ranks 1-100.",
    ),
    (
        "figures/template_items_rank_101_200.png",
        "figures/combined/template_items_rank_101_200.png",
        "Top extracted prompt item classes, ranks 101-200.",
    ),
    (
        "lists/template_items_top200_combined.txt",
        "template_items_top200_combined.txt",
        "Top 200 extracted prompt item classes before manual pruning.",
    ),
    (
        "lists/template_items_top200_train.txt",
        "template_items_top200_train.txt",
        "Top 200 train split prompt item classes before manual pruning.",
    ),
    (
        "lists/template_items_top200_pruned.txt",
        "template_items_top200_pruned.txt",
        "Manually pruned top-200 item classes used to define the target prompt-class set.",
    ),
    (
        "lists/frontier_suffixes.txt",
        "frontier_suffixes.txt",
        "Automatically discovered suffix-frontier candidates before manual pruning.",
    ),
    (
        "lists/frontier_suffixes_pruned.txt",
        "frontier_suffixes_pruned.txt",
        "Manually pruned suffix-frontier list used for item extraction.",
    ),
    (
        "lists/filter_suffixes_used.txt",
        "filter_suffixes_used.txt",
        "Final suffixes used by the prompt item extraction filter.",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gather final-dataset TallyQA prompt-pruning context artifacts."
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET_DIR)
    parser.add_argument("--classes", type=Path, default=DEFAULT_CLASSES)
    return parser.parse_args()


def copy_artifacts(source_dir: Path, target_dir: Path) -> list[dict[str, Any]]:
    copied = []
    for relative_target, relative_source, description in COPY_TARGETS:
        source = source_dir / relative_source
        target = target_dir / relative_target
        target.parent.mkdir(parents=True, exist_ok=True)
        if not source.exists():
            copied.append(
                {
                    "target": str(target),
                    "source": str(source),
                    "description": description,
                    "status": "missing_source",
                }
            )
            continue
        shutil.copy2(source, target)
        copied.append(
            {
                "target": str(target),
                "source": str(source),
                "description": description,
                "status": "copied",
            }
        )
    return copied


def line_count(path: Path) -> int | None:
    if not path.exists():
        return None
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_retention(rows: list[dict[str, Any]], output: Path) -> None:
    rows = [row for row in rows if row["prompt_classes"] is not None]
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    labels = [str(row["stage"]) for row in rows]
    values = [int(row["prompt_classes"]) for row in rows]
    colors = ["#4c78a8", "#72b7b2", "#f58518", "#54a24b"]
    ax.bar(labels, values, color=colors[: len(rows)])
    ax.set_ylabel("Prompt item classes")
    ax.set_title("Prompt-Class Retention Through Pruning")
    ax.tick_params(axis="x", labelrotation=18)
    for index, row in enumerate(rows):
        ax.text(index, values[index] + max(values) * 0.015, str(values[index]), ha="center")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def retention_rows(source_dir: Path, classes: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary = load_summary(source_dir / "summary.json")
    template_items = summary["splits"]["train"]["template_items"]
    top_200_count = line_count(source_dir / "template_items_top200_combined.txt")
    pruned_count = line_count(source_dir / "template_items_top200_pruned.txt")
    materialized_count = line_count(classes)
    rows = [
        {
            "stage": "all_extracted_items",
            "prompt_classes": int(template_items["unique_items"]),
            "description": "Unique 1-2 word item spans extracted after suffix filtering.",
        },
        {
            "stage": "top_200_pre_pruning",
            "prompt_classes": top_200_count,
            "description": "Top 200 extracted item classes by frequency before manual pruning.",
        },
        {
            "stage": "post_pruning",
            "prompt_classes": pruned_count,
            "description": (
                "Manually pruned prompt classes retained from the top-200 list."
                if pruned_count is not None
                else "Pending manual pruning of this split-specific top-200 list."
            ),
        },
        {
            "stage": "materialized_target_classes",
            "prompt_classes": materialized_count,
            "description": (
                "Prompt classes in the materialized final TallyQA target dataset."
                if materialized_count is not None
                else "Pending materialization after train/val-only pruning is locked."
            ),
        },
    ]
    details = {
        "source_summary": str(source_dir / "summary.json"),
        "dataset_rows": int(summary["splits"]["train"]["rows"]),
        "unique_questions": int(summary["splits"]["train"]["unique_questions"]),
        "template_item_candidate_questions": int(template_items["candidate_questions"]),
        "template_item_matched_questions": int(template_items["matched_questions"]),
        "template_item_matched_fraction": float(template_items["matched_fraction"]),
        "suffixes_pruned": line_count(source_dir / "frontier_suffixes_pruned.txt"),
        "target_classes": str(classes),
    }
    return rows, details


def main() -> None:
    args = parse_args()
    args.target_dir.mkdir(parents=True, exist_ok=True)
    copied = copy_artifacts(args.source_dir, args.target_dir)
    rows, details = retention_rows(args.source_dir, args.classes)
    write_csv(args.target_dir / "tables/prompt_class_retention.csv", rows)
    plot_retention(rows, args.target_dir / "figures/prompt_class_retention.png")
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(args.source_dir),
        "target_dir": str(args.target_dir),
        "reasoning": (
            "Gather context artifacts that informed the final TallyQA prompt-class dataset: "
            "answer distribution, prompt prefix regularity, suffix-trie pruning, top-200 item "
            "lists, pruned lists, and retention counts."
        ),
        "retention": {
            "rows": rows,
            **details,
            "csv": str(args.target_dir / "tables/prompt_class_retention.csv"),
            "figure": str(args.target_dir / "figures/prompt_class_retention.png"),
        },
        "copied_artifacts": copied,
    }
    (args.target_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
