#!/usr/bin/env python3
"""Run standard EDA for a Coral Micro TallyQA teacher cache."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DATASET = Path("data/tallyqa_cauldron_target_mobilenet224_letterbox")
DEFAULT_CLASSES = DEFAULT_DATASET / "classes.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--classes", type=Path, default=DEFAULT_CLASSES)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", default="Coral Micro TallyQA")
    parser.add_argument("--answer-min", type=int, default=0)
    parser.add_argument("--answer-max", type=int, default=5)
    parser.add_argument("--collapse-at", type=int, default=5)
    parser.add_argument("--example-count", type=int, default=12)
    parser.add_argument("--example-cols", type=int, default=3)
    parser.add_argument("--skip-examples", action="store_true")
    return parser.parse_args()


def run_command(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def load_manifest(cache: Path) -> dict[str, Any] | None:
    manifest_path = cache.with_suffix(".manifest.json")
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_latency_records(cache: Path) -> list[dict[str, float]]:
    records: list[dict[str, float]] = []
    with cache.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            timing = row.get("board_timing", {})
            if not timing:
                continue
            record = {"dataset_index": float(row.get("dataset_index", len(records)))}
            for key in ("invoke_us", "copy_us", "receive_us", "host_roundtrip_us"):
                if key in timing:
                    record[key] = float(timing[key]) / 1000.0
            records.append(record)
    return records


def plot_latency(manifest: dict[str, Any], output: Path) -> dict[str, Any]:
    latency = manifest.get("latency", {})
    rows = []
    for key in ("invoke_us", "copy_us", "receive_us", "host_roundtrip_us"):
        summary = latency.get(key)
        if not summary:
            continue
        rows.append(
            {
                "name": key,
                "mean_ms": float(summary["mean_us"]) / 1000.0,
                "median_ms": float(summary["median_us"]) / 1000.0,
                "p95_ms": float(summary["p95_us"]) / 1000.0,
            }
        )
    if not rows:
        return {"status": "skipped", "reason": "manifest has no latency block"}

    names = [row["name"] for row in rows]
    x = np.arange(len(rows))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, [row["mean_ms"] for row in rows], width, label="mean")
    ax.bar(x, [row["median_ms"] for row in rows], width, label="median")
    ax.bar(x + width, [row["p95_ms"] for row in rows], width, label="p95")
    ax.set_xticks(x, labels=names, rotation=20, ha="right")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Coral Micro benchmark latency")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return {"status": "written", "output": str(output), "rows": rows}


def write_latency_records_csv(records: list[dict[str, float]], output: Path) -> dict[str, Any]:
    if not records:
        return {"status": "skipped", "reason": "cache has no per-record board_timing"}
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset_index", "invoke_us", "copy_us", "receive_us", "host_roundtrip_us"]
    # Values are stored in ms in records; keep legacy column names out of the CSV header.
    fieldnames_ms = ["dataset_index", "invoke_ms", "copy_ms", "receive_ms", "host_roundtrip_ms"]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames_ms)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "dataset_index": int(record["dataset_index"]),
                    "invoke_ms": record.get("invoke_us"),
                    "copy_ms": record.get("copy_us"),
                    "receive_ms": record.get("receive_us"),
                    "host_roundtrip_ms": record.get("host_roundtrip_us"),
                }
            )
    return {"status": "written", "output": str(output), "rows": len(records)}


def plot_latency_histograms(records: list[dict[str, float]], output: Path) -> dict[str, Any]:
    if not records:
        return {"status": "skipped", "reason": "cache has no per-record board_timing"}
    keys = [
        ("invoke_us", "Invoke"),
        ("copy_us", "Input copy"),
        ("receive_us", "Serial receive"),
        ("host_roundtrip_us", "Host roundtrip"),
    ]
    available = [(key, label) for key, label in keys if any(key in record for record in records)]
    if not available:
        return {"status": "skipped", "reason": "no latency fields found"}

    cols = 2
    rows = int(np.ceil(len(available) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10, max(3.4, 3.2 * rows)))
    axes_array = np.atleast_1d(axes).reshape(rows, cols)
    summary_rows = []
    for ax, (key, label) in zip(axes_array.ravel(), available, strict=False):
        values = np.asarray([record[key] for record in records if key in record], dtype=float)
        bins = min(30, max(1, int(np.sqrt(values.size))))
        ax.hist(values, bins=bins, color="#4c78a8", edgecolor="white", alpha=0.9)
        ax.axvline(float(np.mean(values)), color="#f58518", linestyle="-", linewidth=1.5, label="mean")
        ax.axvline(float(np.median(values)), color="#54a24b", linestyle="--", linewidth=1.5, label="median")
        ax.set_title(f"{label} latency")
        ax.set_xlabel("ms")
        ax.set_ylabel("examples")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
        summary_rows.append(
            {
                "name": key,
                "count": int(values.size),
                "mean_ms": float(np.mean(values)),
                "median_ms": float(np.median(values)),
                "p95_ms": float(np.percentile(values, 95)),
                "min_ms": float(np.min(values)),
                "max_ms": float(np.max(values)),
            }
        )
    for ax in axes_array.ravel()[len(available) :]:
        ax.axis("off")
    fig.suptitle("Coral Micro per-example latency histograms")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return {"status": "written", "output": str(output), "rows": summary_rows}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = args.output_dir / "figures"
    tables = args.output_dir / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    accuracy_dir = args.output_dir / "teacher_accuracy"
    confusion_dir = args.output_dir / "teacher_confusion"
    run_command(
        [
            sys.executable,
            "scripts/plot_tallyqa_teacher_accuracy.py",
            "--cache",
            str(args.cache),
            "--classes",
            str(args.classes),
            "--output-dir",
            str(accuracy_dir),
            "--answer-min",
            str(args.answer_min),
            "--answer-max",
            str(args.answer_max),
            "--collapse-at",
            str(args.collapse_at),
        ]
    )
    run_command(
        [
            sys.executable,
            "scripts/plot_tallyqa_teacher_confusion.py",
            "--cache",
            str(args.cache),
            "--output-dir",
            str(confusion_dir),
            "--answer-min",
            str(args.answer_min),
            "--answer-max",
            str(args.answer_max),
            "--collapse-at",
            str(args.collapse_at),
            "--title",
            f"{args.title} output confusion",
        ]
    )

    examples_report: dict[str, Any] = {"status": "skipped", "reason": "--skip-examples"}
    if not args.skip_examples:
        example_output = figures / "example_predictions.png"
        run_command(
            [
                sys.executable,
                "scripts/visualize_tallyqa_teacher_logits.py",
                "--dataset",
                str(args.dataset),
                "--cache",
                str(args.cache),
                "--output",
                str(example_output),
                "--count",
                str(args.example_count),
                "--cols",
                str(args.example_cols),
                "--answer-min",
                str(args.answer_min),
                "--answer-max",
                str(args.answer_max),
                "--collapse-at",
                str(args.collapse_at),
            ]
        )
        examples_report = {"status": "written", "output": str(example_output)}

    manifest = load_manifest(args.cache)
    latency_report = (
        plot_latency(manifest, figures / "latency_summary.png")
        if manifest is not None
        else {"status": "skipped", "reason": "cache manifest not found"}
    )
    latency_records = load_latency_records(args.cache)
    latency_records_report = write_latency_records_csv(latency_records, tables / "latency_records.csv")
    latency_histogram_report = plot_latency_histograms(
        latency_records,
        figures / "latency_histograms.png",
    )
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cache": str(args.cache),
        "dataset": str(args.dataset),
        "output_dir": str(args.output_dir),
        "artifacts": {
            "accuracy_dir": str(accuracy_dir),
            "confusion_dir": str(confusion_dir),
            "examples": examples_report,
            "latency": latency_report,
            "latency_records": latency_records_report,
            "latency_histograms": latency_histogram_report,
        },
    }
    (args.output_dir / "coral_cache_eda_summary.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
