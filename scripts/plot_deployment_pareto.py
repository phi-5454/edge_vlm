#!/usr/bin/env python3
"""Plot deployment Pareto views from a normalized model metrics table."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


REQUIRED_COLUMNS = {
    "model_id",
    "display_name",
    "target",
    "family",
    "accuracy",
    "accuracy_metric",
}

NUMERIC_COLUMNS = {
    "accuracy",
    "latency_ms",
    "model_size_bytes",
    "runtime_ram_bytes",
    "energy_mj",
    "power_mw",
}

PANEL_SPECS = {
    "A": {
        "x": "latency_ms",
        "y": "accuracy",
        "bubble": "model_size_bytes",
        "title": "A. Latency vs accuracy",
        "xlabel": "Latency per inference (ms)",
        "ylabel": "Accuracy",
        "bubble_label": "model size",
        "filename": "a_latency_accuracy_model_size.png",
        "xscale": "log",
    },
    "B": {
        "x": "runtime_ram_bytes",
        "y": "accuracy",
        "bubble": "model_size_bytes",
        "title": "B. Runtime RAM vs accuracy",
        "xlabel": "Runtime RAM / tensor arena (bytes)",
        "ylabel": "Accuracy",
        "bubble_label": "model size",
        "filename": "b_ram_accuracy_model_size.png",
        "xscale": "log",
    },
    "C": {
        "x": "energy_mj",
        "y": "accuracy",
        "bubble": "latency_ms",
        "title": "C. Energy vs accuracy",
        "xlabel": "Energy per inference (mJ)",
        "ylabel": "Accuracy",
        "bubble_label": "latency",
        "filename": "c_energy_accuracy_latency.png",
        "xscale": "log",
    },
}

TARGET_MARKERS = {
    "pytorch": "o",
    "keras": "o",
    "tflite": "s",
    "coral": "D",
    "max78000": "^",
    "teacher": "P",
    "baseline": "X",
}

FAMILY_COLORS = {
    "student": "#2f6fdd",
    "vlm": "#805ad5",
    "detector": "#2b8a3e",
    "coral": "#d9480f",
    "max78000": "#0b7285",
    "baseline": "#666666",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create deployment Pareto plots from a CSV with accuracy, latency, size, "
            "memory, and optional energy metrics."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/reports/deployment_pareto/model_metrics.csv"),
        help="Normalized model metrics CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/reports/deployment_pareto"),
        help="Directory for figures, copied normalized table, and manifest.",
    )
    parser.add_argument(
        "--accuracy-column",
        default="accuracy",
        help="Accuracy column to plot on the y-axis.",
    )
    parser.add_argument(
        "--label-top-k",
        type=int,
        default=30,
        help="Annotate at most this many points per panel.",
    )
    parser.add_argument(
        "--no-log-x",
        action="store_true",
        help="Use linear x axes even for latency/memory/energy.",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Also write a single three-panel figure.",
    )
    return parser.parse_args()


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    stripped = str(value).strip()
    if not stripped:
        return None
    try:
        parsed = float(stripped)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header.")
        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        rows: list[dict[str, Any]] = []
        for row in reader:
            parsed = dict(row)
            for column in NUMERIC_COLUMNS:
                parsed[column] = parse_float(row.get(column))
            rows.append(parsed)
    if not rows:
        raise ValueError(f"{path} contains no model rows.")
    return rows


def write_normalized_csv(rows: list[dict[str, Any]], output: Path) -> None:
    columns = [
        "model_id",
        "display_name",
        "target",
        "family",
        "variant",
        "accuracy",
        "accuracy_metric",
        "latency_ms",
        "model_size_bytes",
        "runtime_ram_bytes",
        "energy_mj",
        "power_mw",
        "latency_source",
        "size_source",
        "memory_source",
        "energy_source",
        "notes",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def bubble_sizes(values: list[float | None]) -> list[float]:
    finite = [value for value in values if value is not None and value > 0]
    if not finite:
        return [90.0 for _ in values]
    min_value = min(finite)
    max_value = max(finite)
    if max_value <= min_value:
        return [180.0 if value is not None else 90.0 for value in values]
    sizes = []
    for value in values:
        if value is None or value <= 0:
            sizes.append(70.0)
            continue
        progress = (math.log(value) - math.log(min_value)) / (
            math.log(max_value) - math.log(min_value)
        )
        sizes.append(80.0 + 520.0 * progress)
    return sizes


def rows_for_panel(
    rows: list[dict[str, Any]],
    spec: dict[str, str],
    accuracy_column: str,
) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        x_value = row.get(spec["x"])
        y_value = row.get(accuracy_column)
        if isinstance(x_value, (int, float)) and isinstance(y_value, (int, float)):
            if math.isfinite(float(x_value)) and math.isfinite(float(y_value)):
                if spec.get("xscale") == "log" and float(x_value) <= 0:
                    continue
                selected.append(row)
    return selected


def plot_panel(
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    spec: dict[str, str],
    accuracy_column: str,
    label_top_k: int,
    log_x: bool,
) -> dict[str, Any]:
    selected = rows_for_panel(rows, spec, accuracy_column)
    if not selected:
        ax.text(0.5, 0.5, "No complete rows", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(spec["title"])
        return {"status": "skipped", "reason": "no complete rows"}

    bubble_values = [row.get(spec["bubble"]) for row in selected]
    sizes = bubble_sizes([value if isinstance(value, float) else None for value in bubble_values])
    seen_labels: set[tuple[str, str]] = set()
    for row, size in zip(selected, sizes, strict=True):
        family = str(row.get("family") or "unknown")
        target = str(row.get("target") or "unknown")
        label = (family, target)
        legend_label = f"{family} / {target}" if label not in seen_labels else None
        seen_labels.add(label)
        ax.scatter(
            float(row[spec["x"]]),
            float(row[accuracy_column]),
            s=size,
            alpha=0.72,
            edgecolor="black",
            linewidth=0.6,
            marker=TARGET_MARKERS.get(target, "o"),
            color=FAMILY_COLORS.get(family, "#4c78a8"),
            label=legend_label,
        )

    annotate_rows = sorted(
        selected,
        key=lambda row: float(row.get(accuracy_column) or 0),
        reverse=True,
    )[: max(0, label_top_k)]
    for row in annotate_rows:
        ax.annotate(
            str(row.get("display_name") or row.get("model_id")),
            (float(row[spec["x"]]), float(row[accuracy_column])),
            xytext=(5, 3),
            textcoords="offset points",
            fontsize=8,
        )

    if log_x and spec.get("xscale") == "log":
        ax.set_xscale("log")
    ax.set_title(spec["title"])
    ax.set_xlabel(spec["xlabel"])
    ax.set_ylabel(spec["ylabel"])
    ax.grid(True, which="both", alpha=0.25)
    ax.set_ylim(0, max(1.0, min(1.05, max(float(row[accuracy_column]) for row in selected) * 1.08)))
    ax.legend(loc="best", fontsize=8)
    return {
        "status": "written",
        "rows": len(selected),
        "x": spec["x"],
        "y": accuracy_column,
        "bubble": spec["bubble"],
    }


def plot_single_panels(
    rows: list[dict[str, Any]],
    output_dir: Path,
    accuracy_column: str,
    label_top_k: int,
    log_x: bool,
) -> dict[str, Any]:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    reports: dict[str, Any] = {}
    for panel, spec in PANEL_SPECS.items():
        fig, ax = plt.subplots(figsize=(8.2, 5.6))
        report = plot_panel(ax, rows, spec, accuracy_column, label_top_k, log_x)
        fig.tight_layout()
        output = figures / spec["filename"]
        fig.savefig(output, dpi=180)
        plt.close(fig)
        report["path"] = str(output)
        reports[panel] = report
    return reports


def plot_combined(
    rows: list[dict[str, Any]],
    output_dir: Path,
    accuracy_column: str,
    label_top_k: int,
    log_x: bool,
) -> dict[str, Any]:
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.6))
    reports: dict[str, Any] = {}
    for ax, (panel, spec) in zip(axes, PANEL_SPECS.items(), strict=True):
        reports[panel] = plot_panel(
            ax,
            rows,
            spec,
            accuracy_column,
            label_top_k,
            log_x,
        )
    fig.tight_layout()
    output = output_dir / "figures" / "abc_deployment_pareto.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return {"path": str(output), "panels": reports}


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    if args.accuracy_column != "accuracy":
        for row in rows:
            row["accuracy"] = parse_float(row.get(args.accuracy_column))
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_csv = output_dir / "model_metrics.normalized.csv"
    write_normalized_csv(rows, normalized_csv)
    panel_reports = plot_single_panels(
        rows,
        output_dir,
        "accuracy",
        args.label_top_k,
        not args.no_log_x,
    )
    combined_report = (
        plot_combined(rows, output_dir, "accuracy", args.label_top_k, not args.no_log_x)
        if args.combined
        else None
    )
    manifest = {
        "input": str(args.input),
        "normalized_csv": str(normalized_csv),
        "output_dir": str(output_dir),
        "rows": len(rows),
        "columns_required": sorted(REQUIRED_COLUMNS),
        "panels": panel_reports,
        "combined": combined_report,
        "metric_notes": {
            "accuracy": "Use one consistent accuracy metric across rows, preferably prompt_class_output_weighted_accuracy.",
            "latency_ms": "Per-inference model latency, not host data loading.",
            "model_size_bytes": "Deployed model/checkpoint artifact byte size.",
            "runtime_ram_bytes": "Peak runtime RAM/tensor arena/activation memory when available.",
            "energy_mj": "Measured or estimated energy per inference; leave blank until measured.",
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
