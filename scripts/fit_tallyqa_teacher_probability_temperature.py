from __future__ import annotations

import argparse
from datetime import datetime, timezone
import csv
import json
import os
from pathlib import Path
from typing import Any

os.environ["MPLBACKEND"] = "Agg"

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_CACHE = Path(
    "artifacts/teacher_cache/torchvision_fasterrcnn_coco80_letterbox_full_score005_poibin.jsonl"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/reports/tallyqa_teacher_probability_temperature_fit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid-search a scalar probability temperature for TallyQA teacher caches."
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--answer-min", type=int, default=0)
    parser.add_argument("--answer-max", type=int, default=15)
    parser.add_argument("--collapse-at", type=int, default=5)
    parser.add_argument("--temperature-min", type=float, default=0.5)
    parser.add_argument("--temperature-max", type=float, default=8.0)
    parser.add_argument("--temperature-steps", type=int, default=76)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--eps", type=float, default=1e-12)
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
) -> list[float]:
    probabilities = {label: 0.0 for label in classes}
    for candidate in row["teacher_logits"]["numeric_answer_candidates"]:
        label = output_class(int(candidate["answer"]), collapse_at)
        if label in probabilities:
            probabilities[label] += float(candidate["candidate_probability"])
    values = np.array([probabilities[label] for label in classes], dtype=np.float64)
    total = float(values.sum())
    if total <= 0:
        raise ValueError("Teacher probabilities sum to zero.")
    return list(values / total)


def load_arrays(
    path: Path,
    classes: list[int | str],
    collapse_at: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[list[float]] = []
    labels: list[int] = []
    class_to_index = {label: index for index, label in enumerate(classes)}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            rows.append(row_distribution(row, classes, collapse_at))
            labels.append(class_to_index[output_class(int(row["answer"]), collapse_at)])
    return np.array(rows, dtype=np.float64), np.array(labels, dtype=np.int64)


def apply_temperature(probabilities: np.ndarray, temperature: float, eps: float) -> np.ndarray:
    scaled = np.where(probabilities > 0, np.power(np.maximum(probabilities, eps), 1.0 / temperature), 0.0)
    return scaled / np.maximum(scaled.sum(axis=1, keepdims=True), eps)


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    bins: int,
) -> float:
    predictions = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)
    correct = predictions == labels
    ece = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        if index == bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        count = int(mask.sum())
        if count:
            ece += (count / len(labels)) * abs(float(correct[mask].mean()) - float(confidences[mask].mean()))
    return ece


def score_temperature(
    probabilities: np.ndarray,
    labels: np.ndarray,
    temperature: float,
    bins: int,
    eps: float,
) -> dict[str, float]:
    scaled = apply_temperature(probabilities, temperature, eps)
    predictions = scaled.argmax(axis=1)
    confidence = scaled.max(axis=1)
    true_probability = np.maximum(scaled[np.arange(len(labels)), labels], eps)
    return {
        "temperature": float(temperature),
        "accuracy": float((predictions == labels).mean()),
        "mean_confidence": float(confidence.mean()),
        "nll": float(-np.log(true_probability).mean()),
        "ece": expected_calibration_error(scaled, labels, bins),
    }


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_rows(rows: list[dict[str, float]], output: Path) -> None:
    temperatures = [row["temperature"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].plot(temperatures, [row["nll"] for row in rows], linewidth=1.8)
    axes[0].set_xlabel("Probability temperature")
    axes[0].set_ylabel("NLL")
    axes[0].grid(alpha=0.25)
    axes[1].plot(temperatures, [row["ece"] for row in rows], linewidth=1.8)
    axes[1].set_xlabel("Probability temperature")
    axes[1].set_ylabel("ECE")
    axes[1].grid(alpha=0.25)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.temperature_min <= 0 or args.temperature_max <= 0:
        raise ValueError("Temperature bounds must be positive.")
    if args.temperature_steps <= 1:
        raise ValueError("--temperature-steps must be greater than 1.")
    if args.bins <= 0:
        raise ValueError("--bins must be positive.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    classes = output_classes(args.answer_min, args.answer_max, args.collapse_at)
    probabilities, labels = load_arrays(args.cache, classes, args.collapse_at)
    temperatures = np.linspace(args.temperature_min, args.temperature_max, args.temperature_steps)
    rows = [
        score_temperature(probabilities, labels, float(temperature), args.bins, args.eps)
        for temperature in temperatures
    ]
    best_nll = min(rows, key=lambda row: row["nll"])
    best_ece = min(rows, key=lambda row: row["ece"])
    write_csv(args.output_dir / "temperature_grid.csv", rows)
    plot_rows(rows, args.output_dir / "temperature_grid.png")
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cache": str(args.cache),
        "records": int(len(labels)),
        "answer_min": args.answer_min,
        "answer_max": args.answer_max,
        "collapse_at": args.collapse_at,
        "temperature_min": args.temperature_min,
        "temperature_max": args.temperature_max,
        "temperature_steps": args.temperature_steps,
        "bins": args.bins,
        "best_nll": best_nll,
        "best_ece": best_ece,
        "recommended_teacher_probability_temperature": best_nll["temperature"],
        "outputs": {
            "grid_csv": str(args.output_dir / "temperature_grid.csv"),
            "grid_plot": str(args.output_dir / "temperature_grid.png"),
        },
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
