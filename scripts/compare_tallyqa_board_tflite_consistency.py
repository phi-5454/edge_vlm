#!/usr/bin/env python3
"""Compare local TFLite and Coral Micro TallyQA cache outputs.

This is a deployment consistency check, not an accuracy comparison. It expects
two teacher-cache JSONL files produced over overlapping dataset indices and
compares dequantized logits, probabilities, and argmax predictions example by
example.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_OUTPUT_DIR = Path("artifacts/reports/coral/board_tflite_consistency")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-cache", type=Path, required=True)
    parser.add_argument("--board-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--logit-tolerance",
        type=float,
        default=None,
        help=(
            "Absolute dequantized-logit tolerance. Defaults to 1.5x the board output "
            "quantization scale when available, with a 1e-5 floor for float formatting "
            "noise."
        ),
    )
    parser.add_argument("--probability-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--fail-on-mismatch", action="store_true")
    return parser.parse_args()


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64) - float(np.max(logits))
    exp = np.exp(shifted)
    total = float(exp.sum())
    if total <= 0.0:
        return np.full(logits.shape, 1.0 / logits.size, dtype=np.float64)
    return exp / total


def read_cache(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            rows[int(row["dataset_index"])] = row
    return rows


def raw_logits(row: dict[str, Any]) -> np.ndarray:
    logits = row.get("teacher_logits", {}).get("raw_logits")
    if logits is None:
        raise ValueError(f"Row {row.get('dataset_index')} is missing teacher_logits.raw_logits")
    values = np.asarray(logits, dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise ValueError(f"Row {row.get('dataset_index')} has empty logits")
    return values


def prediction(row: dict[str, Any]) -> int | None:
    metrics = row.get("teacher_metrics", {}).get("numeric_answer", {})
    if "prediction" not in metrics:
        return None
    return int(metrics["prediction"])


def output_scale(row: dict[str, Any]) -> float | None:
    for output in row.get("board_result", {}).get("outputs", []):
        scale = output.get("scale")
        if scale:
            return float(scale)
    for output in row.get("teacher_logits", {}).get("board_outputs", []):
        scale = output.get("scale")
        if scale:
            return float(scale)
    return None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    local_rows = read_cache(args.local_cache)
    board_rows = read_cache(args.board_cache)
    common_indices = sorted(set(local_rows) & set(board_rows))
    if not common_indices:
        raise RuntimeError(
            "No overlapping dataset indices. Run local and board caches with the same "
            "--start-index/--max-examples selection."
        )

    inferred_scale = next(
        (scale for index in common_indices if (scale := output_scale(board_rows[index]))),
        None,
    )
    logit_tolerance = (
        float(args.logit_tolerance)
        if args.logit_tolerance is not None
        else max(1.5 * inferred_scale, 1.0e-5) if inferred_scale else 1.0e-5
    )

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index in common_indices:
        local = local_rows[index]
        board = board_rows[index]
        local_logits = raw_logits(local)
        board_logits = raw_logits(board)
        if local_logits.shape != board_logits.shape:
            row = {
                "dataset_index": index,
                "status": "shape_mismatch",
                "local_shape": list(local_logits.shape),
                "board_shape": list(board_logits.shape),
            }
            rows.append(row)
            failures.append(row)
            continue
        local_probs = softmax(local_logits)
        board_probs = softmax(board_logits)
        logit_delta = board_logits.astype(np.float64) - local_logits.astype(np.float64)
        prob_delta = board_probs - local_probs
        local_prediction = prediction(local)
        board_prediction = prediction(board)
        max_abs_logit_delta = float(np.max(np.abs(logit_delta)))
        max_abs_probability_delta = float(np.max(np.abs(prob_delta)))
        row = {
            "dataset_index": index,
            "example_id": local.get("example_id"),
            "image_index": local.get("image_index"),
            "student_prompt": local.get("student_prompt"),
            "answer": local.get("answer"),
            "local_prediction": local_prediction,
            "board_prediction": board_prediction,
            "prediction_match": local_prediction == board_prediction,
            "max_abs_logit_delta": max_abs_logit_delta,
            "mean_abs_logit_delta": float(np.mean(np.abs(logit_delta))),
            "max_abs_probability_delta": max_abs_probability_delta,
            "mean_abs_probability_delta": float(np.mean(np.abs(prob_delta))),
            "within_logit_tolerance": max_abs_logit_delta <= logit_tolerance,
            "within_probability_tolerance": (
                max_abs_probability_delta <= float(args.probability_tolerance)
            ),
        }
        rows.append(row)
        if (
            not row["prediction_match"]
            or not row["within_logit_tolerance"]
            or not row["within_probability_tolerance"]
        ):
            failures.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "board_tflite_consistency_rows.csv"
    failures_path = args.output_dir / "board_tflite_consistency_failures.csv"
    write_csv(rows_path, rows)
    write_csv(failures_path, failures)

    max_logit_deltas = [float(row.get("max_abs_logit_delta", 0.0)) for row in rows]
    max_prob_deltas = [float(row.get("max_abs_probability_delta", 0.0)) for row in rows]
    prediction_matches = [bool(row.get("prediction_match")) for row in rows]
    logit_tolerance_matches = [bool(row.get("within_logit_tolerance")) for row in rows]
    probability_tolerance_matches = [
        bool(row.get("within_probability_tolerance")) for row in rows
    ]
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "local_cache": str(args.local_cache),
        "board_cache": str(args.board_cache),
        "local_records": len(local_rows),
        "board_records": len(board_rows),
        "intersecting_records": len(common_indices),
        "first_common_index": common_indices[0],
        "last_common_index": common_indices[-1],
        "inferred_board_output_scale": inferred_scale,
        "logit_tolerance": logit_tolerance,
        "probability_tolerance": float(args.probability_tolerance),
        "prediction_match_rate": float(np.mean(prediction_matches)) if rows else None,
        "logit_tolerance_pass_rate": (
            float(np.mean(logit_tolerance_matches)) if rows else None
        ),
        "probability_tolerance_pass_rate": (
            float(np.mean(probability_tolerance_matches)) if rows else None
        ),
        "max_abs_logit_delta": max(max_logit_deltas) if max_logit_deltas else None,
        "mean_max_abs_logit_delta": (
            float(np.mean(max_logit_deltas)) if max_logit_deltas else None
        ),
        "max_abs_probability_delta": max(max_prob_deltas) if max_prob_deltas else None,
        "mean_max_abs_probability_delta": (
            float(np.mean(max_prob_deltas)) if max_prob_deltas else None
        ),
        "failure_count": len(failures),
        "outputs": {
            "rows": str(rows_path),
            "failures": str(failures_path),
            "summary": str(args.output_dir / "summary.json"),
        },
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.fail_on_mismatch and failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
