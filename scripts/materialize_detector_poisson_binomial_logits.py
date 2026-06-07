from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm


DEFAULT_INPUT = Path(
    "artifacts/teacher_cache/torchvision_fasterrcnn_coco80_letterbox_full_score005.jsonl"
)
DEFAULT_OUTPUT = Path(
    "artifacts/teacher_cache/torchvision_fasterrcnn_coco80_letterbox_full_score005_poibin.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize soft count logits for detector teacher caches from detection scores "
            "using a Poisson-binomial distribution."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--answer-min", type=int, default=0)
    parser.add_argument("--answer-max", type=int, default=15)
    parser.add_argument("--collapse-at", type=int, default=5)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def poisson_binomial_capped(
    probabilities: list[float],
    answer_max: int,
) -> list[float]:
    """Return P(count=k) for k < answer_max and P(count>=answer_max) at answer_max."""
    distribution = [0.0] * (answer_max + 1)
    distribution[0] = 1.0
    for probability in probabilities:
        p = min(1.0, max(0.0, float(probability)))
        next_distribution = [0.0] * (answer_max + 1)
        for count, mass in enumerate(distribution):
            if mass == 0.0:
                continue
            next_distribution[count] += mass * (1.0 - p)
            next_distribution[min(answer_max, count + 1)] += mass * p
        distribution = next_distribution
    total = sum(distribution)
    if total <= 0:
        distribution[0] = 1.0
        return distribution
    return [mass / total for mass in distribution]


def candidate_scores(
    probabilities: list[float],
    answer_min: int,
    answer_max: int,
    eps: float,
) -> list[dict[str, float | int]]:
    return [
        {
            "answer": answer,
            "candidate_probability": float(probabilities[answer]),
            "candidate_log_likelihood": math.log(max(float(probabilities[answer]), eps)),
        }
        for answer in range(answer_min, answer_max + 1)
    ]


def argmax_answer(probabilities: list[float], answer_min: int, answer_max: int) -> int:
    return max(range(answer_min, answer_max + 1), key=lambda answer: probabilities[answer])


def update_stats(stats: Counter, prompt: str, answer: int, prediction: int, collapse_at: int) -> None:
    for key in ("overall", f"prompt::{prompt}"):
        stats[(key, "total")] += 1
        stats[(key, "correct")] += int(prediction == answer)
        stats[(key, "within_1")] += int(abs(prediction - answer) <= 1)
        stats[(key, "collapsed_correct")] += int(
            min(prediction, collapse_at) == min(answer, collapse_at)
        )


def accuracy_block(stats: Counter, key: str) -> dict[str, Any]:
    total = int(stats[(key, "total")])
    return {
        "records": total,
        "accuracy": stats[(key, "correct")] / total if total else None,
        "within_1_accuracy": stats[(key, "within_1")] / total if total else None,
        "collapsed_accuracy": stats[(key, "collapsed_correct")] / total if total else None,
    }


def process_record(
    row: dict[str, Any],
    answer_min: int,
    answer_max: int,
    collapse_at: int,
    eps: float,
) -> tuple[dict[str, Any], int]:
    scores = [float(detection["score"]) for detection in row.get("detections", [])]
    probabilities = poisson_binomial_capped(scores, answer_max)
    prediction = argmax_answer(probabilities, answer_min, answer_max)
    answer = int(row["answer"])
    old_metrics = row.get("teacher_metrics", {}).get("numeric_answer", {})
    old_prediction = old_metrics.get("prediction")

    row["teacher_logits"] = {
        "numeric_answer_candidates": candidate_scores(
            probabilities,
            answer_min,
            answer_max,
            eps,
        )
    }
    row["teacher_metrics"] = {
        "numeric_answer": {
            "prediction": int(prediction),
            "prediction_text": str(prediction),
            "correct": prediction == answer,
            "within_1": abs(prediction - answer) <= 1,
            "collapsed_correct": min(prediction, collapse_at) == min(answer, collapse_at),
            "previous_hard_count_prediction": old_prediction,
        }
    }
    row["teacher_distribution"] = {
        "kind": "poisson_binomial_from_detection_scores",
        "score_source": "detections.score",
        "num_detection_scores": len(scores),
        "answer_max_tail_policy": f"P(count >= {answer_max}) is accumulated into answer {answer_max}",
        "uncalibrated": True,
    }
    return row, prediction


def main() -> None:
    args = parse_args()
    if args.answer_min != 0:
        raise ValueError("Only --answer-min 0 is currently supported.")
    if args.answer_max < args.answer_min:
        raise ValueError("--answer-max must be >= --answer-min.")
    if args.collapse_at < args.answer_min:
        raise ValueError("--collapse-at must be >= --answer-min.")
    if args.eps <= 0:
        raise ValueError("--eps must be positive.")
    if args.output.exists() and not args.force:
        raise FileExistsError(f"{args.output} exists. Pass --force to replace it.")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    stats: Counter = Counter()
    records = 0
    total_detection_scores = 0
    max_detection_scores = 0
    changed_argmax = 0

    with args.input.open("r", encoding="utf-8") as input_handle, args.output.open(
        "w", encoding="utf-8"
    ) as output_handle:
        for line_number, line in enumerate(tqdm(input_handle, desc="Materializing Poisson-binomial logits"), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {args.input}:{line_number}") from exc
            old_prediction = row.get("teacher_metrics", {}).get("numeric_answer", {}).get("prediction")
            row, prediction = process_record(
                row,
                args.answer_min,
                args.answer_max,
                args.collapse_at,
                args.eps,
            )
            prompt = str(row["student_prompt"])
            update_stats(stats, prompt, int(row["answer"]), prediction, args.collapse_at)
            scores_count = int(row["teacher_distribution"]["num_detection_scores"])
            total_detection_scores += scores_count
            max_detection_scores = max(max_detection_scores, scores_count)
            changed_argmax += int(old_prediction is not None and int(old_prediction) != prediction)
            records += 1
            output_handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    prompt_keys = sorted(
        key.removeprefix("prompt::")
        for key, metric in stats
        if metric == "total" and key.startswith("prompt::")
    )
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input),
        "output": str(args.output),
        "records": records,
        "answer_min": args.answer_min,
        "answer_max": args.answer_max,
        "collapse_at": args.collapse_at,
        "eps": args.eps,
        "distribution": {
            "kind": "poisson_binomial_from_detection_scores",
            "score_source": "detections.score",
            "answer_max_tail_policy": f"P(count >= {args.answer_max}) is accumulated into answer {args.answer_max}",
            "uncalibrated": True,
        },
        "detection_scores": {
            "total": total_detection_scores,
            "mean_per_record": total_detection_scores / records if records else None,
            "max_per_record": max_detection_scores,
        },
        "changed_argmax_from_hard_count": changed_argmax,
        "teacher_metrics": {
            "overall": accuracy_block(stats, "overall"),
            "by_prompt": {prompt: accuracy_block(stats, f"prompt::{prompt}") for prompt in prompt_keys},
        },
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
