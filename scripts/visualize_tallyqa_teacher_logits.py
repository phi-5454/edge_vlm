from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ["MPLBACKEND"] = "Agg"

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DATASET = Path("data/tallyqa_cauldron_target_mobilenet224_letterbox")
DEFAULT_CACHE = Path("artifacts/teacher_cache/smolvlm_tallyqa_target_mobilenet224_letterbox.jsonl")
DEFAULT_OUTPUT = Path("artifacts/reports/tallyqa_teacher_logit_examples/letterbox_examples.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize TallyQA teacher numeric answer probabilities beside images."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--student-prompt", default=None)
    parser.add_argument("--answer", type=int, default=None)
    parser.add_argument("--incorrect-only", action="store_true")
    parser.add_argument("--collapse-at", type=int, default=5)
    parser.add_argument("--answer-min", type=int, default=0)
    parser.add_argument("--answer-max", type=int, default=15)
    return parser.parse_args()


def load_metadata(dataset: Path) -> dict[str, Any]:
    return json.loads((dataset / "metadata.json").read_text(encoding="utf-8"))


def image_memmap(dataset: Path, metadata: dict[str, Any]) -> np.memmap:
    image_meta = metadata["image"]
    shape = tuple(int(dim) for dim in image_meta["shape"])
    return np.memmap(
        dataset / image_meta.get("tensor_file", "images.uint8.bin"),
        dtype=np.uint8,
        mode="r",
        shape=shape,
    )


def chw_to_hwc(image: np.ndarray) -> np.ndarray:
    return np.transpose(np.asarray(image), (1, 2, 0))


def output_class(answer: int, collapse_at: int | None) -> int | str:
    if collapse_at is not None and answer >= collapse_at:
        return f"{collapse_at}+"
    return answer


def output_classes(answer_min: int, answer_max: int, collapse_at: int | None) -> list[int | str]:
    if collapse_at is None:
        return list(range(answer_min, answer_max + 1))
    return list(range(answer_min, min(answer_max, collapse_at - 1) + 1)) + [f"{collapse_at}+"]


def collapsed_candidate_probs(row: dict[str, Any], labels: list[int | str], collapse_at: int) -> list[float]:
    totals = {label: 0.0 for label in labels}
    for candidate in row["teacher_logits"]["numeric_answer_candidates"]:
        label = output_class(int(candidate["answer"]), collapse_at)
        if label in totals:
            totals[label] += float(candidate["candidate_probability"])
    total = sum(totals.values())
    if total > 0:
        return [totals[label] / total for label in labels]
    return [0.0 for _ in labels]


def row_matches(row: dict[str, Any], args: argparse.Namespace) -> bool:
    if args.student_prompt is not None and row["student_prompt"] != args.student_prompt:
        return False
    if args.answer is not None and int(row["answer"]) != args.answer:
        return False
    if args.incorrect_only and bool(row["teacher_metrics"]["numeric_answer"]["correct"]):
        return False
    return True


def selected_cache_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, Any]] = []
    matched = 0
    with args.cache.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {args.cache}:{line_number}") from exc
            if not row_matches(row, args):
                continue
            if args.start_index is not None:
                if matched >= args.start_index and len(rows) < args.count:
                    rows.append(row)
                matched += 1
                if len(rows) >= args.count:
                    break
                continue
            matched += 1
            if len(rows) < args.count:
                rows.append(row)
            else:
                replace_index = int(rng.integers(0, matched))
                if replace_index < args.count:
                    rows[replace_index] = row
    if not rows:
        raise ValueError("No cache records matched the requested filters.")
    return rows


def plot_examples(rows: list[dict[str, Any]], images: np.memmap, args: argparse.Namespace) -> None:
    labels = output_classes(args.answer_min, args.answer_max, args.collapse_at)
    cols = min(args.cols, len(rows))
    plot_rows = math.ceil(len(rows) / cols)
    fig, axes = plt.subplots(
        plot_rows,
        cols * 2,
        figsize=(cols * 5.2, plot_rows * 3.2),
        gridspec_kw={"width_ratios": [1.15, 1.0] * cols},
    )
    axes_array = np.asarray(axes).reshape(plot_rows, cols * 2)
    for axis in axes_array.flat:
        axis.axis("off")

    for example_offset, row in enumerate(rows):
        grid_row = example_offset // cols
        grid_col = (example_offset % cols) * 2
        image_axis = axes_array[grid_row, grid_col]
        bar_axis = axes_array[grid_row, grid_col + 1]

        image_axis.axis("on")
        image_axis.imshow(chw_to_hwc(images[int(row["image_index"])]))
        true_label = output_class(int(row["answer"]), args.collapse_at)
        prediction = output_class(
            int(row["teacher_metrics"]["numeric_answer"]["prediction"]),
            args.collapse_at,
        )
        correct = prediction == true_label
        image_axis.set_title(
            "\n".join(
                [
                    f"idx={row['dataset_index']} image={row['image_index']}",
                    f"prompt: {row['student_prompt']}",
                    f"true={true_label} pred={prediction} {'OK' if correct else 'wrong'}",
                ]
            ),
            fontsize=8,
        )
        image_axis.axis("off")

        bar_axis.axis("on")
        probabilities = collapsed_candidate_probs(row, labels, args.collapse_at)
        y = np.arange(len(labels))
        colors = [
            "#59a14f" if label == true_label else "#e15759" if label == prediction else "#4e79a7"
            for label in labels
        ]
        bar_axis.barh(y, probabilities, color=colors)
        bar_axis.set_yticks(y, labels=[str(label) for label in labels], fontsize=8)
        bar_axis.invert_yaxis()
        bar_axis.set_xlim(0, max(1.0, max(probabilities) * 1.1))
        bar_axis.set_xlabel("teacher prob", fontsize=8)
        bar_axis.tick_params(axis="x", labelsize=7)
        for index, probability in enumerate(probabilities):
            if probability > 0.01:
                bar_axis.text(probability + 0.01, index, f"{probability:.2f}", va="center", fontsize=7)

    fig.suptitle(f"TallyQA teacher numeric answer probabilities from {args.cache}", fontsize=12)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.cols <= 0:
        raise ValueError("--cols must be positive")
    metadata = load_metadata(args.dataset)
    images = image_memmap(args.dataset, metadata)
    if args.count <= 0:
        raise ValueError("--count must be positive")
    rows = selected_cache_rows(args)
    plot_examples(rows, images, args)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "cache": str(args.cache),
        "output": str(args.output),
        "selected_records": len(rows),
        "dataset_indices": [int(row["dataset_index"]) for row in rows],
        "filters": {
            "student_prompt": args.student_prompt,
            "answer": args.answer,
            "incorrect_only": args.incorrect_only,
            "seed": args.seed,
            "start_index": args.start_index,
            "collapse_at": args.collapse_at,
        },
    }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
