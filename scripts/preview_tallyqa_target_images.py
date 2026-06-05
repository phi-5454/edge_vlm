from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import textwrap
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


DEFAULT_DATASET = Path("data/tallyqa_cauldron_target_mobilenet224_letterbox")
DEFAULT_OUTPUT = Path("artifacts/reports/tallyqa_target_preview/letterbox_preview.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview TallyQA target dataset images.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--student-prompt", default=None)
    parser.add_argument("--answer", type=int, default=None)
    parser.add_argument("--max-title-chars", type=int, default=44)
    return parser.parse_args()


def load_metadata(dataset: Path) -> dict[str, Any]:
    return json.loads((dataset / "metadata.json").read_text(encoding="utf-8"))


def load_examples(dataset: Path) -> list[dict[str, Any]]:
    return pq.read_table(dataset / "examples.parquet").to_pylist()


def filtered_indices(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[int]:
    indices = list(range(len(rows)))
    if args.student_prompt is not None:
        wanted = args.student_prompt.strip().lower()
        indices = [
            index for index in indices if str(rows[index]["student_prompt"]).lower() == wanted
        ]
    if args.answer is not None:
        indices = [index for index in indices if int(rows[index]["answer"]) == args.answer]
    if args.random:
        rng = np.random.default_rng(args.seed)
        rng.shuffle(indices)
        return indices[: args.count]
    return indices[args.start_index : args.start_index + args.count]


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


def title_for_row(example_index: int, row: dict[str, Any], max_chars: int) -> str:
    title = (
        f"#{example_index} | {row['student_prompt']} | y={row['answer']} "
        f"| image={row['image_index']}"
    )
    return "\n".join(textwrap.wrap(title, width=max_chars))


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive")
    if args.cols <= 0:
        raise ValueError("--cols must be positive")
    rows = load_examples(args.dataset)
    metadata = load_metadata(args.dataset)
    images = image_memmap(args.dataset, metadata)
    indices = filtered_indices(rows, args)
    if not indices:
        raise ValueError("No examples matched the requested filters.")

    cols = min(args.cols, len(indices))
    rows_count = math.ceil(len(indices) / cols)
    fig, axes = plt.subplots(rows_count, cols, figsize=(cols * 3.0, rows_count * 3.5))
    axes_array = np.asarray(axes).reshape(rows_count, cols)
    for axis in axes_array.flat:
        axis.axis("off")

    for axis, dataset_index in zip(axes_array.flat, indices, strict=False):
        row = rows[dataset_index]
        image_index = int(row["image_index"])
        axis.imshow(chw_to_hwc(images[image_index]))
        axis.set_title(title_for_row(dataset_index, row, args.max_title_chars), fontsize=8)
        axis.axis("off")

    fig.suptitle(
        f"TallyQA target preview: {len(indices)} examples from {args.dataset}",
        fontsize=12,
    )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)

    summary = {
        "dataset": str(args.dataset),
        "output": str(args.output),
        "selected_examples": len(indices),
        "indices": indices,
        "filters": {
            "student_prompt": args.student_prompt,
            "answer": args.answer,
            "random": args.random,
            "seed": args.seed,
        },
    }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
