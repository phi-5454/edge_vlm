from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import matplotlib
from datasets import load_from_disk
from PIL import Image
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from eda_tallyqa import load_pruned_suffixes, match_frontier_suffix, normalize_tokens
from eda_tallyqa_cauldron import clean_question, parse_answer


DEFAULT_DATASET = Path("data/the_cauldron/tallyqa")
DEFAULT_SUFFIXES = Path("artifacts/reports/tallyqa_cauldron_eda/frontier_suffixes_pruned.txt")
DEFAULT_ITEMS = Path("artifacts/reports/tallyqa_cauldron_eda/template_items_top200_pruned.txt")
DEFAULT_OUTPUT_ROOT = Path("data/tallyqa_cauldron_target_mobilenet224")
DEFAULT_REPORT = Path("artifacts/reports/tallyqa_cauldron_target_mobilenet224_summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a targeted Cauldron TallyQA item-counting dataset."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--suffixes", type=Path, default=DEFAULT_SUFFIXES)
    parser.add_argument("--items", type=Path, default=DEFAULT_ITEMS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-sha256", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_item_classes(path: Path) -> list[dict[str, Any]]:
    classes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        item_text = parts[1] if len(parts) >= 2 else parts[0]
        item = " ".join(normalize_tokens(item_text))
        if not item or item in seen:
            continue
        seen.add(item)
        classes.append(
            {
                "class_id": len(classes),
                "item": item,
                "source_rank": int(parts[0]) if parts and parts[0].isdigit() else None,
                "source_count": int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else None,
                "source_frequency": float(parts[3]) if len(parts) >= 4 else None,
            }
        )
    if not classes:
        raise ValueError(f"No item classes found in {path}")
    return classes


def write_classes(classes: list[dict[str, Any]], output_root: Path) -> None:
    (output_root / "classes.json").write_text(
        json.dumps(classes, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_root / "classes.txt").write_text(
        "\n".join(f"{row['class_id']:03d}\t{row['item']}" for row in classes) + "\n",
        encoding="utf-8",
    )


def match_item_question(
    question: str,
    suffixes: list[tuple[str, ...]],
    item_to_class: dict[str, int],
) -> tuple[str, int, str] | None:
    tokens = normalize_tokens(clean_question(question))
    if tokens[:2] != ["how", "many"]:
        return None
    match = match_frontier_suffix(tokens[2:], suffixes)
    if match is None:
        return None
    item, suffix = match
    class_id = item_to_class.get(item)
    if class_id is None:
        return None
    return item, class_id, " ".join(suffix)


def collect_examples(
    dataset_path: Path,
    suffixes: list[tuple[str, ...]],
    item_to_class: dict[str, int],
) -> tuple[list[dict[str, Any]], Counter[str], Counter[int], Counter[str]]:
    dataset = load_from_disk(str(dataset_path)).select_columns(["texts"])
    examples: list[dict[str, Any]] = []
    item_counter: Counter[str] = Counter()
    answer_counter: Counter[int] = Counter()
    suffix_counter: Counter[str] = Counter()
    for image_row_index, sample in enumerate(tqdm(dataset, desc="Filtering prompts", unit="image")):
        texts = sample.get("texts")
        if not isinstance(texts, list):
            continue
        for qa_index, message in enumerate(texts):
            if not isinstance(message, dict):
                continue
            teacher_prompt = message.get("user")
            answer = parse_answer(message.get("assistant"))
            if not isinstance(teacher_prompt, str) or answer is None:
                continue
            match = match_item_question(teacher_prompt, suffixes, item_to_class)
            if match is None:
                continue
            item, class_id, suffix = match
            example_id = len(examples)
            examples.append(
                {
                    "example_id": example_id,
                    "source_subset": "tallyqa",
                    "source_row_index": image_row_index,
                    "qa_index": qa_index,
                    "teacher_prompt": teacher_prompt,
                    "teacher_prompt_clean": clean_question(teacher_prompt),
                    "student_prompt": item,
                    "item": item,
                    "item_class_id": class_id,
                    "matched_suffix": suffix,
                    "answer": answer,
                    "answer_text": str(message.get("assistant") or ""),
                    "source": str(message.get("source") or "TallyQA"),
                    "image_id": f"tallyqa:{image_row_index}",
                    "image_row_index": image_row_index,
                }
            )
            item_counter.update([item])
            answer_counter.update([answer])
            suffix_counter.update([suffix])
    return examples, item_counter, answer_counter, suffix_counter


def image_tensor(image: Image.Image, image_size: int) -> np.ndarray:
    tensor = TF.pil_to_tensor(image.convert("RGB"))
    tensor = TF.resize(tensor, [image_size, image_size], antialias=True)
    return tensor.numpy()


def write_images(
    dataset_path: Path,
    output_root: Path,
    image_row_indices: set[int],
    image_size: int,
) -> dict[str, Any]:
    ordered_indices = sorted(image_row_indices)
    row_to_image_index = {row_index: image_index for image_index, row_index in enumerate(ordered_indices)}
    tensors = np.memmap(
        output_root / "images.uint8.bin",
        dtype=np.uint8,
        mode="w+",
        shape=(len(ordered_indices), 3, image_size, image_size),
    )
    dataset = load_from_disk(str(dataset_path))
    index_path = output_root / "images.index.jsonl"
    with index_path.open("w", encoding="utf-8") as index_handle:
        for row_index in tqdm(ordered_indices, desc="Writing uint8 images", unit="image"):
            row = dataset[row_index]
            images = row.get("images")
            if not isinstance(images, list) or not images:
                raise ValueError(f"Missing image list for dataset row {row_index}")
            image_index = row_to_image_index[row_index]
            tensors[image_index] = image_tensor(images[0], image_size)
            index_handle.write(
                json.dumps(
                    {
                        "image_id": f"tallyqa:{row_index}",
                        "image_index": image_index,
                        "source_row_index": row_index,
                    }
                )
                + "\n"
            )
    tensors.flush()
    return {
        "image_rows": len(ordered_indices),
        "shape": [len(ordered_indices), 3, image_size, image_size],
        "layout": "CHW",
        "dtype": "uint8",
        "resize": "torchvision.transforms.functional.resize to square with antialias=True",
        "normalization": "deferred; convert to float32 and apply MobileNet/ImageNet normalization during training",
        "tensor_file": "images.uint8.bin",
        "index_file": "images.index.jsonl",
    }


def update_example_image_indices(examples: list[dict[str, Any]], output_root: Path) -> None:
    mapping: dict[int, int] = {}
    with (output_root / "images.index.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            mapping[int(row["source_row_index"])] = int(row["image_index"])
    for example in examples:
        example["image_index"] = mapping[int(example["source_row_index"])]


def write_examples(examples: list[dict[str, Any]], output_root: Path, compression: str) -> None:
    table = pa.Table.from_pylist(examples)
    pq.write_table(table, output_root / "examples.parquet", compression=compression)
    with (output_root / "examples.jsonl").open("w", encoding="utf-8") as handle:
        for row in examples:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def plot_answer_histograms(answer_counter: Counter[int], figures_dir: Path) -> list[str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    answers = list(range(min(answer_counter), max(answer_counter) + 1)) if answer_counter else []
    counts = [answer_counter.get(answer, 0) for answer in answers]
    total = sum(counts)

    absolute = figures_dir / "answer_frequency_absolute.png"
    plt.figure(figsize=(8, 5))
    plt.bar([str(answer) for answer in answers], counts, color="#5f7d8c")
    plt.title("Targeted TallyQA answer frequency")
    plt.xlabel("Answer")
    plt.ylabel("Examples")
    plt.tight_layout()
    plt.savefig(absolute, dpi=160)
    plt.close()

    normalized = figures_dir / "answer_frequency_normalized.png"
    frequencies = [count / total if total else 0.0 for count in counts]
    plt.figure(figsize=(8, 5))
    plt.bar([str(answer) for answer in answers], [100 * value for value in frequencies], color="#8a5f2d")
    plt.title("Targeted TallyQA normalized answer frequency")
    plt.xlabel("Answer")
    plt.ylabel("Examples (%)")
    plt.tight_layout()
    plt.savefig(normalized, dpi=160)
    plt.close()
    return [str(absolute), str(normalized)]


def plot_answer_text_histograms(
    original_answer_counter: Counter[str],
    normalized_answer_counter: Counter[str],
    figures_dir: Path,
) -> list[str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for name, counter, color in (
        ("answer_text_original_frequency.png", original_answer_counter, "#5f7d8c"),
        ("answer_text_normalized_frequency.png", normalized_answer_counter, "#8a5f2d"),
    ):
        rows = counter.most_common()
        output = figures_dir / name
        plt.figure(figsize=(max(8, len(rows) * 0.42), 5))
        plt.bar([value for value, _ in rows], [count for _, count in rows], color=color)
        plt.title(name.removesuffix(".png").replace("_", " "))
        plt.xlabel("Answer text")
        plt.ylabel("Examples")
        plt.xticks(rotation=70, ha="right")
        plt.tight_layout()
        plt.savefig(output, dpi=160)
        plt.close()
        outputs.append(str(output))
    return outputs


def plot_student_query_histograms(item_counter: Counter[str], figures_dir: Path) -> list[str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    top_100 = item_counter.most_common(100)
    next_100 = item_counter.most_common(200)[100:200]

    first = figures_dir / "student_query_rank_001_100_log.png"
    plt.figure(figsize=(max(10, len(top_100) * 0.24), 5))
    plt.bar([item for item, _ in top_100], [count for _, count in top_100], color="#5f7d8c")
    plt.yscale("log")
    plt.title("Student query frequency ranks 1-100")
    plt.xlabel("Student query")
    plt.ylabel("Examples (log scale)")
    plt.xticks(rotation=70, ha="right")
    plt.tight_layout()
    plt.savefig(first, dpi=160)
    plt.close()

    second = figures_dir / "student_query_rank_101_200.png"
    plt.figure(figsize=(max(10, len(next_100) * 0.24), 5))
    plt.bar([item for item, _ in next_100], [count for _, count in next_100], color="#8a5f2d")
    plt.title("Student query frequency ranks 101-200")
    plt.xlabel("Student query")
    plt.ylabel("Examples")
    plt.xticks(rotation=70, ha="right")
    plt.tight_layout()
    plt.savefig(second, dpi=160)
    plt.close()
    return [str(first), str(second)]


def prepare_output_root(output_root: Path, force: bool) -> None:
    if output_root.exists() and not force:
        raise FileExistsError(f"{output_root} exists. Pass --force to replace it.")
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)


def main() -> None:
    args = parse_args()
    if args.image_size <= 0:
        raise ValueError("--image-size must be positive")
    suffixes = load_pruned_suffixes(args.suffixes)
    if not suffixes:
        raise ValueError(f"No suffixes found in {args.suffixes}")
    classes = load_item_classes(args.items)
    item_to_class = {str(row["item"]): int(row["class_id"]) for row in classes}

    prepare_output_root(args.output_root, args.force)
    write_classes(classes, args.output_root)

    examples, item_counter, answer_counter, suffix_counter = collect_examples(
        args.dataset,
        suffixes,
        item_to_class,
    )
    if not examples:
        raise ValueError("No examples matched the requested suffix and item filters.")
    image_metadata = write_images(
        args.dataset,
        args.output_root,
        {int(row["source_row_index"]) for row in examples},
        args.image_size,
    )
    update_example_image_indices(examples, args.output_root)
    write_examples(examples, args.output_root, args.compression)
    figures_dir = args.report.parent / "tallyqa_cauldron_target_mobilenet224"
    answer_figure_paths = plot_answer_histograms(answer_counter, figures_dir)
    answer_text_figure_paths = plot_answer_text_histograms(
        Counter(str(row["answer_text"]) for row in examples),
        Counter(str(row["answer"]) for row in examples),
        figures_dir,
    )
    query_figure_paths = plot_student_query_histograms(item_counter, figures_dir)

    tensor_path = args.output_root / "images.uint8.bin"
    index_path = args.output_root / "images.index.jsonl"
    examples_path = args.output_root / "examples.parquet"
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "source_dataset": str(args.dataset),
        "suffixes": str(args.suffixes),
        "items": str(args.items),
        "output_root": str(args.output_root),
        "examples": len(examples),
        "classes": len(classes),
        "unique_images": image_metadata["image_rows"],
        "image": image_metadata,
        "teacher_prompt": "original Cauldron user prompt is preserved, including newline/instruction",
        "student_prompt": "normalized extracted item only",
        "filtering": (
            "cleaned prompt removes the trailing Cauldron brief-answer instruction before "
            "matching normalized 'how many' + item + pruned suffix"
        ),
        "answer_counts": dict(sorted(answer_counter.items())),
        "answer_frequencies": {
            str(answer): count / len(examples) for answer, count in sorted(answer_counter.items())
        },
        "figures": {
            "answer_frequency_absolute": answer_figure_paths[0],
            "answer_frequency_normalized": answer_figure_paths[1],
            "answer_text_original_frequency": answer_text_figure_paths[0],
            "answer_text_normalized_frequency": answer_text_figure_paths[1],
            "student_query_rank_001_100_log": query_figure_paths[0],
            "student_query_rank_101_200": query_figure_paths[1],
        },
        "top_items": [
            {"item": item, "count": count, "frequency": count / len(examples)}
            for item, count in item_counter.most_common(50)
        ],
        "matched_suffixes": [
            {"suffix": suffix, "count": count, "frequency": count / len(examples)}
            for suffix, count in suffix_counter.most_common()
        ],
        "files": {
            "examples_parquet": "examples.parquet",
            "examples_jsonl": "examples.jsonl",
            "classes_json": "classes.json",
            "classes_txt": "classes.txt",
            "image_tensor": "images.uint8.bin",
            "image_index": "images.index.jsonl",
        },
        "sha256": {
            "examples_parquet": None if args.skip_sha256 else file_sha256(examples_path),
            "images_uint8_bin": None if args.skip_sha256 else file_sha256(tensor_path),
            "images_index_jsonl": file_sha256(index_path),
        },
    }
    (args.output_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote targeted dataset: {args.output_root}")
    print(f"Wrote report: {args.report}")


if __name__ == "__main__":
    main()
