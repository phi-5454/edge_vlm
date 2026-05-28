from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_from_disk
from PIL import Image
from tqdm.auto import tqdm


DEFAULT_PROMPT_DATASET = Path("data/the_cauldron_yes_no_vsr_token1000")
DEFAULT_SOURCE_ROOT = Path("data/the_cauldron")
DEFAULT_OUTPUT_ROOT = Path("data/the_cauldron_yes_no_vsr_token1000_img512")
DEFAULT_REPORT = Path("artifacts/reports/cauldron_yes_no_vsr_token1000_img512_summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a student dataset with 512x512 padded sidecar images."
    )
    parser.add_argument("--prompt-dataset", type=Path, default=DEFAULT_PROMPT_DATASET)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--background", nargs=3, type=int, default=[0, 0, 0])
    parser.add_argument("--image-format", choices=["png", "jpeg"], default="jpeg")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def first_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, list):
        for item in value:
            try:
                return first_image(item)
            except ValueError:
                pass
    raise ValueError("No image found in source row")


def source_dataset(source_cache: dict[str, Any], source_root: Path, subset: str) -> Any:
    if subset not in source_cache:
        source_cache[subset] = load_from_disk(source_root / subset).select_columns(["images"])
    return source_cache[subset]


def padded_resize(
    image: Image.Image,
    size: int,
    background: tuple[int, int, int],
) -> tuple[Image.Image, dict[str, Any]]:
    image = image.convert("RGB")
    original_width, original_height = image.size
    resize_metadata = resize_metadata_from_size(
        original_width=original_width,
        original_height=original_height,
        size=size,
        background=background,
    )
    resized_width, resized_height = resize_metadata["resized_content_size"]
    resized = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), background)
    pad_left = resize_metadata["padding"]["left"]
    pad_top = resize_metadata["padding"]["top"]
    canvas.paste(resized, (pad_left, pad_top))
    return canvas, resize_metadata


def resize_metadata_from_size(
    original_width: int,
    original_height: int,
    size: int,
    background: tuple[int, int, int],
) -> dict[str, Any]:
    scale = min(size / original_width, size / original_height)
    resized_width = max(1, round(original_width * scale))
    resized_height = max(1, round(original_height * scale))
    pad_left = (size - resized_width) // 2
    pad_top = (size - resized_height) // 2
    return {
        "original_size": [original_width, original_height],
        "resized_content_size": [resized_width, resized_height],
        "canvas_size": [size, size],
        "scale": scale,
        "padding": {
            "left": pad_left,
            "top": pad_top,
            "right": size - resized_width - pad_left,
            "bottom": size - resized_height - pad_top,
        },
        "background_rgb": list(background),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_image(
    image: Image.Image,
    output: Path,
    image_format: str,
    jpeg_quality: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if image_format == "jpeg":
        image.save(output, format="JPEG", quality=jpeg_quality, optimize=True)
    else:
        image.save(output, format="PNG", optimize=True)


def collect_unique_images(prompt_dataset: Dataset) -> list[tuple[str, int]]:
    return sorted(
        {
            (str(row["source_subset"]), int(row["original_index"]))
            for row in tqdm(prompt_dataset, desc="Collecting unique image IDs", unit="row")
        }
    )


def build_images(
    unique_images: list[tuple[str, int]],
    source_root: Path,
    output_root: Path,
    image_size: int,
    background: tuple[int, int, int],
    image_format: str,
    jpeg_quality: int,
    resume: bool,
) -> dict[str, dict[str, Any]]:
    source_cache: dict[str, Any] = {}
    metadata: dict[str, dict[str, Any]] = {}
    extension = "jpg" if image_format == "jpeg" else "png"
    for subset, original_index in tqdm(unique_images, desc="Writing 512x512 images", unit="image"):
        dataset = source_dataset(source_cache, source_root, subset)
        image = first_image(dataset[original_index]["images"])
        resize_metadata = resize_metadata_from_size(
            original_width=image.width,
            original_height=image.height,
            size=image_size,
            background=background,
        )
        relative_path = Path("images") / subset / f"{original_index:08d}.{extension}"
        output = output_root / relative_path
        if resume and output.exists():
            with Image.open(output) as existing:
                if existing.size != (image_size, image_size):
                    raise ValueError(f"{output} has size {existing.size}, expected {(image_size, image_size)}")
        else:
            resized, resize_metadata = padded_resize(image, image_size, background)
            save_image(resized, output, image_format, jpeg_quality)
        key = f"{subset}:{original_index}"
        metadata[key] = {
            "student_image_path": str(relative_path),
            "student_image_sha256": file_sha256(output),
            "student_image_format": image_format,
            **resize_metadata,
        }
    return metadata


def enrich_split(split: Dataset, image_metadata: dict[str, dict[str, Any]]) -> Dataset:
    rows = []
    for row in tqdm(split, desc="Adding image metadata", unit="row"):
        enriched = dict(row)
        key = f"{row['source_subset']}:{int(row['original_index'])}"
        enriched.update(image_metadata[key])
        rows.append(enriched)
    return Dataset.from_list(rows)


def main() -> None:
    args = parse_args()
    if args.output_root.exists() and not args.force:
        raise FileExistsError(f"{args.output_root} exists. Pass --force to replace it.")
    if args.resume and args.force:
        raise ValueError("Use either --resume or --force, not both.")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = args.output_root.with_name(f".{args.output_root.name}.tmp")
    if tmp_output.exists() and not args.resume:
        shutil.rmtree(tmp_output)
    if args.output_root.exists():
        shutil.rmtree(args.output_root)
    tmp_output.mkdir(parents=True, exist_ok=args.resume)

    source_dataset_dict = load_from_disk(args.prompt_dataset)
    combined = source_dataset_dict["combined"]
    unique_images = collect_unique_images(combined)
    image_metadata = build_images(
        unique_images=unique_images,
        source_root=args.source_root,
        output_root=tmp_output,
        image_size=args.image_size,
        background=tuple(args.background),
        image_format=args.image_format,
        jpeg_quality=args.jpeg_quality,
        resume=args.resume,
    )
    enriched = {
        split_name: enrich_split(split, image_metadata)
        for split_name, split in source_dataset_dict.items()
    }
    DatasetDict(enriched).save_to_disk(tmp_output / "dataset")
    tmp_output.rename(args.output_root)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "prompt_dataset": str(args.prompt_dataset),
        "source_root": str(args.source_root),
        "output_root": str(args.output_root),
        "dataset_path": str(args.output_root / "dataset"),
        "images_path": str(args.output_root / "images"),
        "image_size": args.image_size,
        "resize": "aspect-preserving resize with symmetric padding to square canvas",
        "background_rgb": args.background,
        "image_format": args.image_format,
        "jpeg_quality": args.jpeg_quality if args.image_format == "jpeg" else None,
        "prompt_rows": len(combined),
        "unique_images": len(unique_images),
        "splits": {split_name: len(split) for split_name, split in enriched.items()},
    }
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote student image dataset: {args.output_root / 'dataset'}")
    print(f"Wrote resized images: {args.output_root / 'images'}")
    print(f"Wrote report: {args.report}")


if __name__ == "__main__":
    main()
