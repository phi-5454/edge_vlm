from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, load_from_disk
from tqdm.auto import tqdm


DEFAULT_INPUT_ROOT = Path("data/the_cauldron_yes_no_vsr_token1000_img512")
DEFAULT_OUTPUT_ROOT = Path("data/the_cauldron_yes_no_vsr_token1000_img512_parquet")
DEFAULT_REPORT = Path("artifacts/reports/cauldron_yes_no_vsr_token1000_img512_parquet_summary.json")
EXAMPLE_COLUMNS = [
    "source_subset",
    "original_index",
    "qa_index",
    "source",
    "teacher_prompt",
    "student_prompt",
    "removed_last_line",
    "answer",
    "student_token_ids",
    "student_token_count",
    "student_distinct_token_count",
    "student_image_id",
    "student_image_path",
    "student_image_sha256",
    "student_image_format",
    "original_size",
    "resized_content_size",
    "canvas_size",
    "scale",
    "padding",
    "background_rgb",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the 512x512 student image dataset into parquet files."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def image_id(row: dict[str, Any]) -> str:
    return f"{row['source_subset']}:{int(row['original_index'])}"


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    normalized["student_image_id"] = image_id(row)
    return {column: normalized[column] for column in EXAMPLE_COLUMNS}


def write_dataset_parquet(
    split: Dataset,
    output: Path,
    batch_size: int,
    compression: str,
) -> None:
    writer: pq.ParquetWriter | None = None
    try:
        for start in tqdm(range(0, len(split), batch_size), desc=f"Writing {output.name}", unit="batch"):
            stop = min(len(split), start + batch_size)
            rows = [normalize_row(split[index]) for index in range(start, stop)]
            table = pa.Table.from_pylist(rows)
            if writer is None:
                writer = pq.ParquetWriter(output, table.schema, compression=compression)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def collect_unique_image_rows(combined: Dataset, images_root: Path) -> list[dict[str, Any]]:
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for row in tqdm(combined, desc="Collecting unique image bytes", unit="row"):
        key = image_id(row)
        if key in seen:
            continue
        seen.add(key)
        relative_path = str(row["student_image_path"])
        image_path = images_root / relative_path
        rows.append(
            {
                "student_image_id": key,
                "student_image_path": relative_path,
                "student_image_sha256": row["student_image_sha256"],
                "student_image_format": row["student_image_format"],
                "source_subset": row["source_subset"],
                "original_index": int(row["original_index"]),
                "original_size": row["original_size"],
                "resized_content_size": row["resized_content_size"],
                "canvas_size": row["canvas_size"],
                "scale": row["scale"],
                "padding": row["padding"],
                "background_rgb": row["background_rgb"],
                "image_bytes": image_path.read_bytes(),
            }
        )
    return rows


def write_images_parquet(
    image_rows: list[dict[str, Any]],
    output: Path,
    batch_size: int,
    compression: str,
) -> None:
    writer: pq.ParquetWriter | None = None
    try:
        for start in tqdm(range(0, len(image_rows), batch_size), desc="Writing images.parquet", unit="batch"):
            stop = min(len(image_rows), start + batch_size)
            table = pa.Table.from_pylist(image_rows[start:stop])
            if writer is None:
                writer = pq.ParquetWriter(output, table.schema, compression=compression)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def main() -> None:
    args = parse_args()
    dataset_path = args.input_root / "dataset"
    images_root = args.input_root
    if args.output_root.exists() and not args.force:
        raise FileExistsError(f"{args.output_root} exists. Pass --force to replace it.")
    if args.output_root.exists():
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    dataset_dict = load_from_disk(dataset_path)
    for split_name, split in dataset_dict.items():
        write_dataset_parquet(
            split=split,
            output=args.output_root / f"{split_name}.parquet",
            batch_size=args.batch_size,
            compression=args.compression,
        )

    image_rows = collect_unique_image_rows(dataset_dict["combined"], images_root)
    write_images_parquet(
        image_rows=image_rows,
        output=args.output_root / "images.parquet",
        batch_size=args.batch_size,
        compression=args.compression,
    )

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "dataset_path": str(dataset_path),
        "images_parquet": str(args.output_root / "images.parquet"),
        "compression": args.compression,
        "splits": {split_name: len(split) for split_name, split in dataset_dict.items()},
        "unique_images": len(image_rows),
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote parquet dataset: {args.output_root}")
    print(f"Wrote report: {args.report}")


if __name__ == "__main__":
    main()
