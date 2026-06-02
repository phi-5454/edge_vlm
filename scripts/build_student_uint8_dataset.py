from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from tqdm.auto import tqdm
from torchvision.transforms import functional as TF


DEFAULT_INPUT = Path("data/the_cauldron_yes_no_vsr_token1000_img512_parquet/images.parquet")
DEFAULT_OUTPUT_ROOT = Path("data/the_cauldron_yes_no_vsr_token1000_uint8_224")
DEFAULT_REPORT = Path("artifacts/reports/cauldron_yes_no_vsr_token1000_uint8_224_summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute 224x224 CHW uint8 student images into a memory-mapped binary file."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--skip-sha256",
        action="store_true",
        help="Skip the final output hash pass when iteration speed matters more than traceability.",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def completed_rows(index_path: Path) -> int:
    if not index_path.exists():
        return 0
    with index_path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def uint8_tensor(image_bytes: bytes, image_size: int) -> np.ndarray:
    with Image.open(io.BytesIO(image_bytes)) as image:
        tensor = TF.pil_to_tensor(image.convert("RGB"))
        tensor = TF.resize(tensor, [image_size, image_size], antialias=True)
    return tensor.numpy()


def prepare_output(
    output_root: Path,
    rows: int,
    image_size: int,
    resume: bool,
    force: bool,
) -> tuple[Path, np.memmap, int]:
    if resume and force:
        raise ValueError("Use either --resume or --force, not both.")
    if output_root.exists() and not force:
        raise FileExistsError(f"{output_root} exists. Pass --force to replace it.")

    tmp_output = output_root.with_name(f".{output_root.name}.tmp")
    if force:
        shutil.rmtree(output_root, ignore_errors=True)
        shutil.rmtree(tmp_output, ignore_errors=True)
    elif tmp_output.exists() and not resume:
        raise FileExistsError(f"{tmp_output} exists. Pass --resume or --force.")

    tmp_output.mkdir(parents=True, exist_ok=True)
    tensor_path = tmp_output / "images.uint8.bin"
    index_path = tmp_output / "images.index.jsonl"
    start = completed_rows(index_path) if resume else 0
    if start > rows:
        raise ValueError(f"{index_path} contains {start} rows, but the input contains only {rows}.")
    mode = "r+" if resume and tensor_path.exists() else "w+"
    tensors = np.memmap(tensor_path, dtype=np.uint8, mode=mode, shape=(rows, 3, image_size, image_size))
    return tmp_output, tensors, start


def build_dataset(
    input_path: Path,
    output_root: Path,
    report_path: Path,
    image_size: int,
    resume: bool = False,
    force: bool = False,
    skip_sha256: bool = False,
    command: list[str] | None = None,
) -> dict[str, Any]:
    if image_size <= 0:
        raise ValueError("image_size must be positive.")
    parquet = pq.ParquetFile(input_path)
    rows = parquet.metadata.num_rows
    tmp_output, tensors, start = prepare_output(output_root, rows, image_size, resume, force)
    index_path = tmp_output / "images.index.jsonl"

    current = 0
    mode = "a" if start else "w"
    with index_path.open(mode, encoding="utf-8") as index_handle:
        progress = tqdm(total=rows, initial=start, desc="Writing uint8 images", unit="image")
        try:
            for group_index in range(parquet.num_row_groups):
                table = parquet.read_row_group(group_index, columns=["student_image_id", "image_bytes"])
                for image_id, image_bytes in zip(
                    table.column("student_image_id").to_pylist(),
                    table.column("image_bytes").to_pylist(),
                    strict=True,
                ):
                    if current < start:
                        current += 1
                        continue
                    tensors[current] = uint8_tensor(image_bytes, image_size)
                    index_handle.write(
                        json.dumps({"student_image_id": str(image_id), "row_index": current}) + "\n"
                    )
                    current += 1
                    progress.update()
                tensors.flush()
                index_handle.flush()
        finally:
            progress.close()
    if current != rows:
        raise ValueError(f"Wrote {current} rows, expected {rows}.")

    tensor_path = tmp_output / "images.uint8.bin"
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "input_images_parquet": str(input_path),
        "input_images_parquet_bytes": input_path.stat().st_size,
        "output_root": str(output_root),
        "tensor_file": "images.uint8.bin",
        "index_file": "images.index.jsonl",
        "rows": rows,
        "shape": [rows, 3, image_size, image_size],
        "layout": "CHW",
        "dtype": "uint8",
        "resize": "torchvision.transforms.functional.resize with antialias=True",
        "normalization": "deferred; convert to float32 then apply ImageNet mean/std during training",
        "tensor_file_bytes": tensor_path.stat().st_size,
        "tensor_file_sha256": None if skip_sha256 else file_sha256(tensor_path),
        "index_file_sha256": file_sha256(index_path),
    }
    (tmp_output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    tmp_output.rename(output_root)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    args = parse_args()
    report = build_dataset(
        input_path=args.input,
        output_root=args.output_root,
        report_path=args.report,
        image_size=args.image_size,
        resume=args.resume,
        force=args.force,
        skip_sha256=args.skip_sha256,
        command=sys.argv,
    )
    print(f"Wrote uint8 tensor dataset: {report['output_root']}")
    print(f"Wrote report: {args.report}")


if __name__ == "__main__":
    main()
