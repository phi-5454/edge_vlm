from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from scripts.build_student_uint8_dataset import build_dataset, uint8_tensor


def _jpeg_bytes(color: tuple[int, int, int]) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (32, 48), color).save(output, format="JPEG")
    return output.getvalue()


def test_build_uint8_memmap_dataset(tmp_path: Path) -> None:
    image_rows = [
        {"student_image_id": "clevr:0", "image_bytes": _jpeg_bytes((10, 20, 30))},
        {"student_image_id": "clevr:1", "image_bytes": _jpeg_bytes((40, 50, 60))},
    ]
    input_path = tmp_path / "images.parquet"
    output_root = tmp_path / "uint8"
    report_path = tmp_path / "report.json"
    pq.write_table(pa.Table.from_pylist(image_rows), input_path, row_group_size=1)

    report = build_dataset(input_path, output_root, report_path, image_size=16)

    tensors = np.memmap(output_root / "images.uint8.bin", dtype=np.uint8, mode="r", shape=(2, 3, 16, 16))
    index_rows = [
        json.loads(line)
        for line in (output_root / "images.index.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert report["shape"] == [2, 3, 16, 16]
    assert report["tensor_file_bytes"] == 2 * 3 * 16 * 16
    assert report["tensor_file_sha256"]
    assert index_rows == [
        {"student_image_id": "clevr:0", "row_index": 0},
        {"student_image_id": "clevr:1", "row_index": 1},
    ]
    np.testing.assert_array_equal(tensors[0], uint8_tensor(image_rows[0]["image_bytes"], 16))
    assert json.loads(report_path.read_text(encoding="utf-8")) == report


def test_build_uint8_memmap_dataset_resumes_partial_output(tmp_path: Path) -> None:
    image_rows = [
        {"student_image_id": "clevr:0", "image_bytes": _jpeg_bytes((10, 20, 30))},
        {"student_image_id": "clevr:1", "image_bytes": _jpeg_bytes((40, 50, 60))},
    ]
    input_path = tmp_path / "images.parquet"
    output_root = tmp_path / "uint8"
    tmp_output = tmp_path / ".uint8.tmp"
    pq.write_table(pa.Table.from_pylist(image_rows), input_path)
    tmp_output.mkdir()
    tensors = np.memmap(tmp_output / "images.uint8.bin", dtype=np.uint8, mode="w+", shape=(2, 3, 16, 16))
    tensors[0] = uint8_tensor(image_rows[0]["image_bytes"], 16)
    tensors.flush()
    (tmp_output / "images.index.jsonl").write_text(
        json.dumps({"student_image_id": "clevr:0", "row_index": 0}) + "\n",
        encoding="utf-8",
    )

    build_dataset(input_path, output_root, tmp_path / "report.json", image_size=16, resume=True)

    resumed = np.memmap(output_root / "images.uint8.bin", dtype=np.uint8, mode="r", shape=(2, 3, 16, 16))
    np.testing.assert_array_equal(resumed[0], uint8_tensor(image_rows[0]["image_bytes"], 16))
    np.testing.assert_array_equal(resumed[1], uint8_tensor(image_rows[1]["image_bytes"], 16))
