#!/usr/bin/env python3
"""Write a small JSON report for a TFLite model's IO tensors."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import warnings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/reports/tflite_model_inspection.json"),
    )
    parser.add_argument("--verbose", action="store_true", help="Show TensorFlow import logs.")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_tensor(tensor: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": tensor.get("name"),
        "index": int(tensor["index"]),
        "shape": [int(value) for value in tensor["shape"]],
        "shape_signature": [int(value) for value in tensor.get("shape_signature", [])],
        "dtype": str(tensor["dtype"]),
        "quantization": tuple(float(value) for value in tensor.get("quantization", ())),
        "quantization_parameters": {
            key: [int(v) if hasattr(v, "item") else v for v in value.tolist()]
            if hasattr(value, "tolist")
            else value
            for key, value in tensor.get("quantization_parameters", {}).items()
        },
    }


@contextmanager
def suppress_stderr():
    stderr_fd = 2
    saved_fd = os.dup(stderr_fd)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), stderr_fd)
        yield
    finally:
        os.dup2(saved_fd, stderr_fd)
        os.close(saved_fd)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    warnings.filterwarnings(
        "ignore",
        message=r".*tf\.lite\.Interpreter is deprecated.*",
        category=UserWarning,
    )
    context = nullcontext() if args.verbose else suppress_stderr()
    with context:
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise SystemExit("TensorFlow is required. Install with `uv sync --extra coral`.") from exc

        interpreter = tf.lite.Interpreter(model_path=str(args.model))
    report = {
        "model": str(args.model),
        "bytes": args.model.stat().st_size,
        "sha256": sha256(args.model),
        "inputs": [clean_tensor(tensor) for tensor in interpreter.get_input_details()],
        "outputs": [clean_tensor(tensor) for tensor in interpreter.get_output_details()],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
