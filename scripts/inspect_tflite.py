#!/usr/bin/env python3
"""Write a JSON/text report for TFLite IO tensors and operator stack."""

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
    parser.add_argument(
        "--text-output",
        type=Path,
        default=None,
        help="Optional plain-text operator stack report.",
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


EDGE_TPU_SUPPORTED_OPS = {
    "ADD",
    "AVERAGE_POOL_2D",
    "CONCATENATION",
    "CONV_2D",
    "DEPTHWISE_CONV_2D",
    "EXPAND_DIMS",
    "FULLY_CONNECTED",
    "L2_NORMALIZATION",
    "LOGISTIC",
    "LSTM",
    "MAXIMUM",
    "MAX_POOL_2D",
    "MEAN",
    "MINIMUM",
    "MUL",
    "PACK",
    "PAD",
    "PRELU",
    "QUANTIZE",
    "REDUCE_MAX",
    "REDUCE_MIN",
    "RELU",
    "RELU6",
    "RELU_N1_TO_1",
    "RESHAPE",
    "RESIZE_BILINEAR",
    "RESIZE_NEAREST_NEIGHBOR",
    "RSQRT",
    "SLICE",
    "SOFTMAX",
    "SPACE_TO_DEPTH",
    "SPLIT",
    "SQUEEZE",
    "STRIDED_SLICE",
    "SUB",
    "SUM",
    "SQUARED_DIFFERENCE",
    "TANH",
    "TRANSPOSE",
    "TRANSPOSE_CONV",
}


def tensor_summary(index: int, tensors: dict[int, dict[str, Any]]) -> dict[str, Any]:
    if index < 0:
        return {"index": int(index), "name": "<none>", "shape": [], "shape_signature": []}
    tensor = tensors[int(index)]
    return {
        "index": int(index),
        "name": tensor.get("name"),
        "shape": [int(value) for value in tensor.get("shape", [])],
        "shape_signature": [int(value) for value in tensor.get("shape_signature", [])],
        "dtype": str(tensor.get("dtype")),
    }


def has_dynamic_shape(tensor: dict[str, Any]) -> bool:
    return any(int(value) < 0 for value in tensor.get("shape_signature", []))


def op_warnings(op: dict[str, Any]) -> list[str]:
    op_name = op["op_name"]
    warnings_: list[str] = []
    if op_name == "DELEGATE":
        return warnings_
    if op_name not in EDGE_TPU_SUPPORTED_OPS:
        warnings_.append("op not in Edge TPU supported-op list")
    for tensor in [*op["inputs"], *op["outputs"]]:
        if has_dynamic_shape(tensor):
            warnings_.append("dynamic tensor shape_signature contains -1")
            break
    if op_name == "SOFTMAX":
        for tensor in op["inputs"]:
            shape = tensor.get("shape", [])
            elements = 1
            for value in shape:
                elements *= int(value)
            if len(shape) != 1:
                warnings_.append("Edge TPU docs list Softmax support only for 1-D input")
            if elements > 16_000:
                warnings_.append("Softmax input has more than 16,000 elements")
    if op_name == "MEAN":
        for tensor in op["inputs"]:
            shape = tensor.get("shape", [])
            if len(shape) >= 4 and int(shape[-1]) % 4 != 0:
                warnings_.append("Mean z-dimension may need to be a multiple of 4")
                break
    if op_name == "FULLY_CONNECTED":
        for tensor in op["outputs"]:
            shape = tensor.get("shape", [])
            if len(shape) > 2:
                warnings_.append("FullyConnected output is not one-dimensional/batched vector")
                break
    return sorted(set(warnings_))


def clean_op(op: dict[str, Any], tensors: dict[int, dict[str, Any]], index: int) -> dict[str, Any]:
    cleaned = {
        "index": index,
        "op_name": op.get("op_name"),
        "inputs": [tensor_summary(int(value), tensors) for value in op.get("inputs", [])],
        "outputs": [tensor_summary(int(value), tensors) for value in op.get("outputs", [])],
    }
    cleaned["warnings"] = op_warnings(cleaned)
    return cleaned


def format_shape(tensor: dict[str, Any]) -> str:
    shape = tensor.get("shape", [])
    signature = tensor.get("shape_signature", [])
    if signature and signature != shape:
        return f"{shape} sig={signature}"
    return str(shape)


def text_report(report: dict[str, Any]) -> str:
    lines = [
        f"Model: {report['model']}",
        f"Bytes: {report['bytes']}",
        f"SHA256: {report['sha256']}",
        "",
        "Inputs:",
    ]
    for tensor in report["inputs"]:
        lines.append(
            f"  #{tensor['index']} {tensor['name']} {tensor['dtype']} "
            f"shape={format_shape(tensor)} quant={tensor['quantization']}"
        )
    lines.append("")
    lines.append("Outputs:")
    for tensor in report["outputs"]:
        lines.append(
            f"  #{tensor['index']} {tensor['name']} {tensor['dtype']} "
            f"shape={format_shape(tensor)} quant={tensor['quantization']}"
        )
    lines.append("")
    lines.append("Operator Stack:")
    for op in report["operators"]:
        inputs = ", ".join(f"#{tensor['index']}:{format_shape(tensor)}" for tensor in op["inputs"])
        outputs = ", ".join(f"#{tensor['index']}:{format_shape(tensor)}" for tensor in op["outputs"])
        warning = ""
        if op["warnings"]:
            warning = "  WARN: " + "; ".join(op["warnings"])
        lines.append(f"  {op['index']:03d} {op['op_name']:<22} in=[{inputs}] out=[{outputs}]{warning}")
    lines.append("")
    lines.append("Warning Summary:")
    warning_counts: dict[str, int] = {}
    for op in report["operators"]:
        for warning in op["warnings"]:
            warning_counts[warning] = warning_counts.get(warning, 0) + 1
    if warning_counts:
        for warning, count in sorted(warning_counts.items()):
            lines.append(f"  {count:3d}x {warning}")
    else:
        lines.append("  none")
    return "\n".join(lines) + "\n"


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

        interpreter = tf.lite.Interpreter(
            model_path=str(args.model),
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
            experimental_preserve_all_tensors=True,
            experimental_delegates=[],
        )
        interpreter.allocate_tensors()
        tensors = {
            int(tensor["index"]): clean_tensor(tensor)
            for tensor in interpreter.get_tensor_details()
        }
    report = {
        "model": str(args.model),
        "bytes": args.model.stat().st_size,
        "sha256": sha256(args.model),
        "inputs": [clean_tensor(tensor) for tensor in interpreter.get_input_details()],
        "outputs": [clean_tensor(tensor) for tensor in interpreter.get_output_details()],
        "operators": [
            clean_op(op, tensors, index)
            for index, op in enumerate(interpreter._get_ops_details())  # noqa: SLF001
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    if args.text_output is not None:
        args.text_output.parent.mkdir(parents=True, exist_ok=True)
        args.text_output.write_text(text_report(report), encoding="utf-8")
        print(f"Wrote {args.text_output}")


if __name__ == "__main__":
    main()
