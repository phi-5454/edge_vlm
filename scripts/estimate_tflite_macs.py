#!/usr/bin/env python3
"""Estimate MAC counts from a static-shape TFLite model."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import warnings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-output", type=Path, default=None)
    parser.add_argument(
        "--compiler-summary",
        type=Path,
        default=None,
        help="Optional compiler_summary.json to update with mac_estimate.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def shape(tensor: dict[str, Any]) -> list[int]:
    return [int(value) for value in tensor.get("shape", [])]


def product(values: list[int]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def conv2d_macs(inputs: list[dict[str, Any]], outputs: list[dict[str, Any]]) -> int:
    output_shape = shape(outputs[0])
    weight_shape = shape(inputs[1])
    if len(output_shape) != 4 or len(weight_shape) != 4:
        return 0
    batch, out_h, out_w, out_c = output_shape
    weight_out_c, kernel_h, kernel_w, in_c = weight_shape
    if weight_out_c != out_c:
        # TFLite Conv2D weights are normally [out_c, kernel_h, kernel_w, in_c].
        out_c = weight_out_c
    return int(batch * out_h * out_w * out_c * kernel_h * kernel_w * in_c)


def depthwise_conv2d_macs(inputs: list[dict[str, Any]], outputs: list[dict[str, Any]]) -> int:
    output_shape = shape(outputs[0])
    weight_shape = shape(inputs[1])
    if len(output_shape) != 4 or len(weight_shape) != 4:
        return 0
    batch, out_h, out_w, out_c = output_shape
    _, kernel_h, kernel_w, _ = weight_shape
    return int(batch * out_h * out_w * out_c * kernel_h * kernel_w)


def fully_connected_macs(inputs: list[dict[str, Any]], outputs: list[dict[str, Any]]) -> int:
    input_shape = shape(inputs[0])
    output_shape = shape(outputs[0])
    weight_shape = shape(inputs[1])
    if len(weight_shape) == 2:
        out_features, in_features = weight_shape
        batch = product(output_shape) // max(1, out_features)
        return int(batch * out_features * in_features)
    if input_shape and output_shape:
        return int(product(output_shape) * input_shape[-1])
    return 0


def reduction_ops(inputs: list[dict[str, Any]], outputs: list[dict[str, Any]]) -> int:
    if not inputs or not outputs:
        return 0
    return max(0, product(shape(inputs[0])) - product(shape(outputs[0])))


def elementwise_ops(inputs: list[dict[str, Any]], outputs: list[dict[str, Any]]) -> int:
    if outputs:
        return product(shape(outputs[0]))
    if inputs:
        return product(shape(inputs[0]))
    return 0


def clean_tensor(index: int, tensors: dict[int, dict[str, Any]]) -> dict[str, Any]:
    if index < 0:
        return {"index": int(index), "name": "<none>", "shape": []}
    tensor = tensors[int(index)]
    return {
        "index": int(index),
        "name": tensor.get("name"),
        "shape": shape(tensor),
        "dtype": str(tensor.get("dtype")),
    }


def estimate_op(op: dict[str, Any], tensors: dict[int, dict[str, Any]], index: int) -> dict[str, Any]:
    inputs = [clean_tensor(int(value), tensors) for value in op.get("inputs", [])]
    outputs = [clean_tensor(int(value), tensors) for value in op.get("outputs", [])]
    op_name = str(op.get("op_name"))
    macs = 0
    non_mac_ops = 0
    category = "other"
    if op_name == "CONV_2D":
        macs = conv2d_macs(inputs, outputs)
        category = "mac"
    elif op_name == "DEPTHWISE_CONV_2D":
        macs = depthwise_conv2d_macs(inputs, outputs)
        category = "mac"
    elif op_name == "FULLY_CONNECTED":
        macs = fully_connected_macs(inputs, outputs)
        category = "mac"
    elif op_name in {"ADD", "SUB", "MUL", "MAXIMUM", "MINIMUM", "SQUARED_DIFFERENCE"}:
        non_mac_ops = elementwise_ops(inputs, outputs)
        category = "elementwise"
    elif op_name in {"MEAN", "SUM", "REDUCE_MAX", "REDUCE_MIN"}:
        non_mac_ops = reduction_ops(inputs, outputs)
        category = "reduction"
    elif op_name in {"RELU", "RELU6", "LOGISTIC", "TANH", "SOFTMAX", "QUANTIZE", "DEQUANTIZE"}:
        non_mac_ops = elementwise_ops(inputs, outputs)
        category = "activation_or_quantization"
    return {
        "index": int(index),
        "op_name": op_name,
        "category": category,
        "macs": int(macs),
        "non_mac_ops_estimate": int(non_mac_ops),
        "inputs": inputs,
        "outputs": outputs,
    }


def human_count(value: int) -> str:
    for suffix, scale in (("G", 1_000_000_000), ("M", 1_000_000), ("K", 1_000)):
        if abs(value) >= scale:
            return f"{value / scale:.3f}{suffix}"
    return str(value)


def text_report(report: dict[str, Any]) -> str:
    lines = [
        f"Model: {report['model']}",
        f"SHA256: {report['sha256']}",
        f"Total MACs: {report['totals']['macs']} ({human_count(report['totals']['macs'])})",
        f"Non-MAC ops estimate: {report['totals']['non_mac_ops_estimate']} "
        f"({human_count(report['totals']['non_mac_ops_estimate'])})",
        f"MAC + non-MAC op estimate: "
        f"{report['totals']['macs'] + report['totals']['non_mac_ops_estimate']} "
        f"({human_count(report['totals']['macs'] + report['totals']['non_mac_ops_estimate'])})",
        "",
        "MACs by op type:",
    ]
    for op_name, value in sorted(report["totals"]["macs_by_op_type"].items()):
        lines.append(f"  {op_name:<20} {value:>14}  {human_count(value)}")
    lines.append("")
    lines.append("Non-MAC ops by op type:")
    for op_name, value in sorted(report["totals"]["non_mac_ops_by_op_type"].items()):
        lines.append(f"  {op_name:<20} {value:>14}  {human_count(value)}")
    lines.append("")
    lines.append("Per-op estimates:")
    for op in report["ops"]:
        if op["macs"] or op["non_mac_ops_estimate"]:
            lines.append(
                f"  {op['index']:03d} {op['op_name']:<22} "
                f"macs={op['macs']:>12} non_mac={op['non_mac_ops_estimate']:>10}"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    warnings.filterwarnings("ignore", message=r".*tf\.lite\.Interpreter is deprecated.*")
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
        tensors = {int(tensor["index"]): tensor for tensor in interpreter.get_tensor_details()}
        ops = [
            estimate_op(op, tensors, index)
            for index, op in enumerate(interpreter._get_ops_details())  # noqa: SLF001
        ]

    macs_by_op_type: dict[str, int] = {}
    non_mac_by_op_type: dict[str, int] = {}
    for op in ops:
        macs_by_op_type[op["op_name"]] = macs_by_op_type.get(op["op_name"], 0) + int(op["macs"])
        non_mac_by_op_type[op["op_name"]] = non_mac_by_op_type.get(op["op_name"], 0) + int(
            op["non_mac_ops_estimate"]
        )
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(args.model),
        "bytes": args.model.stat().st_size,
        "sha256": sha256(args.model),
        "method": {
            "mac_definition": "One multiply-accumulate counted as one MAC.",
            "counted_as_macs": ["CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED"],
            "not_counted_as_macs": [
                "elementwise ops",
                "reductions",
                "quantize/dequantize",
                "reshape/slice/transpose/pad",
            ],
            "limitations": [
                "Static TFLite graph estimate, not measured Edge TPU cycles.",
                "Padding/boundary effects are approximated from output tensor shape and kernel shape.",
            ],
        },
        "totals": {
            "macs": int(sum(op["macs"] for op in ops)),
            "non_mac_ops_estimate": int(sum(op["non_mac_ops_estimate"] for op in ops)),
            "macs_by_op_type": {key: int(value) for key, value in sorted(macs_by_op_type.items()) if value},
            "non_mac_ops_by_op_type": {
                key: int(value) for key, value in sorted(non_mac_by_op_type.items()) if value
            },
        },
        "ops": ops,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    if args.text_output is not None:
        args.text_output.parent.mkdir(parents=True, exist_ok=True)
        args.text_output.write_text(text_report(report), encoding="utf-8")
        print(f"Wrote {args.text_output}")
    if args.compiler_summary is not None:
        summary = json.loads(args.compiler_summary.read_text(encoding="utf-8"))
        summary["mac_estimate"] = {
            "summary_path": str(args.output),
            "total_macs": report["totals"]["macs"],
            "total_macs_human": human_count(report["totals"]["macs"]),
            "method": report["method"],
        }
        args.compiler_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"Updated {args.compiler_summary}")


if __name__ == "__main__":
    main()
