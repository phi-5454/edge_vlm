#!/usr/bin/env python3
"""Write readable reports for the staged MAX78000 TallyQA count model."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch


DEFAULT_MODEL_FILE = Path("max78000/ai8x_training/models/ai85net-tallyqa-mbv3-small.py")
DEFAULT_AI8X_TRAINING = Path("../MAX78000/ai8x-training")
DEFAULT_OUTPUT_DIR = Path("artifacts/reports/max78000/model_reports/tallyqa_count")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-file", type=Path, default=DEFAULT_MODEL_FILE)
    parser.add_argument("--ai8x-training", type=Path, default=DEFAULT_AI8X_TRAINING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--factory", default="ai85tallyqambv3smallcount")
    parser.add_argument("--device", type=int, default=85)
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--round-avg", action="store_true")
    parser.add_argument("--input-channels", type=int, default=12)
    parser.add_argument("--prompt-embedding-channels", type=int, default=0)
    parser.add_argument("--input-size", type=int, default=56)
    parser.add_argument("--num-classes", type=int, default=5)
    return parser.parse_args()


def format_count(value: int | float) -> str:
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(int(value))


def load_model_module(model_file: Path, ai8x_training: Path):
    sys.path.insert(0, str((ai8x_training).resolve()))
    sys.path.insert(0, str((ai8x_training / "models").resolve()))
    spec = importlib.util.spec_from_file_location("vlm_micro_max78000_tallyqa_model", model_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {model_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parameter_rows(model: torch.nn.Module) -> list[dict[str, Any]]:
    rows = []
    for name, parameter in model.named_parameters():
        top_level = name.split(".", 1)[0]
        rows.append(
            {
                "name": name,
                "top_level": top_level,
                "shape": list(parameter.shape),
                "parameters": int(parameter.numel()),
                "trainable": bool(parameter.requires_grad),
            }
        )
    return rows


def top_level_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        group = grouped.setdefault(
            row["top_level"],
            {"module": row["top_level"], "total": 0, "trainable": 0, "frozen": 0},
        )
        group["total"] += row["parameters"]
        if row["trainable"]:
            group["trainable"] += row["parameters"]
        else:
            group["frozen"] += row["parameters"]
    return sorted(grouped.values(), key=lambda item: item["total"], reverse=True)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def shape_text(shape: Any) -> str:
    if isinstance(shape, list) and shape and all(isinstance(item, list) for item in shape):
        return " + ".join("x".join(str(value) for value in item) for item in shape)
    if isinstance(shape, list):
        return "x".join(str(value) for value in shape)
    return str(shape)


def write_parameter_chart(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        return
    labels = [row["module"] for row in rows]
    trainable = [row["trainable"] for row in rows]
    frozen = [row["frozen"] for row in rows]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * len(rows) + 1.2)))
    y_positions = list(range(len(rows)))
    ax.barh(y_positions, frozen, label="frozen", color="#9aa4b2")
    ax.barh(y_positions, trainable, left=frozen, label="trainable", color="#2f80ed")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Parameters")
    ax.set_title("MAX78000 Model Parameters")
    ax.xaxis.set_major_formatter(lambda value, _pos: format_count(value))
    ax.legend(loc="lower right")
    for y_pos, row in zip(y_positions, rows, strict=True):
        ax.text(row["total"], y_pos, f" {format_count(row['total'])}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def tensor_shape_for(module: torch.nn.Module, sample: torch.Tensor | tuple[torch.Tensor, ...]) -> list[int] | None:
    module.eval()
    with torch.no_grad():
        try:
            return list(module(sample).shape)
        except Exception:
            return None


def main() -> None:
    args = parse_args()
    if not args.model_file.exists():
        raise FileNotFoundError(args.model_file)
    if not args.ai8x_training.exists():
        raise FileNotFoundError(args.ai8x_training)

    module = load_model_module(args.model_file.resolve(), args.ai8x_training.resolve())
    if "ai8x" in sys.modules:
        sys.modules["ai8x"].set_device(
            args.device,
            simulate=bool(args.simulate),
            round_avg=bool(args.round_avg),
            verbose=False,
        )
    factory = getattr(module, args.factory)
    model = factory(
        num_classes=args.num_classes,
        num_channels=args.input_channels,
        dimensions=(args.input_size, args.input_size),
        bias=True,
    )
    model.eval()
    image_sample = torch.zeros((1, args.input_channels, args.input_size, args.input_size))
    if args.prompt_embedding_channels > 0:
        sample: torch.Tensor | tuple[torch.Tensor, torch.Tensor] = (
            image_sample,
            torch.zeros((1, args.prompt_embedding_channels)),
        )
        input_shape: list[Any] = [
            [1, args.input_channels, args.input_size, args.input_size],
            [1, args.prompt_embedding_channels],
        ]
    else:
        sample = image_sample
        input_shape = [1, args.input_channels, args.input_size, args.input_size]
    with torch.no_grad():
        output_shape = list(model(sample).shape)
        feature_shape = (
            list(model.forward_features(sample).shape)
            if hasattr(model, "forward_features")
            else None
        )
    rows = parameter_rows(model)
    grouped_rows = top_level_rows(rows)
    total = sum(row["parameters"] for row in rows)
    trainable = sum(row["parameters"] for row in rows if row["trainable"])
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_file": str(args.model_file),
        "ai8x_training": str(args.ai8x_training),
        "factory": args.factory,
        "device": args.device,
        "simulate": bool(args.simulate),
        "round_avg": bool(args.round_avg),
        "input_shape": input_shape,
        "image_input_shape": [1, args.input_channels, args.input_size, args.input_size],
        "prompt_embedding_shape": (
            [1, args.prompt_embedding_channels]
            if args.prompt_embedding_channels > 0
            else None
        ),
        "feature_shape": feature_shape,
        "output_shape": output_shape,
        "parameter_counts": {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
        },
        "top_level_modules": grouped_rows,
        "parameters": rows,
        "module_tree": str(model),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "architecture.json"
    markdown_path = args.output_dir / "architecture_readable.md"
    text_path = args.output_dir / "architecture_tree.txt"
    chart_path = args.output_dir / "architecture_parameters.png"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    text_path.write_text(str(model) + "\n", encoding="utf-8")
    write_parameter_chart(grouped_rows, chart_path)

    overview_rows = [
        ["input", shape_text(report["input_shape"])],
        ["feature cut", "x".join(str(value) for value in feature_shape or [])],
        ["output", "x".join(str(value) for value in output_shape)],
        ["parameters", f"{format_count(total)} total / {format_count(trainable)} trainable"],
    ]
    module_rows = [
        [
            str(row["module"]),
            format_count(row["total"]),
            format_count(row["trainable"]),
            format_count(row["frozen"]),
        ]
        for row in grouped_rows
    ]
    markdown = "\n\n".join(
        [
            "# MAX78000 TallyQA Count Model Report",
            "## Overview",
            markdown_table(["item", "value"], overview_rows),
            "## Top-Level Modules",
            markdown_table(["module", "total", "trainable", "frozen"], module_rows),
            "## Module Tree",
            f"```text\n{model}\n```",
        ]
    )
    markdown_path.write_text(markdown + "\n", encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")
    print(f"Wrote {chart_path}")


if __name__ == "__main__":
    main()
