#!/usr/bin/env python3
"""Visualize Keras MobileNetV3 variants with VisualKeras."""

from __future__ import annotations

import argparse
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


DEFAULT_OUTPUT_DIR = Path("artifacts/reports/keras_mobilenetv3_minimalistic_visualkeras")
LAYER_COLORS = {
    "InputLayer": "#d9d9d9",
    "Rescaling": "#f2c94c",
    "Conv2D": "#2f80ed",
    "DepthwiseConv2D": "#eb5757",
    "BatchNormalization": "#6fcf97",
    "ReLU": "#bb6bd9",
    "Add": "#f2994a",
    "ZeroPadding2D": "#56ccf2",
    "GlobalAveragePooling2D": "#219653",
    "Dropout": "#828282",
    "Flatten": "#9b51e0",
    "Activation": "#27ae60",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        action="append",
        choices=["large", "small"],
        help="Variant to visualize. Repeat to select multiple. Defaults to large and small.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--minimalistic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-top", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--cutoff",
        default="none",
        help="Use 'none', 'auto', or a Keras layer name. Auto matches the Keras student code.",
    )
    parser.add_argument(
        "--include-preprocessing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the Keras Rescaling preprocessing layer.",
    )
    parser.add_argument(
        "--max-view-width",
        type=int,
        default=4096,
        help="Write an additional downscaled *_view.png if the render is wider than this.",
    )
    return parser.parse_args()


def layer_shape(value: Any) -> list[int | None] | str | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return [None if dim is None else int(dim) for dim in shape]
    except TypeError:
        return str(shape)


def layer_metadata(layer: Any) -> dict[str, Any]:
    config = layer.get_config()
    metadata: dict[str, Any] = {
        "name": layer.name,
        "class_name": layer.__class__.__name__,
        "output_shape": layer_shape(getattr(layer, "output", None)),
    }
    for key in (
        "filters",
        "kernel_size",
        "strides",
        "padding",
        "groups",
        "activation",
        "use_bias",
        "pool_size",
        "size",
        "target_shape",
        "units",
    ):
        if key in config:
            metadata[key] = config[key]
    return metadata


def max78000_notes(layers: list[dict[str, Any]]) -> dict[str, Any]:
    unsupported: list[dict[str, str]] = []
    cautions: list[dict[str, str]] = []
    for layer in layers:
        class_name = layer["class_name"]
        name = layer["name"]
        if class_name == "DepthwiseConv2D":
            unsupported.append(
                {
                    "layer": name,
                    "reason": (
                        "DepthwiseConv2D is central to Keras MobileNetV3, but the local ADI "
                        "ai8x.py tree restricts depthwise layers to MAX78002 rather than MAX78000."
                    ),
                }
            )
        elif class_name in {"Rescaling", "Dropout", "Softmax"}:
            cautions.append(
                {
                    "layer": name,
                    "reason": f"{class_name} should be moved out of the synthesized MAX78000 graph.",
                }
            )
        elif class_name == "Conv2D":
            strides = tuple(layer.get("strides") or ())
            if any(int(stride) > 1 for stride in strides):
                cautions.append(
                    {
                        "layer": name,
                        "reason": (
                            "Strided Conv2D should be replaced with pooling-based downsampling for "
                            "the conservative MAX78000 path."
                        ),
                    }
                )
            kernel = tuple(layer.get("kernel_size") or ())
            if any(int(size) not in {1, 3} for size in kernel):
                unsupported.append(
                    {
                        "layer": name,
                        "reason": f"Kernel {kernel} is outside the 1x1/3x3 constraint.",
                    }
                )
    return {
        "directly_max78000_compatible": not unsupported,
        "unsupported": unsupported,
        "cautions": cautions,
    }


def build_model(tf: Any, variant: str, args: argparse.Namespace) -> Any:
    include_top = bool(args.include_top) and str(args.cutoff) in {"", "none", "None"}
    common = {
        "input_shape": (args.input_size, args.input_size, 3),
        "alpha": args.alpha,
        "minimalistic": bool(args.minimalistic),
        "include_top": include_top,
        "weights": None,
        "include_preprocessing": args.include_preprocessing,
    }
    if include_top:
        common["classes"] = 1000
    if variant == "large":
        backbone = tf.keras.applications.MobileNetV3Large(**common)
    else:
        backbone = tf.keras.applications.MobileNetV3Small(**common)

    cutoff = resolve_cutoff(variant, str(args.cutoff))
    cutoff = resolve_existing_layer_name(backbone, cutoff)
    if cutoff is None:
        return backbone
    return tf.keras.Model(backbone.input, backbone.get_layer(cutoff).output, name=f"{backbone.name}_cutoff")


def resolve_cutoff(variant: str, cutoff: str) -> str | None:
    if cutoff in {"", "none", "None"}:
        return None
    if cutoff != "auto":
        return cutoff
    if variant == "large":
        return "expanded_conv_11/Add"
    return "expanded_conv_7/Add"


def resolve_existing_layer_name(model: Any, layer_name: str | None) -> str | None:
    if layer_name is None:
        return None
    names = {layer.name for layer in model.layers}
    if layer_name in names:
        return layer_name
    keras3_name = layer_name.replace("/Add", "_add")
    if keras3_name in names:
        return keras3_name
    raise ValueError(
        f"No such layer {layer_name!r}. Tried Keras 3 spelling {keras3_name!r}. "
        f"Available residual add layers: {[name for name in sorted(names) if name.endswith('_add')]}"
    )


def color_map_for_model(model: Any) -> dict[type, dict[str, str]]:
    color_map: dict[type, dict[str, str]] = {}
    for layer in model.layers:
        fill = LAYER_COLORS.get(layer.__class__.__name__)
        if fill is not None:
            color_map[layer.__class__] = {"fill": fill, "outline": "#222222"}
    return color_map


def write_legend(layer_counts: dict[str, int], output: Path) -> None:
    rows = [(name, count, LAYER_COLORS.get(name, "#cccccc")) for name, count in layer_counts.items()]
    row_height = 28
    width = 420
    height = 18 + row_height * len(rows)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default()
    except OSError:
        font = None
    y = 10
    for name, count, color in rows:
        draw.rectangle([10, y + 3, 28, y + 21], fill=color, outline="#222222")
        draw.text((38, y + 5), f"{name} ({count})", fill="black", font=font)
        y += row_height
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def maybe_write_view_image(source: Path, output: Path, max_width: int) -> str | None:
    image = Image.open(source)
    if image.width <= max_width:
        return None
    ratio = max_width / image.width
    resized = image.resize((max_width, max(1, int(image.height * ratio))), Image.Resampling.LANCZOS)
    if resized.mode != "RGB":
        resized = resized.convert("RGB")
    resized.save(output)
    return str(output)


def write_visualkeras(model: Any, output: Path, max_view_width: int) -> dict[str, Any]:
    import visualkeras

    output.parent.mkdir(parents=True, exist_ok=True)
    for layer in model.layers:
        if not hasattr(layer, "output_shape"):
            shape = layer_shape(getattr(layer, "output", None))
            if isinstance(shape, list):
                layer.output_shape = tuple(shape)
    visualkeras.layered_view(
        model,
        legend=True,
        to_file=str(output),
        min_z=8,
        min_xy=8,
        max_z=90,
        max_xy=180,
        scale_z=0.6,
        scale_xy=1.0,
        spacing=3,
        padding=6,
        color_map=color_map_for_model(model),
        background_fill="white",
    )
    view_path = output.with_name(f"{output.stem}_view.png")
    view = maybe_write_view_image(output, view_path, max_view_width)
    return {"status": "written", "path": str(output), "view_path": view}


def write_summary(model: Any, output: Path) -> None:
    buffer = io.StringIO()
    model.summary(print_fn=lambda line: buffer.write(line + "\n"))
    output.write_text(buffer.getvalue(), encoding="utf-8")


def main() -> None:
    args = parse_args()
    variants = args.variant or ["large", "small"]

    import tensorflow as tf

    args.output_dir.mkdir(parents=True, exist_ok=True)
    index: dict[str, Any] = {
        "input_size": args.input_size,
        "alpha": args.alpha,
        "minimalistic": bool(args.minimalistic),
        "include_top": args.include_top,
        "include_preprocessing": args.include_preprocessing,
        "cutoff": args.cutoff,
        "variants": {},
    }

    for variant in variants:
        model = build_model(tf, variant, args)
        requested_cutoff = resolve_cutoff(variant, str(args.cutoff))
        resolved_cutoff = resolve_existing_layer_name(model, requested_cutoff)
        cutoff_slug = "" if resolved_cutoff is None else f"_cutoff_{resolved_cutoff.replace('/', '_')}"
        slug = (
            f"mobilenet_v3_{variant}_{'minimalistic' if args.minimalistic else 'standard'}"
            f"_{args.alpha:g}_{args.input_size}"
            f"{'_top' if bool(args.include_top) and resolved_cutoff is None else '_notop'}"
            f"{cutoff_slug}"
        )
        png_path = args.output_dir / f"{slug}.png"
        legend_path = args.output_dir / f"{slug}_legend.png"
        summary_path = args.output_dir / f"{slug}_summary.txt"
        report_path = args.output_dir / f"{slug}_layers.json"

        layers = [layer_metadata(layer) for layer in model.layers]
        layer_counts = dict(Counter(layer["class_name"] for layer in layers))
        visualkeras_report = write_visualkeras(model, png_path, args.max_view_width)
        write_legend(layer_counts, legend_path)
        write_summary(model, summary_path)
        report = {
            "name": slug,
            "keras_model_name": model.name,
            "minimalistic": bool(args.minimalistic),
            "cutoff": resolved_cutoff,
            "layers": layers,
            "layer_class_counts": layer_counts,
            "max78000_notes": max78000_notes(layers),
            "visualkeras": visualkeras_report,
            "legend": str(legend_path),
            "summary": str(summary_path),
        }
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        index["variants"][variant] = {
            "png": str(png_path),
            "summary": str(summary_path),
            "layers": str(report_path),
            "legend": str(legend_path),
            "cutoff": resolved_cutoff,
            "layer_class_counts": report["layer_class_counts"],
            "max78000_directly_compatible": report["max78000_notes"][
                "directly_max78000_compatible"
            ],
        }
        print(f"Wrote {png_path}")
        print(f"Wrote {summary_path}")
        print(f"Wrote {report_path}")

    index_path = args.output_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {index_path}")


if __name__ == "__main__":
    main()
