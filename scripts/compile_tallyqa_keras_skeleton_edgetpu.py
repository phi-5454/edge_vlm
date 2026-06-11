#!/usr/bin/env python3
"""Export and optionally Edge-TPU-compile an untrained Keras TallyQA skeleton."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from omegaconf import OmegaConf

from scripts.train_tallyqa_keras_student import (
    apply_feature_film_tokens,
    broadcast_query_to_feature_map,
    build_keras_student_model,
    build_keras_mobilenet,
    static_normformer_block,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the current Keras TallyQA student with synthetic prompt embeddings, "
            "export int8 TFLite, and run edgetpu_compiler when available."
        )
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-root", default="artifacts/reports/coral/edgetpu_compiler")
    parser.add_argument("--tflite-root", default="artifacts/exports/coral/skeletons")
    parser.add_argument(
        "--skeleton-kind",
        choices=["current_student", "fusion_only"],
        default="current_student",
        help=(
            "current_student includes prompt lookup and MobileNet. fusion_only feeds "
            "fixed prompt/image tokens directly to isolate fusion-layer compiler support."
        ),
    )
    parser.add_argument(
        "--fusion-mode",
        choices=["normformer", "mlp", "film_mlp", "prompt_patch_mlp"],
        default="normformer",
    )
    parser.add_argument(
        "--image-film-at",
        choices=["none", "image_tokens", "token_projection"],
        default="none",
        help=(
            "Prompt-FiLM location for current_student/film_mlp. The Keras path "
            "supports FiLM on the projected visual token map; PyTorch-only "
            "pre-depthwise MobileNet FiLM is not represented in this skeleton."
        ),
    )
    parser.add_argument(
        "--attention-impl",
        choices=["keras", "static"],
        default="static",
        help=(
            "Attention implementation for normformer. 'static' avoids Keras "
            "MultiHeadAttention dynamic shape plumbing for Edge TPU compiler probes."
        ),
    )
    parser.add_argument(
        "--attention-normalization",
        choices=["softmax", "none"],
        default="softmax",
        help="Use 'none' to drop softmax and probe whether BATCH_MATMUL remains the blocker.",
    )
    parser.add_argument("--image-backbone", choices=["mobilenet_v3_small", "mobilenet_v3_large"], default="mobilenet_v3_small")
    parser.add_argument("--image-feature-cutoff", default="auto")
    parser.add_argument(
        "--mobilenet-minimalistic",
        action="store_true",
        help="Use Keras MobileNetV3 minimalistic blocks with ReLU-style activations for compiler probing.",
    )
    parser.add_argument("--fusion-dim", type=int, default=128)
    parser.add_argument("--fusion-depth", type=int, default=4)
    parser.add_argument("--fusion-heads", type=int, default=4)
    parser.add_argument("--fusion-mlp-ratio", type=int, default=4)
    parser.add_argument(
        "--activation",
        default="relu",
        choices=["relu", "gelu", "hard_swish", "swish", "linear"],
        help=(
            "Activation in prompt/fusion MLPs. Use relu for Edge TPU compiler "
            "probing; GELU can emit newer TFLite builtin ops that older compilers "
            "cannot parse."
        ),
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--prompt-length", type=int, default=1)
    parser.add_argument("--prompt-vocab-size", type=int, default=64)
    parser.add_argument("--prompt-embedding-dim", type=int, default=384)
    parser.add_argument("--num-outputs", type=int, default=6)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--representative-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained-mobilenet", action="store_true")
    parser.add_argument(
        "--fold-static-prompt",
        action="store_true",
        help=(
            "For prompt_patch_mlp, compile the deployable per-prompt graph with "
            "prompt conditioning folded into the first patch Conv2D. This removes "
            "runtime prompt lookup/broadcast ops such as TILE."
        ),
    )
    parser.add_argument("--keep-float", action="store_true", help="Also export a float TFLite model.")
    parser.add_argument("--skip-compile", action="store_true", help="Only export TFLite; do not call edgetpu_compiler.")
    parser.add_argument(
        "--skip-visualkeras",
        action="store_true",
        help="Do not write a VisualKeras architecture PNG next to the compiler report.",
    )
    parser.add_argument("--compiler-bin", default="edgetpu_compiler")
    parser.add_argument("--compiler-extra-arg", action="append", default=[])
    parser.add_argument(
        "--compiler-container-image",
        default=None,
        help=(
            "Docker image containing edgetpu_compiler. When set, compilation runs "
            "inside the container with this repository mounted at /workspace."
        ),
    )
    parser.add_argument(
        "--docker-bin",
        default="docker",
        help="Docker-compatible executable to use with --compiler-container-image.",
    )
    parser.add_argument(
        "--docker-user",
        default="host",
        help=(
            "User for docker run. Use 'host' to pass the current uid:gid, 'root' to omit "
            "--user, or an explicit value accepted by docker."
        ),
    )
    return parser.parse_args()


def maybe_write_visualkeras(model: tf.keras.Model, output: Path) -> dict[str, str | bool]:
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        import visualkeras
    except ImportError:
        return {
            "enabled": True,
            "status": "missing_dependency",
            "message": "Install visualkeras to generate model architecture PNGs.",
            "output": str(output),
        }
    try:
        visualkeras.layered_view(model, legend=True, to_file=str(output))
    except Exception as exc:
        return {
            "enabled": True,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "output": str(output),
        }
    return {"enabled": True, "status": "written", "output": str(output)}


def skeleton_cfg(args: argparse.Namespace):
    image_film_at = args.image_film_at
    if image_film_at == "none" and args.fusion_mode == "film_mlp":
        image_film_at = "image_tokens"
    return OmegaConf.create(
        {
            "model": {
                "num_outputs": args.num_outputs,
                "fusion_dim": args.fusion_dim,
                "fusion_depth": args.fusion_depth,
                "fusion_heads": args.fusion_heads,
                "fusion_mlp_ratio": args.fusion_mlp_ratio,
                "activation": args.activation,
                "dropout": args.dropout,
                "freeze_embeddings": True,
                "freeze_image_features": True,
                "image_pretrained": bool(args.pretrained_mobilenet),
                "image_backbone": args.image_backbone,
                "image_feature_cutoff": args.image_feature_cutoff,
                "use_prompt_identity": True,
                "use_image_positional_embeddings": True,
            },
            "keras_model": {
                "architecture": "current_student",
                "image_size": args.image_size,
                "batch_size": 1,
                "image_backbone": args.image_backbone,
                "image_feature_cutoff": args.image_feature_cutoff,
                "mobilenet_minimalistic": bool(args.mobilenet_minimalistic),
                "include_mobilenet_preprocessing": False,
                "fusion_mode": args.fusion_mode,
                "image_film_at": None if image_film_at == "none" else image_film_at,
                "attention_impl": args.attention_impl,
                "attention_normalization": args.attention_normalization,
                "fusion_dim": args.fusion_dim,
                "fusion_depth": args.fusion_depth,
                "fusion_heads": args.fusion_heads,
                "fusion_mlp_ratio": args.fusion_mlp_ratio,
                "activation": args.activation,
                "dropout": args.dropout,
                "use_prompt_identity": True,
                "use_image_positional_embeddings": True,
                "mask_zero_prompt_embeddings": False,
                "static_single_prompt_token": args.prompt_length == 1,
            },
        }
    )


def representative_dataset(
    rng: np.random.Generator,
    samples: int,
    prompt_length: int,
    prompt_vocab_size: int,
    image_size: int,
) -> Iterable[list[np.ndarray]]:
    for _ in range(samples):
        token_ids = rng.integers(1, prompt_vocab_size + 1, size=(1, prompt_length), dtype=np.int32)
        images = rng.uniform(-1.0, 1.0, size=(1, image_size, image_size, 3)).astype(np.float32)
        yield [token_ids, images]


def representative_image_dataset(
    rng: np.random.Generator,
    samples: int,
    image_size: int,
) -> Iterable[list[np.ndarray]]:
    for _ in range(samples):
        images = rng.uniform(-1.0, 1.0, size=(1, image_size, image_size, 3)).astype(np.float32)
        yield [images]


def representative_fusion_dataset(
    rng: np.random.Generator,
    samples: int,
    token_count: int,
    fusion_dim: int,
) -> Iterable[list[np.ndarray]]:
    for _ in range(samples):
        prompt = rng.normal(0.0, 1.0, size=(1, 1, fusion_dim)).astype(np.float32)
        image = rng.normal(0.0, 1.0, size=(1, token_count, fusion_dim)).astype(np.float32)
        yield [image, prompt]


def build_fusion_only_model(args: argparse.Namespace) -> tf.keras.Model:
    token_count = 14 * 14
    image_tokens = tf.keras.Input(
        shape=(token_count, args.fusion_dim),
        batch_size=1,
        dtype=tf.float32,
        name="image_tokens",
    )
    prompt_token = tf.keras.Input(
        shape=(1, args.fusion_dim),
        batch_size=1,
        dtype=tf.float32,
        name="prompt_token",
    )

    if args.fusion_mode == "mlp":
        image = tf.keras.layers.GlobalAveragePooling1D(name="image_token_mean")(image_tokens)
        prompt = tf.keras.layers.Lambda(lambda x: x[:, 0, :], name="prompt_token_flat")(prompt_token)
        fused = tf.keras.layers.Concatenate(name="prompt_image_concat")([prompt, image])
        fused = tf.keras.layers.Dense(
            args.fusion_dim * args.fusion_mlp_ratio,
            activation=args.activation,
            name="fusion_mlp_0",
        )(fused)
        fused = tf.keras.layers.Dense(args.fusion_dim, activation=args.activation, name="fusion_mlp_1")(
            fused
        )
    elif args.fusion_mode == "film_mlp":
        conditioned = apply_feature_film_tokens(
            image_tokens,
            tf.keras.layers.Lambda(lambda x: x[:, 0, :], name="prompt_token_flat")(prompt_token),
            args.fusion_dim,
            name="image_token_film",
        )
        image = tf.keras.layers.GlobalAveragePooling1D(name="image_token_mean")(conditioned)
        fused = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="fusion_mlp_input_norm")(image)
        fused = tf.keras.layers.Dense(
            args.fusion_dim * args.fusion_mlp_ratio,
            activation=args.activation,
            name="fusion_mlp_0",
        )(fused)
        fused = tf.keras.layers.Dense(args.fusion_dim, activation=args.activation, name="fusion_mlp_1")(
            fused
        )
        fused = tf.keras.layers.Dense(args.fusion_dim, activation=args.activation, name="fusion_mlp_2")(
            fused
        )
    elif args.fusion_mode == "prompt_patch_mlp":
        image_map = tf.keras.layers.Reshape(
            (14, 14, args.fusion_dim),
            name="image_token_map",
        )(image_tokens)
        prompt = tf.keras.layers.Lambda(lambda x: x[:, 0, :], name="prompt_token_flat")(
            prompt_token
        )
        prompt_map = broadcast_query_to_feature_map(
            prompt,
            image_map,
            args.fusion_dim,
            name="prompt_patch_query",
        )
        conditioned = tf.keras.layers.Concatenate(axis=-1, name="prompt_patch_concat")(
            [image_map, prompt_map]
        )
        conditioned = tf.keras.layers.Conv2D(
            args.fusion_dim * args.fusion_mlp_ratio,
            kernel_size=1,
            padding="same",
            activation=args.activation,
            name="prompt_patch_conv1x1",
        )(conditioned)
        conditioned = tf.keras.layers.Conv2D(
            128,
            kernel_size=3,
            padding="same",
            activation=args.activation,
            name="prompt_patch_conv3x3",
        )(conditioned)
        fused = tf.keras.layers.GlobalAveragePooling2D(name="prompt_patch_mean_pool")(
            conditioned
        )
    else:
        tokens = tf.keras.layers.Concatenate(axis=1, name="prompt_image_tokens")(
            [prompt_token, image_tokens]
        )
        for index in range(args.fusion_depth):
            tokens = static_normformer_block(
                tokens,
                token_count + 1,
                args.fusion_dim,
                args.fusion_heads,
                args.fusion_mlp_ratio,
                args.dropout,
                args.activation,
                args.attention_normalization,
                index,
            )
        fused = tf.keras.layers.GlobalAveragePooling1D(name="fusion_token_mean")(tokens)
        fused = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="fusion_output_norm")(fused)

    logits = tf.keras.layers.Dense(args.num_outputs, name="logits")(fused)
    return tf.keras.Model(
        inputs={"image_tokens": image_tokens, "prompt_token": prompt_token},
        outputs=logits,
        name=f"tallyqa_fusion_only_{args.fusion_mode}",
    )


def build_static_prompt_patch_model(args: argparse.Namespace) -> tf.keras.Model:
    if args.fusion_mode != "prompt_patch_mlp":
        raise ValueError("--fold-static-prompt is currently only valid for prompt_patch_mlp.")

    images = tf.keras.Input(
        shape=(args.image_size, args.image_size, 3),
        batch_size=1,
        dtype=tf.float32,
        name="images",
    )
    cfg = skeleton_cfg(args)
    backbone = build_keras_mobilenet(cfg, images)
    backbone.trainable = False
    image_features = backbone(images)
    x = tf.keras.layers.Conv2D(
        args.fusion_dim * args.fusion_mlp_ratio,
        kernel_size=1,
        padding="same",
        activation=args.activation,
        name="prompt_folded_patch_conv1x1",
    )(image_features)
    x = tf.keras.layers.Conv2D(
        128,
        kernel_size=3,
        padding="same",
        activation=args.activation,
        name="prompt_patch_conv3x3",
    )(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="prompt_patch_mean_pool")(x)
    logits = tf.keras.layers.Dense(args.num_outputs, name="logits")(x)
    return tf.keras.Model(
        inputs={"images": images},
        outputs=logits,
        name="tallyqa_static_prompt_patch_mlp_student",
    )


def export_float_tflite(model: tf.keras.Model, output: Path) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    output.write_bytes(converter.convert())


def export_int8_tflite(model: tf.keras.Model, output: Path, args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed + 1)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if args.skeleton_kind == "fusion_only":
        converter.representative_dataset = lambda: representative_fusion_dataset(
            rng,
            args.representative_samples,
            14 * 14,
            args.fusion_dim,
        )
    elif args.fold_static_prompt:
        converter.representative_dataset = lambda: representative_image_dataset(
            rng,
            args.representative_samples,
            args.image_size,
        )
    else:
        converter.representative_dataset = lambda: representative_dataset(
            rng,
            args.representative_samples,
            args.prompt_length,
            args.prompt_vocab_size,
            args.image_size,
        )
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.int8
    output.write_bytes(converter.convert())


def inspect_tflite(path: Path) -> dict:
    interpreter = tf.lite.Interpreter(
        model_path=str(path),
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
        experimental_preserve_all_tensors=True,
        experimental_delegates=[],
    )
    interpreter.allocate_tensors()

    def quantization_tuple(item: dict) -> list[float | int]:
        scale, zero_point = item["quantization"]
        return [float(scale), int(zero_point)]

    return {
        "inputs": [
            {
                "name": item["name"],
                "shape": item["shape"].tolist(),
                "dtype": str(item["dtype"]),
                "quantization": quantization_tuple(item),
            }
            for item in interpreter.get_input_details()
        ],
        "outputs": [
            {
                "name": item["name"],
                "shape": item["shape"].tolist(),
                "dtype": str(item["dtype"]),
                "quantization": quantization_tuple(item),
            }
            for item in interpreter.get_output_details()
        ],
        "operator_count": len(interpreter._get_ops_details()),  # noqa: SLF001
        "operators": [
            {
                "index": index,
                "op_name": item.get("op_name"),
                "inputs": [int(value) for value in item.get("inputs", [])],
                "outputs": [int(value) for value in item.get("outputs", [])],
            }
            for index, item in enumerate(interpreter._get_ops_details())  # noqa: SLF001
        ],
    }


def run_compiler(args: argparse.Namespace, tflite_path: Path, report_dir: Path) -> dict:
    if args.compiler_container_image:
        return run_compiler_in_container(args, tflite_path, report_dir)

    compiler = find_compiler(args.compiler_bin)
    if compiler is None:
        return {
            "status": "skipped",
            "reason": (
                f"{args.compiler_bin!r} not found on PATH or under common sibling Coral "
                "SDK/tool directories"
            ),
        }
    cmd = [compiler, "-o", str(report_dir), str(tflite_path), *args.compiler_extra_arg]
    result = subprocess.run(cmd, text=True, capture_output=True, check=False)
    (report_dir / "edgetpu_compiler.stdout.txt").write_text(result.stdout, encoding="utf-8")
    (report_dir / "edgetpu_compiler.stderr.txt").write_text(result.stderr, encoding="utf-8")
    return {
        "status": "ok" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "command": cmd,
    }


def run_compiler_in_container(args: argparse.Namespace, tflite_path: Path, report_dir: Path) -> dict:
    docker = shutil.which(args.docker_bin)
    if docker is None:
        return {
            "status": "skipped",
            "reason": f"{args.docker_bin!r} not found on PATH",
        }

    repo_root = Path.cwd().resolve()
    tflite_abs = tflite_path.resolve()
    report_abs = report_dir.resolve()
    try:
        tflite_container = Path("/workspace") / tflite_abs.relative_to(repo_root)
        report_container = Path("/workspace") / report_abs.relative_to(repo_root)
    except ValueError as exc:
        raise SystemExit(
            "Container compilation expects tflite/report paths under the repository root."
        ) from exc

    user_args: list[str] = []
    if args.docker_user == "host":
        user_args = ["--user", f"{os.getuid()}:{os.getgid()}"]
    elif args.docker_user != "root":
        user_args = ["--user", args.docker_user]

    cmd = [
        docker,
        "run",
        "--rm",
        *user_args,
        "-v",
        f"{repo_root}:/workspace",
        "-w",
        "/workspace",
        args.compiler_container_image,
        args.compiler_bin,
        "-o",
        str(report_container),
        str(tflite_container),
        *args.compiler_extra_arg,
    ]
    result = subprocess.run(cmd, text=True, capture_output=True, check=False)
    (report_dir / "edgetpu_compiler.docker.stdout.txt").write_text(result.stdout, encoding="utf-8")
    (report_dir / "edgetpu_compiler.docker.stderr.txt").write_text(result.stderr, encoding="utf-8")
    return {
        "status": "ok" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "command": cmd,
        "container_image": args.compiler_container_image,
    }


def find_compiler(compiler_bin: str) -> str | None:
    compiler_path = Path(compiler_bin)
    if compiler_path.exists():
        return str(compiler_path)

    compiler = shutil.which(compiler_bin)
    if compiler is not None:
        return compiler

    search_roots = [
        Path("../coral"),
        Path("../coralmicro"),
        Path("../coral/edgetpu_compiler"),
        Path("../coralmicro/edgetpu_compiler"),
        Path("../coral/tools"),
        Path("../coralmicro/tools"),
        Path("../coral/bin"),
        Path("../coralmicro/bin"),
    ]
    for root in search_roots:
        if not root.exists():
            continue
        direct = root / compiler_bin
        if direct.exists():
            return str(direct)
        matches = sorted(root.rglob(compiler_bin))
        if matches:
            return str(matches[0])
    return None


def main() -> None:
    args = parse_args()
    if args.fusion_dim % args.fusion_heads != 0:
        raise SystemExit("--fusion-dim must be divisible by --fusion-heads")

    run_name = args.run_name or (
        f"tallyqa_skeleton_{args.image_backbone}_{args.fusion_mode}"
        f"_d{args.fusion_dim}_l{args.fusion_depth}_h{args.fusion_heads}_m{args.fusion_mlp_ratio}"
    )
    report_dir = Path(args.output_root) / run_name / "ptq"
    tflite_dir = Path(args.tflite_root) / run_name
    report_dir.mkdir(parents=True, exist_ok=True)
    tflite_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    embedding_rows = rng.normal(
        loc=0.0,
        scale=0.02,
        size=(args.prompt_vocab_size, args.prompt_embedding_dim),
    ).astype(np.float32)

    tf.keras.utils.set_random_seed(args.seed)
    if args.skeleton_kind == "fusion_only":
        model = build_fusion_only_model(args)
    elif args.fold_static_prompt:
        model = build_static_prompt_patch_model(args)
    else:
        model = build_keras_student_model(skeleton_cfg(args), embedding_rows, args.prompt_length)
    model_summary: list[str] = []
    model.summary(print_fn=model_summary.append)
    (report_dir / "keras_model_summary.txt").write_text("\n".join(model_summary) + "\n", encoding="utf-8")
    visualkeras_report = (
        {"enabled": False, "status": "skipped", "reason": "--skip-visualkeras was set"}
        if args.skip_visualkeras
        else maybe_write_visualkeras(model, report_dir / "model_architecture_visualkeras.png")
    )

    if args.keep_float:
        export_float_tflite(model, tflite_dir / "model_float.tflite")

    quantized_path = tflite_dir / "model_int8.tflite"
    export_int8_tflite(model, quantized_path, args)
    shutil.copy2(quantized_path, report_dir / quantized_path.name)

    inspection = inspect_tflite(quantized_path)
    compiler = {"status": "skipped", "reason": "--skip-compile was set"}
    if not args.skip_compile:
        compiler = run_compiler(args, quantized_path, report_dir)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_name": run_name,
        "config": vars(args),
        "model": {
            "params": int(model.count_params()),
            "inputs": [item.name for item in model.inputs],
            "outputs": [item.name for item in model.outputs],
        },
        "artifacts": {
            "report_dir": str(report_dir),
            "tflite_dir": str(tflite_dir),
            "quantized_tflite": str(quantized_path),
            "visualkeras": visualkeras_report,
        },
        "tflite_inspection": inspection,
        "edgetpu_compiler": compiler,
    }
    (report_dir / "compiler_summary.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
