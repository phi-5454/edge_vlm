#!/usr/bin/env python3
"""Export a tiny two-input raw-prompt TFLite model for Coral app testing."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf


DEFAULT_PROMPT_LOOKUP = Path("artifacts/exports/coral/prompt_embedding_lookup")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="dummy_raw_prompt_embedding_app_test")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/reports/coral/app_test_dummy"))
    parser.add_argument("--tflite-root", type=Path, default=Path("artifacts/exports/coral/app_test_dummy"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--prompt-dim", type=int, default=None)
    parser.add_argument("--num-outputs", type=int, default=6)
    parser.add_argument("--representative-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--prompt-lookup-dir",
        type=Path,
        default=DEFAULT_PROMPT_LOOKUP,
        help="Use the existing quantized prompt lookup to infer prompt dim and calibration range.",
    )
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--compiler-container-image", default="edge-vlm-edgetpu-compiler:latest")
    parser.add_argument("--compiler-bin", default="edgetpu_compiler")
    parser.add_argument("--docker-bin", default="docker")
    parser.add_argument("--docker-user", default="host")
    return parser.parse_args()


def read_prompt_lookup(prompt_lookup_dir: Path) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    npy_path = prompt_lookup_dir / "prompt_embedding_lookup_uint8.npy"
    manifest_path = prompt_lookup_dir / "prompt_embedding_lookup_manifest.json"
    if not npy_path.exists() or not manifest_path.exists():
        return None, None
    table = np.load(npy_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return table, manifest


def prompt_representative_rows(
    rng: np.random.Generator,
    samples: int,
    prompt_dim: int,
    prompt_table: np.ndarray | None,
    prompt_manifest: dict[str, Any] | None,
) -> Iterable[np.ndarray]:
    if prompt_table is not None and prompt_manifest is not None:
        quant = prompt_manifest.get("quantization", {})
        scale = float(quant.get("scale", 1.0))
        zero_point = float(quant.get("zero_point", 0.0))
        rows = (prompt_table.astype(np.float32) - zero_point) * scale
        for _ in range(samples):
            yield rows[int(rng.integers(0, rows.shape[0]))][np.newaxis, :]
        return

    for _ in range(samples):
        yield rng.normal(0.0, 0.25, size=(1, prompt_dim)).astype(np.float32)


def representative_dataset(
    rng: np.random.Generator,
    samples: int,
    image_size: int,
    prompt_dim: int,
    prompt_table: np.ndarray | None,
    prompt_manifest: dict[str, Any] | None,
) -> Iterable[dict[str, np.ndarray]]:
    prompt_rows = prompt_representative_rows(
        rng,
        samples,
        prompt_dim,
        prompt_table,
        prompt_manifest,
    )
    for prompt_embedding in prompt_rows:
        images = rng.uniform(
            -1.0,
            1.0,
            size=(1, image_size, image_size, 3),
        ).astype(np.float32)
        yield {"images": images, "prompt_embedding": prompt_embedding.astype(np.float32)}


def build_model(image_size: int, prompt_dim: int, num_outputs: int) -> tf.keras.Model:
    images = tf.keras.Input(
        shape=(image_size, image_size, 3),
        batch_size=1,
        dtype=tf.float32,
        name="images",
    )
    prompt_embedding = tf.keras.Input(
        shape=(prompt_dim,),
        batch_size=1,
        dtype=tf.float32,
        name="prompt_embedding",
    )

    x = tf.keras.layers.Conv2D(8, 3, strides=2, padding="same", activation="relu", name="image_conv0")(
        images
    )
    x = tf.keras.layers.DepthwiseConv2D(
        3,
        strides=2,
        padding="same",
        activation="relu",
        name="image_dw0",
    )(x)
    x = tf.keras.layers.Conv2D(16, 1, padding="same", activation="relu", name="image_conv1")(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="image_mean")(x)

    p = tf.keras.layers.Dense(16, activation="relu", name="prompt_fc0")(prompt_embedding)
    fused = tf.keras.layers.Add(name="image_prompt_add")([x, p])
    fused = tf.keras.layers.Dense(16, activation="relu", name="fusion_fc0")(fused)
    logits = tf.keras.layers.Dense(num_outputs, name="logits")(fused)
    return tf.keras.Model(
        inputs={"images": images, "prompt_embedding": prompt_embedding},
        outputs=logits,
        name="dummy_raw_prompt_embedding_app_test",
    )


def export_int8_tflite(
    model: tf.keras.Model,
    output: Path,
    args: argparse.Namespace,
    prompt_table: np.ndarray | None,
    prompt_manifest: dict[str, Any] | None,
) -> None:
    rng = np.random.default_rng(args.seed + 1)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(
        rng,
        args.representative_samples,
        args.image_size,
        args.prompt_dim,
        prompt_table,
        prompt_manifest,
    )
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.int8
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(converter.convert())


def clean_tensor(item: dict[str, Any]) -> dict[str, Any]:
    scale, zero_point = item.get("quantization", (0.0, 0))
    return {
        "name": item.get("name"),
        "index": int(item["index"]),
        "shape": [int(value) for value in item["shape"]],
        "dtype": str(item["dtype"]),
        "quantization": [float(scale), int(zero_point)],
    }


def inspect_tflite(path: Path) -> dict[str, Any]:
    interpreter = tf.lite.Interpreter(
        model_path=str(path),
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
        experimental_delegates=[],
    )
    interpreter.allocate_tensors()
    ops = interpreter._get_ops_details()  # noqa: SLF001
    return {
        "inputs": [clean_tensor(item) for item in interpreter.get_input_details()],
        "outputs": [clean_tensor(item) for item in interpreter.get_output_details()],
        "operator_count": len(ops),
        "operators": [item.get("op_name") for item in ops],
    }


def run_compiler(args: argparse.Namespace, tflite_path: Path, report_dir: Path) -> dict[str, Any]:
    docker = shutil.which(args.docker_bin)
    if docker is None:
        return {"status": "skipped", "reason": f"{args.docker_bin!r} not found on PATH"}

    repo_root = Path.cwd().resolve()
    tflite_abs = tflite_path.resolve()
    report_abs = report_dir.resolve()
    tflite_container = Path("/workspace") / tflite_abs.relative_to(repo_root)
    report_container = Path("/workspace") / report_abs.relative_to(repo_root)

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
    ]
    result = subprocess.run(cmd, text=True, capture_output=True, check=False)
    (report_dir / "edgetpu_compiler.docker.stdout.txt").write_text(result.stdout, encoding="utf-8")
    (report_dir / "edgetpu_compiler.docker.stderr.txt").write_text(result.stderr, encoding="utf-8")
    return {
        "status": "ok" if result.returncode == 0 else "failed",
        "returncode": int(result.returncode),
        "command": cmd,
        "container_image": args.compiler_container_image,
    }


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    prompt_table, prompt_manifest = read_prompt_lookup(args.prompt_lookup_dir)
    if args.prompt_dim is None:
        if prompt_table is None:
            raise SystemExit("--prompt-dim is required when no prompt lookup table exists.")
        args.prompt_dim = int(prompt_table.shape[1])
    if prompt_table is not None and int(prompt_table.shape[1]) != int(args.prompt_dim):
        raise SystemExit(
            f"Prompt lookup dim {prompt_table.shape[1]} does not match --prompt-dim {args.prompt_dim}."
        )

    tf.keras.utils.set_random_seed(args.seed)
    report_dir = args.output_root / args.run_name
    tflite_dir = args.tflite_root / args.run_name
    report_dir.mkdir(parents=True, exist_ok=True)
    tflite_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(args.image_size, int(args.prompt_dim), args.num_outputs)
    summary: list[str] = []
    model.summary(print_fn=summary.append)
    (report_dir / "keras_model_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")

    int8_path = tflite_dir / "model_int8.tflite"
    export_int8_tflite(model, int8_path, args, prompt_table, prompt_manifest)
    shutil.copy2(int8_path, report_dir / int8_path.name)
    inspection = inspect_tflite(int8_path)

    compiler = {"status": "skipped", "reason": "--skip-compile was set"}
    if not args.skip_compile:
        compiler = run_compiler(args, int8_path, report_dir)

    compiled_path = report_dir / "model_int8_edgetpu.tflite"
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "description": (
            "Dummy untrained two-input model for testing the Coral Micro app contract. "
            "Inputs match the current raw prompt embedding paradigm: image tensor plus "
            "quantized prompt embedding tensor."
        ),
        "run_name": args.run_name,
        "config": {
            key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
        },
        "model": {
            "params": int(model.count_params()),
            "inputs": [item.name for item in model.inputs],
            "outputs": [item.name for item in model.outputs],
        },
        "artifacts": {
            "report_dir": str(report_dir),
            "tflite_dir": str(tflite_dir),
            "int8_tflite": str(int8_path),
            "int8_tflite_sha256": sha256(int8_path),
            "compiled_tflite": str(compiled_path) if compiled_path.exists() else None,
            "compiled_tflite_sha256": sha256(compiled_path) if compiled_path.exists() else None,
        },
        "prompt_lookup": {
            "dir": str(args.prompt_lookup_dir),
            "loaded": prompt_table is not None,
            "shape": list(prompt_table.shape) if prompt_table is not None else None,
            "manifest": str(args.prompt_lookup_dir / "prompt_embedding_lookup_manifest.json")
            if prompt_manifest is not None
            else None,
        },
        "tflite_inspection": inspection,
        "edgetpu_compiler": compiler,
    }
    (report_dir / "dummy_raw_prompt_embedding_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
