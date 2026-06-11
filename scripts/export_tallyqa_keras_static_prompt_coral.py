#!/usr/bin/env python3
"""Export a trained Keras TallyQA student as a one-prompt Coral TFLite model.

The normal Keras student has two inputs: prompt token ids and an image. The
current Coral Micro serial benchmark app streams only image bytes, so this
script wraps a trained student with a constant prompt token tensor and exports a
single-input, full-int8 TFLite model.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterable

os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import hydra
import numpy as np
from omegaconf import DictConfig
import tensorflow as tf

from scripts.train_tallyqa_keras_student import (
    KerasTallyQAData,
    absolute_path,
    build_keras_student_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="tallyqa_keras_student")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--skip-mismatch", action="store_true")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/exports/coral/static_prompt"))
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("artifacts/reports/coral/static_prompt_export"),
    )
    parser.add_argument("--representative-samples", type=int, default=1024)
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--compiler-bin", default="edgetpu_compiler")
    parser.add_argument("--compiler-container-image", default=None)
    parser.add_argument("--docker-bin", default="docker")
    parser.add_argument("--docker-user", default="host")
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Hydra-style overrides for tallyqa_keras_student. Prefix with -- before overrides.",
    )
    return parser.parse_args()


def clean_overrides(values: list[str]) -> list[str]:
    if values and values[0] == "--":
        return values[1:]
    return values


def compose_cfg(config_name: str, overrides: list[str]) -> DictConfig:
    config_dir = str((Path(__file__).resolve().parents[1] / "conf").resolve())
    with hydra.initialize_config_dir(version_base=None, config_dir=config_dir):
        return hydra.compose(config_name=config_name, overrides=overrides)


def prompt_token_ids(data: KerasTallyQAData, prompt: str) -> np.ndarray:
    class_ids = {
        int(row["item_class_id"])
        for row in data.rows
        if str(row.get("student_prompt")) == prompt
    }
    if not class_ids:
        available = sorted({str(row.get("student_prompt")) for row in data.rows})
        sample = ", ".join(available[:30])
        raise ValueError(f"Prompt {prompt!r} not found. First available prompts: {sample}")
    if len(class_ids) != 1:
        raise ValueError(f"Prompt {prompt!r} maps to multiple item_class_id values: {sorted(class_ids)}")
    class_id = next(iter(class_ids))
    return data.prompt_token_ids[class_id : class_id + 1].astype(np.int32)


def static_prompt_model(
    student: tf.keras.Model,
    cfg: DictConfig,
    token_ids: np.ndarray,
) -> tf.keras.Model:
    batch_size = cfg.keras_model.get("batch_size", None)
    batch_size = None if batch_size is None else int(batch_size)
    images = tf.keras.Input(
        shape=(int(cfg.keras_model.image_size), int(cfg.keras_model.image_size), 3),
        batch_size=batch_size,
        dtype=tf.float32,
        name="images",
    )
    constant_tokens = tf.constant(token_ids, dtype=tf.int32)
    logits = student({"token_ids": constant_tokens, "images": images}, training=False)
    return tf.keras.Model(images, logits, name=f"{student.name}_static_prompt")


def representative_images(
    data: KerasTallyQAData,
    max_samples: int,
) -> Iterable[list[np.ndarray]]:
    indices = [
        *data.indices.get("train", []),
        *data.indices.get("val", []),
        *data.indices.get("test", []),
    ]
    if not indices:
        raise ValueError("No active dataset examples available for representative calibration.")
    rng = np.random.default_rng(int(data.cfg.seed))
    chosen = rng.choice(
        np.asarray(indices, dtype=np.int64),
        size=min(max_samples, len(indices)),
        replace=False,
    )
    for index in chosen.tolist():
        row = data.rows[int(index)]
        yield [data.image(int(row["image_index"]))[np.newaxis, ...].astype(np.float32)]


def export_full_int8_tflite(
    model: tf.keras.Model,
    output: Path,
    data: KerasTallyQAData,
    representative_samples: int,
    cfg: DictConfig,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_images(data, representative_samples)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    if bool(cfg.export.quantization.get("full_integer", True)):
        input_type = str(cfg.export.quantization.get("inference_input_type", "uint8"))
        output_type = str(cfg.export.quantization.get("inference_output_type", "int8"))
        converter.inference_input_type = tf.uint8 if input_type == "uint8" else tf.int8
        converter.inference_output_type = tf.uint8 if output_type == "uint8" else tf.int8
    output.write_bytes(converter.convert())


def inspect_tflite(path: Path) -> dict[str, Any]:
    interpreter = tf.lite.Interpreter(
        model_path=str(path),
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
        experimental_preserve_all_tensors=True,
        experimental_delegates=[],
    )
    interpreter.allocate_tensors()

    def tensor(item: dict[str, Any]) -> dict[str, Any]:
        scale, zero_point = item.get("quantization", (0.0, 0))
        return {
            "name": str(item["name"]),
            "shape": [int(value) for value in item["shape"].tolist()],
            "shape_signature": [
                int(value) for value in item.get("shape_signature", item["shape"]).tolist()
            ],
            "dtype": str(item["dtype"]),
            "quantization": [float(scale), int(zero_point)],
        }

    return {
        "inputs": [tensor(item) for item in interpreter.get_input_details()],
        "outputs": [tensor(item) for item in interpreter.get_output_details()],
        "operators": [
            {
                "index": index,
                "op_name": str(item.get("op_name")),
                "inputs": [int(value) for value in item.get("inputs", [])],
                "outputs": [int(value) for value in item.get("outputs", [])],
            }
            for index, item in enumerate(interpreter._get_ops_details())  # noqa: SLF001
        ],
    }


def find_compiler(compiler_bin: str) -> str | None:
    compiler_path = Path(compiler_bin)
    if compiler_path.exists():
        return str(compiler_path)
    found = shutil.which(compiler_bin)
    if found is not None:
        return found
    for root in [Path("../coral"), Path("../coralmicro")]:
        if not root.exists():
            continue
        direct = root / compiler_bin
        if direct.exists():
            return str(direct)
        matches = sorted(root.rglob(compiler_bin))
        if matches:
            return str(matches[0])
    return None


def run_compiler(args: argparse.Namespace, tflite_path: Path, report_dir: Path) -> dict[str, Any]:
    report_dir.mkdir(parents=True, exist_ok=True)
    if args.compiler_container_image:
        docker = shutil.which(args.docker_bin)
        if docker is None:
            return {"status": "skipped", "reason": f"{args.docker_bin!r} not found"}
        repo_root = Path.cwd().resolve()
        tflite_abs = tflite_path.resolve()
        report_abs = report_dir.resolve()
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
            str(Path("/workspace") / report_abs.relative_to(repo_root)),
            str(Path("/workspace") / tflite_abs.relative_to(repo_root)),
        ]
        result = subprocess.run(cmd, text=True, capture_output=True, check=False)
        (report_dir / "edgetpu_compiler.docker.stdout.txt").write_text(
            result.stdout,
            encoding="utf-8",
        )
        (report_dir / "edgetpu_compiler.docker.stderr.txt").write_text(
            result.stderr,
            encoding="utf-8",
        )
    else:
        compiler = find_compiler(args.compiler_bin)
        if compiler is None:
            return {"status": "skipped", "reason": f"{args.compiler_bin!r} not found"}
        cmd = [compiler, "-o", str(report_dir), str(tflite_path)]
        result = subprocess.run(cmd, text=True, capture_output=True, check=False)
        (report_dir / "edgetpu_compiler.stdout.txt").write_text(result.stdout, encoding="utf-8")
        (report_dir / "edgetpu_compiler.stderr.txt").write_text(result.stderr, encoding="utf-8")
    return {
        "status": "ok" if result.returncode == 0 else "failed",
        "returncode": int(result.returncode),
        "command": cmd,
    }


def main() -> None:
    args = parse_args()
    run_name = args.run_name or f"static_prompt_{args.prompt.replace(' ', '_')}"
    output_dir = args.output_dir / run_name
    report_dir = args.report_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    overrides = clean_overrides(args.overrides)
    if not any(
        item.startswith("keras_model.batch_size=") or item.startswith("+keras_model.batch_size=")
        for item in overrides
    ):
        overrides.append("+keras_model.batch_size=1")
    if not any(item.startswith("data.require_teacher_cache=") for item in overrides):
        overrides.append("data.require_teacher_cache=false")
    if not any(item.startswith("data.missing_teacher_policy=") for item in overrides):
        overrides.append("data.missing_teacher_policy=keep")
    cfg = compose_cfg(args.config_name, overrides)
    data = KerasTallyQAData(cfg)
    prompt_tokens = prompt_token_ids(data, args.prompt)
    prompt_length = int(data.prompt_token_ids.shape[1])

    student = build_keras_student_model(cfg, data.embedding_rows, prompt_length)
    weights = absolute_path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(weights)
    student.load_weights(weights, skip_mismatch=args.skip_mismatch)
    wrapped = static_prompt_model(student, cfg, prompt_tokens)

    float_path = output_dir / "model_float.tflite"
    int8_path = output_dir / "model_int8.tflite"
    tf.lite.TFLiteConverter.from_keras_model(wrapped).convert()
    float_path.write_bytes(tf.lite.TFLiteConverter.from_keras_model(wrapped).convert())
    export_full_int8_tflite(wrapped, int8_path, data, args.representative_samples, cfg)

    inspection = inspect_tflite(int8_path)
    compiler = {"status": "skipped", "reason": "--skip-compile was set"}
    if not args.skip_compile:
        compiler = run_compiler(args, int8_path, report_dir)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": run_name,
        "prompt": args.prompt,
        "prompt_token_ids": prompt_tokens.reshape(-1).astype(int).tolist(),
        "weights": str(weights),
        "skip_mismatch": bool(args.skip_mismatch),
        "config_name": args.config_name,
        "overrides": overrides,
        "outputs": {
            "float_tflite": str(float_path),
            "int8_tflite": str(int8_path),
        },
        "tflite_inspection": inspection,
        "edgetpu_compiler": compiler,
        "notes": {
            "static_prompt": (
                "The prompt token ids are embedded as a constant, producing an image-only model "
                "compatible with the current Coral Micro serial benchmark protocol."
            )
        },
    }
    if compiler.get("status") == "ok":
        compiled = sorted(report_dir.glob("*_edgetpu.tflite"))
        if compiled:
            manifest["outputs"]["edgetpu_tflite"] = str(compiled[0])
    (report_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["outputs"], indent=2))
    print(f"Wrote {report_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
