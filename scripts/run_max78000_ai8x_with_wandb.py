#!/usr/bin/env python3
"""Run ADI ai8x-training while streaming metrics and artifacts to W&B."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import wandb
from dotenv import load_dotenv


TRAIN_RE = re.compile(
    r"^Epoch:\s+\[(?P<epoch>-?\d+)\]\[\s*(?P<step>\d+)/\s*(?P<total>\d+)\]\s+(?P<body>.*)$"
)
TEST_RE = re.compile(r"^Test:\s+\[\s*(?P<step>\d+)/\s*(?P<total>\d+)\]\s+(?P<body>.*)$")
SUMMARY_RE = re.compile(r"^==>\s+(?P<body>.*)$")
BEST_RE = re.compile(r"^==>\s+Best\s+\[(?P<body>.*)\]$")
PAIR_RE = re.compile(
    r"(?P<key>[A-Za-z][A-Za-z0-9 _/@().-]*?)\s*[: ]\s*"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    r"(?=\s+[A-Za-z=]|\s*$|\])"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--entity", default=None)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--job-type", default="max78000-training")
    parser.add_argument("--mode", default="online")
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-report-dir", type=Path, default=None)
    parser.add_argument("--distiller-pythonpath", type=Path, default=None)
    parser.add_argument("--checkpoint-root", type=Path, default=None)
    parser.add_argument("--checkpoint-run-name", default=None)
    parser.add_argument("--post-eval-output-dir", type=Path, default=None)
    parser.add_argument("--post-eval-samples", type=int, default=4)
    parser.add_argument("--post-eval-batch-size", type=int, default=256)
    parser.add_argument("--skip-post-eval", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        raise SystemExit("Missing command after --")
    return args


def clean_key(value: str) -> str:
    key = value.strip().lower()
    key = key.replace("@", "")
    key = re.sub(r"[^a-z0-9]+", "_", key).strip("_")
    return key or "value"


def parse_pairs(body: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for match in PAIR_RE.finditer(body):
        metrics[clean_key(match.group("key"))] = float(match.group("value"))
    return metrics


def parse_line(line: str, last_epoch: int | None) -> tuple[dict[str, Any], int | None]:
    stripped = line.strip()
    train_match = TRAIN_RE.match(stripped)
    if train_match:
        epoch = int(train_match.group("epoch"))
        step = int(train_match.group("step"))
        total = int(train_match.group("total"))
        global_step = max(epoch, 0) * total + step
        metrics = {
            "max78000/epoch": epoch,
            "max78000/train/step": step,
            "max78000/train/steps_per_epoch": total,
            "max78000/global_step": global_step,
        }
        for key, value in parse_pairs(train_match.group("body")).items():
            metrics[f"max78000/train/{key}"] = value
        return metrics, epoch

    test_match = TEST_RE.match(stripped)
    if test_match:
        step = int(test_match.group("step"))
        total = int(test_match.group("total"))
        metrics = {
            "max78000/eval/step": step,
            "max78000/eval/steps_total": total,
        }
        if last_epoch is not None:
            metrics["max78000/epoch"] = last_epoch
            metrics["max78000/global_step"] = last_epoch * total + step
        for key, value in parse_pairs(test_match.group("body")).items():
            metrics[f"max78000/eval/{key}"] = value
        return metrics, last_epoch

    best_match = BEST_RE.match(stripped)
    if best_match:
        metrics = {}
        for key, value in parse_pairs(best_match.group("body")).items():
            metrics[f"max78000/best/{key}"] = value
        return metrics, last_epoch

    summary_match = SUMMARY_RE.match(stripped)
    if summary_match:
        metrics = {}
        for key, value in parse_pairs(summary_match.group("body")).items():
            metrics[f"max78000/val/{key}"] = value
        if last_epoch is not None:
            metrics["max78000/epoch"] = last_epoch
        return metrics, last_epoch

    return {}, last_epoch


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def iter_artifact_files(manifest: Path, log_file: Path, model_report_dir: Path | None) -> list[Path]:
    files = [manifest, log_file]
    if model_report_dir is not None and model_report_dir.exists():
        files.extend(path for path in sorted(model_report_dir.rglob("*")) if path.is_file())
    return [path for path in files if path.exists()]


def iter_output_files(path: Path | None) -> list[Path]:
    if path is None or not path.exists():
        return []
    if path.is_file():
        return [path]
    return sorted(item for item in path.rglob("*") if item.is_file())


def add_files_to_artifact(artifact: wandb.Artifact, files: list[Path], base_path: Path | None = None) -> None:
    for path in files:
        if base_path is not None:
            try:
                artifact.add_file(str(path), name=str(path.relative_to(base_path)))
                continue
            except ValueError:
                pass
        artifact.add_file(str(path))


def log_model_report_payload(model_report_dir: Path | None) -> None:
    if model_report_dir is None or not model_report_dir.exists():
        return
    payload: dict[str, Any] = {}
    readable = model_report_dir / "architecture_readable.md"
    if readable.exists():
        text = readable.read_text(encoding="utf-8")
        payload["model/architecture"] = wandb.Html(
            "<pre>" + text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") + "</pre>"
        )
    parameter_chart = model_report_dir / "architecture_parameters.png"
    if parameter_chart.exists():
        payload["model/layer_parameter_bars"] = wandb.Image(str(parameter_chart))
    architecture_json = model_report_dir / "architecture.json"
    if architecture_json.exists():
        try:
            architecture = json.loads(architecture_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            architecture = None
        if isinstance(architecture, dict):
            for key in ("factory", "input_shape", "output_shape", "total_parameters"):
                if key in architecture:
                    payload[f"model/{key}"] = architecture[key]
    if payload:
        wandb.log(payload)


def find_best_checkpoint(root: Path | None, run_name: str | None) -> Path | None:
    if root is None or not root.exists():
        return None
    candidates: list[Path] = []
    if run_name:
        candidates.extend(root.rglob(f"{run_name}_best.pth.tar"))
        candidates.extend(
            path for path in root.rglob("*_best.pth.tar") if run_name in str(path.parent)
        )
    else:
        candidates.extend(root.rglob("best.pth.tar"))
        candidates.extend(root.rglob("*_best.pth.tar"))
    deduped = sorted(
        {path.resolve() for path in candidates if path.is_file()},
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return deduped[0] if deduped else None


def find_checkpoints(root: Path | None, run_name: str | None) -> list[Path]:
    if root is None or not root.exists():
        return []
    candidates: list[Path] = []
    if run_name:
        candidates.extend(root.rglob(f"{run_name}*.pth.tar"))
        candidates.extend(
            path for path in root.rglob("*.pth.tar") if run_name in str(path.parent)
        )
    else:
        candidates.extend(root.rglob("*.pth.tar"))
    deduped = sorted(
        {path.resolve() for path in candidates if path.is_file()},
        key=lambda path: (path.stat().st_mtime, str(path)),
    )
    return deduped


def add_checkpoint_files_to_artifact(
    artifact: wandb.Artifact,
    checkpoints: list[Path],
    root: Path | None,
) -> None:
    for checkpoint in checkpoints:
        if root is not None:
            try:
                artifact.add_file(str(checkpoint), name=str(checkpoint.relative_to(root)))
                continue
            except ValueError:
                pass
        artifact.add_file(str(checkpoint))


def log_checkpoint_histograms(checkpoint: Path, max_tensors: int = 96) -> None:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - optional post-run diagnostics.
        print(f"Warning: could not import torch for checkpoint histograms: {exc}")
        return
    try:
        payload = torch.load(checkpoint, map_location="cpu")
    except Exception as exc:  # pragma: no cover - best-effort logging.
        print(f"Warning: could not load checkpoint histograms from {checkpoint}: {exc}")
        return
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        return

    histogram_payload: dict[str, Any] = {}
    logged = 0
    skipped = 0
    for name, tensor in sorted(state.items()):
        if logged >= max_tensors:
            skipped += 1
            continue
        if not torch.is_tensor(tensor) or tensor.numel() == 0 or not tensor.is_floating_point():
            continue
        values = tensor.detach().cpu().float().flatten().numpy()
        histogram_payload[f"max78000/weights/{clean_key(str(name))}"] = wandb.Histogram(values)
        logged += 1
    if histogram_payload:
        histogram_payload["max78000/weights/histogram_tensors"] = logged
        histogram_payload["max78000/weights/histogram_tensors_skipped"] = skipped
        wandb.log(histogram_payload)


def run_post_eval(
    *,
    args: argparse.Namespace,
    manifest: dict[str, Any],
    checkpoint: Path,
) -> dict[str, Any] | None:
    if args.skip_post_eval:
        return None
    data_dir = manifest.get("data_dir")
    model_name = manifest.get("model_name")
    input_channels = manifest.get("model_input_channels")
    if not data_dir or not model_name or not input_channels:
        print("Warning: manifest is missing data_dir/model_name/model_input_channels; skipping post-eval plots.")
        return None
    output_dir = args.post_eval_output_dir or (
        args.manifest.parent / "max78000_wandb_eval"
    )
    command = [
        sys.executable,
        str(Path(__file__).with_name("evaluate_max78000_tallyqa_wandb_outputs.py")),
        "--ai8x-training",
        str(args.cwd),
        "--checkpoint",
        str(checkpoint),
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(output_dir),
        "--factory",
        str(model_name),
        "--input-channels",
        str(int(input_channels)),
        "--num-classes",
        "6",
        "--batch-size",
        str(args.post_eval_batch_size),
        "--samples",
        str(args.post_eval_samples),
    ]
    print("+ " + " ".join(command))
    try:
        subprocess.run(command, check=True)
    except Exception as exc:  # pragma: no cover - this is best-effort logging glue.
        print(f"Warning: MAX78000 post-eval plots failed: {exc}")
        return None
    result_path = output_dir / "max78000_eval_results.json"
    if not result_path.exists():
        print(f"Warning: MAX78000 post-eval did not write {result_path}")
        return None
    return json.loads(result_path.read_text(encoding="utf-8"))


def log_post_eval_payload(results: dict[str, Any], output_dir: Path | None) -> None:
    payload: dict[str, Any] = {}
    for split, split_result in (results.get("splits") or {}).items():
        for key, value in (split_result.get("metrics") or {}).items():
            payload[f"max78000/{split}/{key}"] = float(value)
        confusion = split_result.get("confusion_matrix")
        if confusion:
            payload[f"{split}_plots/confusion_matrix"] = wandb.Image(str(confusion))
        image_encoding = split_result.get("image_encoding")
        if image_encoding:
            payload[f"{split}_plots/image_encoding"] = wandb.Image(str(image_encoding))
            payload[f"{split}_plots/image_encoding_count"] = int(
                split_result.get("unique_plot_samples") or 0
            )
        prediction_examples = split_result.get("prediction_examples")
        if prediction_examples:
            payload[f"{split}_plots/example_predictions"] = wandb.Image(str(prediction_examples))
    if payload:
        wandb.log(payload)
    for path in iter_output_files(output_dir):
        wandb.save(str(path), policy="now")


def main() -> None:
    args = parse_args()
    if args.env_file is not None and args.env_file.exists():
        load_dotenv(args.env_file, override=False)

    manifest = load_manifest(args.manifest)
    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.run_name,
        job_type=args.job_type,
        mode=args.mode,
        config={
            "max78000": manifest,
            "command": args.command,
            "cwd": str(args.cwd),
        },
    )
    wandb.define_metric("max78000/global_step")
    wandb.define_metric("max78000/train/*", step_metric="max78000/global_step")
    wandb.define_metric("max78000/eval/*", step_metric="max78000/global_step")
    wandb.define_metric("max78000/val/*", step_metric="max78000/epoch")
    wandb.define_metric("max78000/test/*", step_metric="max78000/epoch")
    wandb.define_metric("max78000/best/*", step_metric="max78000/epoch")
    log_model_report_payload(args.model_report_dir)

    env = os.environ.copy()
    if args.distiller_pythonpath is not None:
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{args.distiller_pythonpath}:{existing}" if existing else str(args.distiller_pythonpath)

    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    last_epoch: int | None = None
    returncode = 1
    with args.log_file.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            args.command,
            cwd=args.cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
            metrics, last_epoch = parse_line(line, last_epoch)
            if metrics:
                wandb.log(metrics)
        returncode = process.wait()

    wandb.log({"max78000/process/return_code": returncode})
    for path in iter_artifact_files(args.manifest, args.log_file, args.model_report_dir):
        wandb.save(str(path), policy="now")

    artifact = wandb.Artifact(f"{args.run_name}-max78000-training-report", type="training-report")
    add_files_to_artifact(
        artifact,
        iter_artifact_files(args.manifest, args.log_file, args.model_report_dir),
        base_path=args.manifest.parent,
    )
    run.log_artifact(artifact, aliases=["latest"])

    checkpoint_run_name = args.checkpoint_run_name or args.run_name
    all_checkpoints = find_checkpoints(args.checkpoint_root, checkpoint_run_name)
    if all_checkpoints:
        checkpoints_artifact = wandb.Artifact(
            f"{args.run_name}-max78000-checkpoints",
            type="model-checkpoints",
        )
        add_checkpoint_files_to_artifact(checkpoints_artifact, all_checkpoints, args.checkpoint_root)
        checkpoints_artifact.metadata.update(
            {
                "checkpoint_count": len(all_checkpoints),
                "checkpoint_paths": [str(path) for path in all_checkpoints],
            }
        )
        run.log_artifact(checkpoints_artifact, aliases=["latest"])
        run.summary["max78000/checkpoint_count"] = len(all_checkpoints)

    checkpoint = find_best_checkpoint(args.checkpoint_root, checkpoint_run_name)
    post_eval_results = None
    if checkpoint is not None:
        run.summary["max78000/best_checkpoint"] = str(checkpoint)
        log_checkpoint_histograms(checkpoint)
        post_eval_results = run_post_eval(args=args, manifest=manifest, checkpoint=checkpoint)
        if post_eval_results is not None:
            log_post_eval_payload(post_eval_results, args.post_eval_output_dir or (args.manifest.parent / "max78000_wandb_eval"))
        checkpoint_artifact = wandb.Artifact(
            f"{args.run_name}-chosen-test-checkpoint",
            type="model-checkpoint",
        )
        checkpoint_artifact.add_file(str(checkpoint))
        checkpoint_artifact.add_file(str(args.manifest))
        checkpoint_artifact.add_file(str(args.log_file))
        eval_output_dir = args.post_eval_output_dir or (args.manifest.parent / "max78000_wandb_eval")
        add_files_to_artifact(
            checkpoint_artifact,
            iter_output_files(eval_output_dir),
            base_path=eval_output_dir,
        )
        run.log_artifact(checkpoint_artifact, aliases=["best", "test-evaluated"])
        wandb.save(str(checkpoint), policy="now")
    elif returncode == 0:
        print("Warning: no MAX78000 checkpoint found; skipping post-eval plots.")

    output_artifact = wandb.Artifact(f"{args.run_name}-outputs", type="training-output")
    add_files_to_artifact(
        output_artifact,
        iter_artifact_files(args.manifest, args.log_file, args.model_report_dir),
        base_path=args.manifest.parent,
    )
    if checkpoint is not None:
        output_artifact.add_file(str(checkpoint))
    eval_output_dir = args.post_eval_output_dir or (args.manifest.parent / "max78000_wandb_eval")
    add_files_to_artifact(
        output_artifact,
        iter_output_files(eval_output_dir),
        base_path=eval_output_dir,
    )
    if post_eval_results is not None:
        output_artifact.metadata.update({"post_eval": post_eval_results})
    run.log_artifact(output_artifact, aliases=["latest"])
    run.finish(exit_code=returncode)
    sys.exit(returncode)


if __name__ == "__main__":
    main()
