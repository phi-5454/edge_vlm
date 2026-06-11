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


def find_best_checkpoint(root: Path | None, run_name: str | None) -> Path | None:
    if root is None or not root.exists():
        return None
    candidates: list[Path] = []
    if run_name:
        candidates.extend(root.rglob(f"{run_name}_best.pth.tar"))
    candidates.extend(root.rglob("best.pth.tar"))
    candidates.extend(root.rglob("*_best.pth.tar"))
    deduped = sorted(
        {path.resolve() for path in candidates if path.is_file()},
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return deduped[0] if deduped else None


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
    wandb.define_metric("max78000/best/*", step_metric="max78000/epoch")

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
    for path in iter_artifact_files(args.manifest, args.log_file, args.model_report_dir):
        artifact.add_file(str(path))
    run.log_artifact(artifact, aliases=["latest"])

    checkpoint = find_best_checkpoint(args.checkpoint_root, args.checkpoint_run_name or args.run_name)
    if checkpoint is not None:
        checkpoint_artifact = wandb.Artifact(
            f"{args.run_name}-chosen-test-checkpoint",
            type="model-checkpoint",
        )
        checkpoint_artifact.add_file(str(checkpoint))
        checkpoint_artifact.add_file(str(args.manifest))
        checkpoint_artifact.add_file(str(args.log_file))
        run.log_artifact(checkpoint_artifact, aliases=["best", "test-evaluated"])
        wandb.save(str(checkpoint), policy="now")
    run.finish(exit_code=returncode)
    sys.exit(returncode)


if __name__ == "__main__":
    main()
