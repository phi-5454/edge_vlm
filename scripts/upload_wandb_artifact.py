#!/usr/bin/env python3
"""Upload one or more local files as a W&B artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

import wandb
from dotenv import load_dotenv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--entity", default=None)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--job-type", default="artifact-upload")
    parser.add_argument("--artifact-name", required=True)
    parser.add_argument("--artifact-type", default="model-checkpoint")
    parser.add_argument("--alias", action="append", default=[])
    parser.add_argument("--file", type=Path, action="append", required=True)
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument("--mode", default="online")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.env_file is not None and args.env_file.exists():
        load_dotenv(args.env_file, override=False)
    paths = [path for path in args.file if path.exists()]
    if not paths:
        raise FileNotFoundError("None of the requested artifact files exist.")
    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.run_name,
        job_type=args.job_type,
        mode=args.mode,
    )
    artifact = wandb.Artifact(args.artifact_name, type=args.artifact_type)
    for path in paths:
        artifact.add_file(str(path))
    run.log_artifact(artifact, aliases=args.alias or None)
    run.finish()


if __name__ == "__main__":
    main()
