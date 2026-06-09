#!/usr/bin/env python3
"""Stage repo-owned MAX78000 people-count training files into ai8x-training."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_AI8X_TRAINING = Path("../MAX78000/ai8x-training")
FILES = (
    (
        Path("max78000/ai8x_training/models/ai85net-tallyqa-mbv3-small.py"),
        Path("models/ai85net-tallyqa-mbv3-small.py"),
    ),
    (
        Path("max78000/ai8x_training/datasets/tallyqa_people.py"),
        Path("datasets/tallyqa_people.py"),
    ),
    (
        Path("max78000/ai8x_training/policies/qat_policy_tallyqa_people.yaml"),
        Path("policies/qat_policy_tallyqa_people.yaml"),
    ),
    (
        Path("max78000/ai8x_training/policies/schedule-tallyqa-people.yaml"),
        Path("policies/schedule-tallyqa-people.yaml"),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ai8x-training", type=Path, default=DEFAULT_AI8X_TRAINING)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.ai8x_training.exists():
        raise FileNotFoundError(args.ai8x_training)

    for source, relative_target in FILES:
        if not source.exists():
            raise FileNotFoundError(source)
        target = args.ai8x_training / relative_target
        if target.exists() and not args.force:
            raise FileExistsError(f"{target} exists. Re-run with --force to replace it.")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        print(f"Staged {source} -> {target}")


if __name__ == "__main__":
    main()
