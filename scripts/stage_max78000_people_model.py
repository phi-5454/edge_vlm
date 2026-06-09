#!/usr/bin/env python3
"""Stage the repo-owned MAX78000 people-count model into ai8x-training."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


SOURCE = Path("max78000/ai8x_training/models/ai85net-tallyqa-mbv3-small.py")
DEFAULT_AI8X_TRAINING = Path("../MAX78000/ai8x-training")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ai8x-training", type=Path, default=DEFAULT_AI8X_TRAINING)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = args.ai8x_training / "models" / SOURCE.name
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    if not args.ai8x_training.exists():
        raise FileNotFoundError(args.ai8x_training)
    if target.exists() and not args.force:
        raise FileExistsError(f"{target} exists. Re-run with --force to replace it.")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE, target)
    print(f"Staged {SOURCE} -> {target}")


if __name__ == "__main__":
    main()
