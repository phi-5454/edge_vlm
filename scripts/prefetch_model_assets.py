from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path


DEFAULT_CACHE_ROOT = Path("data/model_cache")
DEFAULT_SMOLVLM_MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prefetch SmolVLM and TorchVision MobileNet weights into an explicit cache root."
    )
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--smolvlm-model", default=DEFAULT_SMOLVLM_MODEL)
    parser.add_argument("--skip-smolvlm", action="store_true")
    parser.add_argument("--skip-mobilenet", action="store_true")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/reports/model_asset_cache_summary.json"),
    )
    return parser.parse_args()


def configure_cache_env(cache_root: Path) -> dict[str, str]:
    hf_home = cache_root / "huggingface"
    torch_home = cache_root / "torch"
    env = {
        "HF_HOME": str(hf_home),
        "HF_HUB_CACHE": str(hf_home / "hub"),
        "TRANSFORMERS_CACHE": str(hf_home / "hub"),
        "TORCH_HOME": str(torch_home),
    }
    os.environ.update(env)
    return env


def prefetch_smolvlm(model_name: str, local_files_only: bool, trust_remote_code: bool) -> None:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    AutoProcessor.from_pretrained(
        model_name,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    AutoModelForImageTextToText.from_pretrained(
        model_name,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        torch_dtype="auto",
    )


def prefetch_mobilenets() -> None:
    from torchvision.models import (
        MobileNet_V3_Large_Weights,
        MobileNet_V3_Small_Weights,
        mobilenet_v3_large,
        mobilenet_v3_small,
    )

    mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
    mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)


def file_summary(root: Path) -> dict[str, int]:
    if not root.exists():
        return {"files": 0, "bytes": 0}
    files = [path for path in root.rglob("*") if path.is_file()]
    return {
        "files": len(files),
        "bytes": sum(path.stat().st_size for path in files),
    }


def main() -> None:
    args = parse_args()
    cache_root = args.cache_root
    cache_root.mkdir(parents=True, exist_ok=True)
    env = configure_cache_env(cache_root)

    actions = {
        "smolvlm": not args.skip_smolvlm,
        "mobilenet": not args.skip_mobilenet,
    }
    if args.dry_run:
        print(
            json.dumps(
                {
                    "cache_root": str(cache_root),
                    "environment": env,
                    "actions": actions,
                    "smolvlm_model": args.smolvlm_model,
                    "report": str(args.report),
                },
                indent=2,
            )
        )
        return

    if actions["smolvlm"]:
        prefetch_smolvlm(
            args.smolvlm_model,
            local_files_only=args.local_files_only,
            trust_remote_code=args.trust_remote_code,
        )
    if actions["mobilenet"]:
        prefetch_mobilenets()

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cache_root": str(cache_root),
        "environment": env,
        "actions": actions,
        "smolvlm_model": args.smolvlm_model,
        "huggingface_cache": file_summary(Path(env["HF_HOME"])),
        "torch_cache": file_summary(Path(env["TORCH_HOME"])),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote cache report: {args.report}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
