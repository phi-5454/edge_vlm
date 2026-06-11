#!/usr/bin/env python3
"""Stage the repo-owned Coral Micro TallyQA serial benchmark app into the SDK."""

from __future__ import annotations

import argparse
import filecmp
import shutil
from pathlib import Path


APP_NAME = "vlm_micro_tallyqa_benchmark_serial"
MODEL_NAME = "tallyqa_prompt_patch_mlp_edgetpu.tflite"
DEFAULT_MODEL = Path(
    "artifacts/reports/coral/edgetpu_compiler/"
    "prompt_patch_mlp_static_prompt_minimalistic_large_compile_probe_docker/"
    "ptq/model_int8_edgetpu.tflite"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coralmicro", type=Path, default=Path("../coralmicro"))
    parser.add_argument(
        "--source-app",
        type=Path,
        default=Path("coral_micro/tallyqa_benchmark_serial"),
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--force", action="store_true", help="Overwrite staged app files.")
    return parser.parse_args()


def copy_file(src: Path, dst: Path, force: bool) -> bool:
    if dst.exists() and filecmp.cmp(src, dst, shallow=False):
        return False
    if dst.exists() and not force:
        raise FileExistsError(f"{dst} exists and differs; rerun with --force to overwrite.")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def ensure_examples_cmake(cmake_path: Path) -> bool:
    line = f"add_subdirectory({APP_NAME})"
    text = cmake_path.read_text(encoding="utf-8")
    if line in text:
        return False
    cmake_path.write_text(text.rstrip() + f"\n{line}\n", encoding="utf-8")
    return True


def main() -> None:
    args = parse_args()
    sdk = args.coralmicro.resolve()
    if not (sdk / "build.sh").exists():
        raise FileNotFoundError(f"{sdk} does not look like a coralmicro SDK checkout.")
    if not args.source_app.is_dir():
        raise FileNotFoundError(args.source_app)
    if not args.model.exists():
        raise FileNotFoundError(args.model)

    staged_app = sdk / "examples" / APP_NAME
    changed: list[str] = []
    for src in sorted(args.source_app.iterdir()):
        if src.is_file():
            dst = staged_app / src.name
            if copy_file(src, dst, args.force):
                changed.append(str(dst))

    model_dst = sdk / "models" / MODEL_NAME
    if copy_file(args.model, model_dst, args.force):
        changed.append(str(model_dst))

    examples_cmake = sdk / "examples" / "CMakeLists.txt"
    if ensure_examples_cmake(examples_cmake):
        changed.append(str(examples_cmake))

    if changed:
        print("Staged/updated:")
        for path in changed:
            print(f"  {path}")
    else:
        print("No staging changes needed.")

    print("\nBuild and flash from the Coral SDK root:")
    print("  bash build.sh")
    print(f"  python3 scripts/flashtool.py -e {APP_NAME}")


if __name__ == "__main__":
    main()
