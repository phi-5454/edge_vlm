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
DEFAULT_PROMPT_LOOKUP_HEADER = Path(
    "artifacts/exports/coral/prompt_embedding_lookup/tallyqa_prompt_embedding_lookup.h"
)
CONFIG_NAME = "vlm_micro_selftest_config.h"


def infer_model_kind(model: Path) -> str:
    name = model.name.lower()
    path = str(model).lower()
    if "ssd" in name or "mobiledet" in path or "mobilenet_v2" in path:
        return "detection"
    return "tallyqa"


def config_text(model_kind: str) -> str:
    if model_kind == "detection":
        return (
            "#pragma once\n\n"
            "#define VLM_MICRO_ENABLE_PROMPT_LOOKUP 0\n"
            "#define VLM_MICRO_ENABLE_DETECTION_POSTPROCESS 1\n"
            '#define VLM_MICRO_MODEL_KIND "detection"\n'
        )
    if model_kind == "tallyqa":
        return (
            "#pragma once\n\n"
            "#define VLM_MICRO_ENABLE_PROMPT_LOOKUP 1\n"
            "#define VLM_MICRO_ENABLE_DETECTION_POSTPROCESS 0\n"
            '#define VLM_MICRO_MODEL_KIND "tallyqa"\n'
        )
    raise ValueError(f"Unknown model kind: {model_kind}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coralmicro", type=Path, default=Path("../coralmicro"))
    parser.add_argument(
        "--source-app",
        type=Path,
        default=Path("coral_micro/tallyqa_benchmark_serial"),
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--prompt-lookup-header",
        type=Path,
        default=DEFAULT_PROMPT_LOOKUP_HEADER,
        help=(
            "Quantized prompt embedding lookup header to stage into the app. "
            "Required by the two-input prompt-embedding benchmark app."
        ),
    )
    parser.add_argument(
        "--model-kind",
        choices=("auto", "tallyqa", "detection"),
        default="auto",
        help="Controls which TFLM op resolver/prompt code is compiled.",
    )
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
    model_kind = infer_model_kind(args.model) if args.model_kind == "auto" else args.model_kind
    if model_kind == "tallyqa" and not args.prompt_lookup_header.exists():
        raise FileNotFoundError(args.prompt_lookup_header)

    staged_app = sdk / "examples" / APP_NAME
    changed: list[str] = []
    for src in sorted(args.source_app.iterdir()):
        if src.is_file():
            dst = staged_app / src.name
            if copy_file(src, dst, args.force):
                changed.append(str(dst))

    config_dst = staged_app / CONFIG_NAME
    rendered_config = config_text(model_kind)
    if not config_dst.exists() or config_dst.read_text(encoding="utf-8") != rendered_config:
        if config_dst.exists() and not args.force:
            raise FileExistsError(
                f"{config_dst} exists and differs; rerun with --force to overwrite."
            )
        config_dst.write_text(rendered_config, encoding="utf-8")
        changed.append(str(config_dst))

    model_dst = sdk / "models" / MODEL_NAME
    if copy_file(args.model, model_dst, args.force):
        changed.append(str(model_dst))

    if model_kind == "tallyqa":
        lookup_dst = staged_app / args.prompt_lookup_header.name
        if copy_file(args.prompt_lookup_header, lookup_dst, args.force):
            changed.append(str(lookup_dst))

    examples_cmake = sdk / "examples" / "CMakeLists.txt"
    if ensure_examples_cmake(examples_cmake):
        changed.append(str(examples_cmake))

    if changed:
        print("Staged/updated:")
        for path in changed:
            print(f"  {path}")
    else:
        print("No staging changes needed.")
    print(f"Model kind: {model_kind}")

    print("\nBuild and flash from the Coral SDK root:")
    print("  bash build.sh -c")
    print(f"  make -C build -j \"$(nproc)\" {APP_NAME}")
    print(
        "  if [[ -e build/examples/{app}/{app} ]]; then "
        "python3 scripts/flashtool.py -e {app}; "
        "else python3 scripts/flashtool.py "
        "--elf_path build/examples/{app}/{app}.stripped "
        "--data_dir build/examples/{app}; fi".format(app=APP_NAME)
    )


if __name__ == "__main__":
    main()
