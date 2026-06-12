#!/usr/bin/env python3
"""Export a quantized prompt-embedding lookup table for Coral Micro firmware."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import torch


DEFAULT_PROMPT_ARTIFACT = Path("artifacts/models/tallyqa_smolvlm_prompt_embeddings_letterbox_synonyms.pt")
DEFAULT_TIERED_CURRICULUM_DIR = Path(
    "artifacts/reports/final_dataset/post_pruning_teacher_eda/"
    "composite_teacher_ece_temp_smol1p1_frcnn2p2_beta12p968/tiered_curriculum"
)
DEFAULT_QUANT_REPORT = Path(
    "artifacts/reports/tallyqa_prompt_query_quantization/"
    "raw-prompt-embedding-qaq-cache-smoke/prompt_embedding_quantization_metrics.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-artifact", type=Path, default=DEFAULT_PROMPT_ARTIFACT)
    parser.add_argument("--tiered-curriculum-dir", type=Path, default=DEFAULT_TIERED_CURRICULUM_DIR)
    parser.add_argument("--quant-report", type=Path, default=DEFAULT_QUANT_REPORT)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--zero-point", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/exports/coral/prompt_embedding_lookup"),
    )
    parser.add_argument("--header-name", default="tallyqa_prompt_embedding_lookup.h")
    return parser.parse_args()


def read_prompt_list(path: Path) -> list[str]:
    prompts: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            prompts.append(stripped)
    return prompts


def discover_tier_prompts(root: Path) -> tuple[list[dict[str, Any]], set[str]]:
    tiers: list[dict[str, Any]] = []
    union: set[str] = set()
    for prompt_file in sorted(root.glob("tier_*/prompt_classes.txt")):
        prompts = read_prompt_list(prompt_file)
        tier_name = prompt_file.parent.name
        tiers.append(
            {
                "tier": tier_name,
                "prompt_file": str(prompt_file),
                "prompts": prompts,
                "count": len(prompts),
            }
        )
        union.update(prompts)
    if not tiers:
        raise FileNotFoundError(f"No tier_*/prompt_classes.txt files found under {root}")
    return tiers, union


def load_quantization(path: Path, scale: float | None, zero_point: int | None) -> tuple[float, int]:
    if scale is not None and zero_point is not None:
        return float(scale), int(zero_point)
    if not path.exists():
        raise FileNotFoundError(
            f"Quantization report not found: {path}. Pass --scale and --zero-point explicitly."
        )
    report = json.loads(path.read_text(encoding="utf-8"))
    for item in report["int8_tflite_inspection"]["inputs"]:
        if "prompt" in str(item["name"]).lower():
            item_scale, item_zero_point = item["quantization"]
            return float(item_scale), int(item_zero_point)
    raise ValueError(f"No prompt input found in quantization report: {path}")


def pooled_embedding(
    embedding_rows: np.ndarray,
    compact_token_ids: list[int],
    attention_mask: list[bool],
) -> np.ndarray:
    compact_rows = np.concatenate(
        [np.zeros((1, embedding_rows.shape[1]), dtype=np.float32), embedding_rows],
        axis=0,
    )
    token_ids = np.asarray(compact_token_ids, dtype=np.int32)
    mask = np.asarray(attention_mask, dtype=np.float32)
    embedded = compact_rows[token_ids]
    denom = max(float(mask.sum()), 1.0)
    return ((embedded * mask[:, np.newaxis]).sum(axis=0) / denom).astype(np.float32)


def quantize_embedding(value: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    quantized = np.round(value / scale + zero_point)
    return np.clip(quantized, 0, 255).astype(np.uint8)


def c_identifier(value: str) -> str:
    identifier = re.sub(r"[^0-9A-Za-z_]+", "_", value.strip().lower()).strip("_")
    if not identifier:
        identifier = "prompt"
    if identifier[0].isdigit():
        identifier = f"prompt_{identifier}"
    return identifier


def c_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def write_header(path: Path, table: np.ndarray, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "#pragma once",
        "",
        "#include <stdint.h>",
        "",
        f"#define TALLYQA_PROMPT_EMBEDDING_COUNT {table.shape[0]}",
        f"#define TALLYQA_PROMPT_EMBEDDING_DIM {table.shape[1]}",
        "",
        "static const char* const kTallyQAPromptEmbeddingNames[TALLYQA_PROMPT_EMBEDDING_COUNT] = {",
    ]
    for entry in entries:
        lines.append(f'  "{c_string(str(entry["prompt"]))}",')
    lines.extend(
        [
            "};",
            "",
            "static const uint8_t kTallyQAPromptEmbeddingTable",
            "    [TALLYQA_PROMPT_EMBEDDING_COUNT][TALLYQA_PROMPT_EMBEDDING_DIM] = {",
        ]
    )
    for row_index, row in enumerate(table):
        prompt = c_identifier(str(entries[row_index]["prompt"]))
        lines.append(f"  // {row_index}: {prompt}")
        chunks = [
            ", ".join(str(int(value)) for value in row[start : start + 24])
            for start in range(0, len(row), 24)
        ]
        lines.append("  {")
        for chunk in chunks:
            lines.append(f"    {chunk},")
        lines.append("  },")
    lines.append("};")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    prompt_payload = torch.load(args.prompt_artifact, map_location="cpu", weights_only=False)
    embedding_rows = prompt_payload["embedding_rows"].float().numpy().astype(np.float32)
    scale, zero_point = load_quantization(args.quant_report, args.scale, args.zero_point)
    tiers, tier_prompt_union = discover_tier_prompts(args.tiered_curriculum_dir)

    entries_by_prompt: dict[str, dict[str, Any]] = {}
    for row in prompt_payload["prompt_classes"]:
        prompt = str(row["item"])
        entries_by_prompt[prompt] = {
            "prompt": prompt,
            "kind": "tier_prompt" if prompt in tier_prompt_union else "base_prompt",
            "source_item": prompt,
            "class_id": int(row["class_id"]),
            "compact_token_ids": [int(value) for value in row["compact_token_ids"]],
            "attention_mask": [bool(value) for value in row["attention_mask"]],
            "tiers": [
                tier["tier"]
                for tier in tiers
                if prompt in set(str(item) for item in tier["prompts"])
            ],
        }

    for row in prompt_payload.get("alias_prompt_classes", []):
        alias = str(row["alias"])
        entries_by_prompt.setdefault(
            alias,
            {
                "prompt": alias,
                "kind": "near_synonym",
                "source_item": str(row["source_item"]),
                "source_class_id": int(row["source_class_id"]),
                "compact_token_ids": [int(value) for value in row["compact_token_ids"]],
                "attention_mask": [bool(value) for value in row["attention_mask"]],
                "tiers": [],
            },
        )

    entries = sorted(
        entries_by_prompt.values(),
        key=lambda item: (
            {"tier_prompt": 0, "base_prompt": 1, "near_synonym": 2}.get(str(item["kind"]), 3),
            str(item["prompt"]),
        ),
    )
    if not entries:
        raise RuntimeError("No prompt lookup entries were generated.")

    float_table = np.stack(
        [
            pooled_embedding(
                embedding_rows,
                entry["compact_token_ids"],
                entry["attention_mask"],
            )
            for entry in entries
        ],
        axis=0,
    )
    quantized_table = quantize_embedding(float_table, scale, zero_point)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    npy_path = args.output_dir / "prompt_embedding_lookup_uint8.npy"
    json_path = args.output_dir / "prompt_embedding_lookup_manifest.json"
    header_path = args.output_dir / args.header_name
    np.save(npy_path, quantized_table)
    write_header(header_path, quantized_table, entries)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "prompt_artifact": str(args.prompt_artifact),
        "tiered_curriculum_dir": str(args.tiered_curriculum_dir),
        "quant_report": str(args.quant_report) if args.quant_report is not None else None,
        "quantization": {
            "dtype": "uint8",
            "scale": scale,
            "zero_point": zero_point,
            "dimension": int(quantized_table.shape[1]),
        },
        "counts": {
            "entries": len(entries),
            "tier_prompts": sum(1 for entry in entries if entry["kind"] == "tier_prompt"),
            "base_prompts": sum(1 for entry in entries if entry["kind"] == "base_prompt"),
            "near_synonyms": sum(1 for entry in entries if entry["kind"] == "near_synonym"),
            "tiers": len(tiers),
        },
        "tiers": tiers,
        "entries": entries,
        "outputs": {
            "npy": str(npy_path),
            "header": str(header_path),
            "manifest": str(json_path),
        },
    }
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"outputs": manifest["outputs"], "counts": manifest["counts"]}, indent=2))


if __name__ == "__main__":
    main()
