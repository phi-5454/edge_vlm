#!/usr/bin/env python3
"""Precompute tracked 16-d prompt vectors for MAX78000 prompt-plane inputs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import torch


DEFAULT_INPUT = Path("artifacts/models/tallyqa_smolvlm_prompt_embeddings_letterbox.pt")
DEFAULT_TIERED_CURRICULUM = Path(
    "artifacts/reports/final_dataset/post_pruning_teacher_eda/"
    "composite_teacher_ece_temp_smol1p1_frcnn2p2_beta12p968/tiered_curriculum"
)
DEFAULT_SYNONYMS = Path("conf/tallyqa_prompt_synonyms.json")
DEFAULT_OUTPUT = Path("max78000/prompt_embeddings/tallyqa_prompt_planes16_random.json")


def normalize_prompt(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def singular_form(prompt: str) -> str | None:
    prompt = normalize_prompt(prompt)
    irregular = {
        "people": "person",
        "men": "man",
        "women": "woman",
        "children": "child",
        "geese": "goose",
        "mice": "mouse",
    }
    if prompt in irregular:
        return irregular[prompt]
    if prompt.endswith("ies") and len(prompt) > 3:
        return prompt[:-3] + "y"
    if prompt.endswith("ses") and len(prompt) > 3:
        return prompt[:-2]
    if prompt.endswith("ches") or prompt.endswith("shes") or prompt.endswith("xes"):
        return prompt[:-2]
    if prompt.endswith("s") and not prompt.endswith("ss") and len(prompt) > 1:
        return prompt[:-1]
    return None


def read_prompt_file(path: Path) -> list[str]:
    return [normalize_prompt(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def collect_tier_prompts(tiered_curriculum: Path) -> tuple[list[str], dict[str, list[str]]]:
    prompts: set[str] = set()
    prompt_sources: dict[str, list[str]] = {}
    for prompt_file in sorted(tiered_curriculum.glob("tier_*/prompt_classes.txt")):
        tier_name = prompt_file.parent.name
        for prompt in read_prompt_file(prompt_file):
            prompts.add(prompt)
            prompt_sources.setdefault(prompt, []).append(tier_name)
    return sorted(prompts), prompt_sources


def load_synonyms(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    result: dict[str, list[str]] = {}
    for prompt, aliases in payload.items():
        if isinstance(aliases, str):
            aliases = [aliases]
        if not isinstance(aliases, list):
            raise ValueError(f"Synonyms for {prompt!r} must be a string or list.")
        prompt = normalize_prompt(prompt)
        cleaned = []
        seen = set()
        for alias in aliases:
            alias = normalize_prompt(str(alias))
            if alias and alias != prompt and alias not in seen:
                cleaned.append(alias)
                seen.add(alias)
        if cleaned:
            result[prompt] = cleaned
    return result


def base_vectors(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    classes = payload["prompt_classes"]
    pooled = payload["pooled_prompt_embeddings"].float()
    vectors: dict[str, torch.Tensor] = {}
    for index, item in enumerate(classes):
        prompt = normalize_prompt(item.get("item", item) if isinstance(item, dict) else str(item))
        vectors[prompt] = pooled[index].contiguous()
    return vectors


def projected_vectors(
    vectors: dict[str, torch.Tensor],
    prompts: list[str],
    aliases: dict[str, str],
    output_dim: int,
    seed: int,
) -> tuple[dict[str, list[float]], list[dict[str, Any]]]:
    generator = torch.Generator().manual_seed(seed)
    input_dim = next(iter(vectors.values())).numel()
    projection = torch.randn(input_dim, output_dim, generator=generator) / math.sqrt(input_dim)
    output: dict[str, list[float]] = {}
    rows: list[dict[str, Any]] = []
    for prompt in sorted(set(prompts) | set(aliases)):
        canonical = aliases.get(prompt, prompt)
        if canonical not in vectors:
            raise KeyError(f"Canonical prompt {canonical!r} for {prompt!r} missing from {DEFAULT_INPUT}")
        vector = vectors[canonical].float()
        vector = vector / vector.norm(p=2).clamp_min(1e-6)
        compact = vector @ projection
        compact = compact / compact.norm(p=2).clamp_min(1e-6)
        values = [round(float(value), 8) for value in compact.tolist()]
        output[prompt] = values
        rows.append(
            {
                "prompt": prompt,
                "canonical_prompt": canonical,
                "is_alias": prompt != canonical,
                "vector": values,
            }
        )
    return output, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--tiered-curriculum-dir", type=Path, default=DEFAULT_TIERED_CURRICULUM)
    parser.add_argument("--synonyms", type=Path, default=DEFAULT_SYNONYMS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"{args.output} exists. Pass --force to replace it.")
    if args.output_dim < 1:
        raise ValueError("--output-dim must be positive.")

    tier_prompts, prompt_sources = collect_tier_prompts(args.tiered_curriculum_dir)
    if not tier_prompts:
        raise RuntimeError(f"No tier prompt classes found under {args.tiered_curriculum_dir}.")

    first_tier_file = args.tiered_curriculum_dir / "tier_0_acc_ge_0p60_n_ge_1000" / "prompt_classes.txt"
    first_tier_prompts = read_prompt_file(first_tier_file) if first_tier_file.exists() else []
    aliases: dict[str, str] = {}
    alias_sources: dict[str, list[str]] = {}
    for prompt in first_tier_prompts:
        singular = singular_form(prompt)
        if singular and singular != prompt:
            aliases[singular] = prompt
            alias_sources.setdefault(singular, []).append("tier0_singular")
    for prompt, synonym_list in load_synonyms(args.synonyms).items():
        if prompt not in tier_prompts:
            continue
        for alias in synonym_list:
            aliases[alias] = prompt
            alias_sources.setdefault(alias, []).append("synonym")

    vectors = base_vectors(args.input)
    missing = [prompt for prompt in tier_prompts if prompt not in vectors]
    if missing:
        raise KeyError(f"Base prompt embedding artifact is missing tier prompts: {missing[:20]}")

    prompt_vectors, rows = projected_vectors(
        vectors=vectors,
        prompts=tier_prompts,
        aliases=aliases,
        output_dim=args.output_dim,
        seed=args.seed,
    )
    prompt_to_row = {row["prompt"]: index for index, row in enumerate(rows)}
    for row in rows:
        prompt = str(row["prompt"])
        row["tier_sources"] = prompt_sources.get(prompt, [])
        row["alias_sources"] = alias_sources.get(prompt, [])

    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_prompt_embeddings": str(args.input),
        "tiered_curriculum_dir": str(args.tiered_curriculum_dir),
        "synonyms": str(args.synonyms),
        "method": "l2_normalize_then_seeded_gaussian_random_projection_then_l2_normalize",
        "projection_seed": args.seed,
        "input_dim": int(next(iter(vectors.values())).numel()),
        "output_dim": args.output_dim,
        "plane_channels": args.output_dim,
        "canonical_prompt_count": len(tier_prompts),
        "alias_count": len(aliases),
        "prompt_count": len(rows),
        "prompt_to_row": prompt_to_row,
        "prompt_vectors": prompt_vectors,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} with {len(rows)} prompts/aliases")


if __name__ == "__main__":
    main()
