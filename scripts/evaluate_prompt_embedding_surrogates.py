#!/usr/bin/env python3
"""Evaluate prompt embedding surrogate swaps on TallyQA Keras checkpoints.

This is a cheap feasibility experiment: it keeps model weights and selected
dataset rows fixed, then replaces the source prompt's compact embedding row(s)
with the mean SmolVLM token embedding for a surrogate prompt such as
``people=humans``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

hydra = None
tf = None
torch = None
tqdm = None
AutoModelForImageTextToText = None
AutoProcessor = None
KerasTallyQAData = None
MulticlassAccumulator = None
absolute_path = None
build_keras_student_model = None
build_student_for_weights = None


DEFAULT_WEIGHTS = Path(
    "artifacts/models/tallyqa_keras_student/"
    "tallyqa-keras-tier0-current-prompt-patch-mlp-ptq-outputs_v2/best.weights.h5"
)
DEFAULT_OUTPUT = Path("artifacts/reports/tallyqa_prompt_surrogate_embedding_eval.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="tallyqa_keras_student")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        metavar="SOURCE=SURROGATE",
        help=(
            "Prompt surrogate pair. May be repeated. Defaults to "
            "people=humans and horses=stallions."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-examples-per-prompt", type=int, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=None, help="SmolVLM model for surrogate embeddings.")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--embedding-helper-json", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--embedding-helper-output", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Hydra-style overrides for tallyqa_keras_student. Prefix with -- before overrides.",
    )
    return parser.parse_args()


def run_embedding_helper(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if args.embedding_helper_json is None or args.embedding_helper_output is None:
        raise ValueError("Embedding helper requires input and output paths.")
    request = json.loads(args.embedding_helper_json.read_text(encoding="utf-8"))
    model_name = str(request["model"])
    texts = [str(value) for value in request["texts"]]
    processor = AutoProcessor.from_pretrained(
        model_name,
        local_files_only=bool(args.local_files_only),
        trust_remote_code=bool(args.trust_remote_code),
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        local_files_only=bool(args.local_files_only),
        trust_remote_code=bool(args.trust_remote_code),
        torch_dtype=str(args.torch_dtype),
    )
    embedding = model.get_input_embeddings()
    if embedding is None:
        raise ValueError(f"{model_name} does not expose input embeddings.")
    results = {}
    for text in texts:
        token_ids = [
            int(token_id)
            for token_id in processor.tokenizer(text, add_special_tokens=False)["input_ids"]
        ]
        if not token_ids:
            raise ValueError(f"Surrogate prompt {text!r} produced no tokens.")
        tokens = [str(token) for token in processor.tokenizer.convert_ids_to_tokens(token_ids)]
        rows = embedding.weight.detach().cpu()[token_ids].float().numpy()
        results[text] = {
            "text": text,
            "teacher_token_ids": token_ids,
            "teacher_tokens": tokens,
            "token_count": len(token_ids),
            "mean_embedding": rows.mean(axis=0).astype(float).tolist(),
        }
    args.embedding_helper_output.write_text(json.dumps(results), encoding="utf-8")


def clean_overrides(values: list[str]) -> list[str]:
    if values and values[0] == "--":
        return values[1:]
    return values


def compose_cfg(config_name: str, overrides: list[str], batch_size: int):
    config_dir = str((Path(__file__).resolve().parents[1] / "conf").resolve())
    cleaned = clean_overrides(overrides)
    defaults = [
        f"data.batch_size={int(batch_size)}",
        "data.require_teacher_cache=false",
        "data.missing_teacher_policy=keep",
        "keras_model.visualkeras.enabled=false",
    ]
    existing_keys = {item.split("=", 1)[0] for item in cleaned if "=" in item}
    final_overrides = [
        item for item in defaults if item.split("=", 1)[0] not in existing_keys
    ] + cleaned
    with hydra.initialize_config_dir(version_base=None, config_dir=config_dir):
        return hydra.compose(config_name=config_name, overrides=final_overrides)


def parse_pairs(values: list[str]) -> list[tuple[str, str]]:
    if not values:
        values = ["people=humans", "horses=stallions"]
    pairs: list[tuple[str, str]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Pair {value!r} must have form SOURCE=SURROGATE.")
        source, surrogate = value.split("=", 1)
        source = " ".join(source.strip().lower().split())
        surrogate = " ".join(surrogate.strip().lower().split())
        if not source or not surrogate:
            raise ValueError(f"Pair {value!r} must have non-empty source and surrogate.")
        pairs.append((source, surrogate))
    return pairs


def prompt_rows_by_item(data: KerasTallyQAData) -> dict[str, dict[str, Any]]:
    payload = torch.load(absolute_path(data.cfg.paths.prompt_embeddings), map_location="cpu")
    rows = {}
    for row in payload["prompt_classes"]:
        rows[" ".join(str(row["item"]).strip().lower().split())] = row
    return rows


def selected_indices(
    data: KerasTallyQAData,
    split: str,
    prompt: str,
    max_examples: int | None,
) -> list[int]:
    indices = [
        index
        for index in data.indices[split]
        if " ".join(str(data.rows[index]["student_prompt"]).strip().lower().split()) == prompt
    ]
    if max_examples is not None:
        indices = indices[: max(0, int(max_examples))]
    return indices


def load_student(cfg, data, weights: Path) -> tuple[Any, str]:
    prompt_length = int(data.prompt_token_ids.shape[1])
    return build_student_for_weights(
        cfg,
        data,
        prompt_length,
        weights,
        "auto",
        False,
    )


def surrogate_embeddings_via_helper(
    args: argparse.Namespace,
    teacher_model_name: str,
    texts: list[str],
) -> dict[str, dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="prompt-surrogate-") as tmpdir:
        request_path = Path(tmpdir) / "request.json"
        output_path = Path(tmpdir) / "output.json"
        request_path.write_text(
            json.dumps({"model": teacher_model_name, "texts": texts}),
            encoding="utf-8",
        )
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--embedding-helper-json",
            str(request_path),
            "--embedding-helper-output",
            str(output_path),
            "--model",
            teacher_model_name,
            "--torch-dtype",
            str(args.torch_dtype),
        ]
        cmd.append("--local-files-only" if args.local_files_only else "--no-local-files-only")
        cmd.append("--trust-remote-code" if args.trust_remote_code else "--no-trust-remote-code")
        subprocess.run(cmd, check=True)
        return json.loads(output_path.read_text(encoding="utf-8"))


def patch_prompt_embedding(
    model: tf.keras.Model,
    source_compact_ids: list[int],
    surrogate_mean: np.ndarray,
) -> None:
    layer = model.get_layer("compact_prompt_embedding")
    weights = layer.get_weights()
    if len(weights) != 1:
        raise ValueError("compact_prompt_embedding should have exactly one weight tensor.")
    table = weights[0].copy()
    for compact_id in source_compact_ids:
        table[int(compact_id)] = surrogate_mean
    layer.set_weights([table])


def evaluate(
    model: tf.keras.Model,
    data: KerasTallyQAData,
    indices: list[int],
    batch_size: int,
    description: str,
) -> dict[str, Any]:
    accumulator = MulticlassAccumulator(data.num_classes)
    label_counts: Counter[int] = Counter()
    for start in tqdm(
        range(0, len(indices), batch_size),
        desc=description,
        unit="batch",
        dynamic_ncols=True,
        leave=False,
    ):
        batch_indices = indices[start : start + batch_size]
        inputs, targets = data._batch_from_indices(batch_indices)
        logits = model.predict_on_batch(inputs)
        labels = targets["labels"].astype(np.int32)
        prompts = [str(data.rows[index]["student_prompt"]) for index in batch_indices]
        accumulator.update(labels, np.asarray(logits), prompts)
        label_counts.update(int(label) for label in labels.tolist())
    metrics = accumulator.metrics()
    return {
        "examples": len(indices),
        "label_counts": {str(key): int(label_counts[key]) for key in sorted(label_counts)},
        "metrics": {
            key: float(value)
            for key, value in metrics.items()
            if key
            in {
                "accuracy",
                "within_1_accuracy",
                "mae",
                "class_weighted_accuracy",
                "class_weighted_within_1_accuracy",
                "class_weighted_mae",
            }
        },
        "confusion": accumulator.confusion.astype(int).tolist(),
    }


def main() -> None:
    args = parse_args()
    if args.embedding_helper_json is not None:
        run_embedding_helper(args)
        return

    global hydra
    global tf
    global torch
    global tqdm
    global KerasTallyQAData
    global MulticlassAccumulator
    global absolute_path
    global build_keras_student_model
    global build_student_for_weights

    import hydra as hydra_module
    import tensorflow as tf_module
    import torch as torch_module
    from tqdm import tqdm as tqdm_module

    from scripts.train_tallyqa_keras_student import (
        KerasTallyQAData as KerasTallyQAData_class,
        MulticlassAccumulator as MulticlassAccumulator_class,
        absolute_path as absolute_path_function,
        build_keras_student_model as build_keras_student_model_function,
    )
    from scripts.export_tallyqa_keras_static_prompt_coral import (
        build_student_for_weights as build_student_for_weights_function,
    )

    hydra = hydra_module
    tf = tf_module
    torch = torch_module
    tqdm = tqdm_module
    KerasTallyQAData = KerasTallyQAData_class
    MulticlassAccumulator = MulticlassAccumulator_class
    absolute_path = absolute_path_function
    build_keras_student_model = build_keras_student_model_function
    build_student_for_weights = build_student_for_weights_function

    if args.output.exists() and not args.force:
        raise FileExistsError(f"{args.output} exists. Pass --force to replace it.")
    weights = absolute_path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(weights)

    pairs = parse_pairs(args.pair)
    cfg = compose_cfg(args.config_name, args.overrides, args.batch_size)
    data = KerasTallyQAData(cfg)
    prompt_rows = prompt_rows_by_item(data)
    teacher_model_name = args.model or str(
        torch.load(absolute_path(cfg.paths.prompt_embeddings), map_location="cpu").get(
            "teacher_model",
            "HuggingFaceTB/SmolVLM-256M-Instruct",
        )
    )
    surrogate_payload = surrogate_embeddings_via_helper(
        args,
        teacher_model_name,
        sorted({surrogate for _, surrogate in pairs}),
    )

    results = []
    for source, surrogate in pairs:
        if source not in prompt_rows:
            raise ValueError(f"Source prompt {source!r} not found in prompt embedding artifact.")
        indices = selected_indices(data, args.split, source, args.max_examples_per_prompt)
        if not indices:
            raise ValueError(f"No {args.split!r} rows found for prompt {source!r}.")
        source_row = prompt_rows[source]
        source_compact_ids = [
            int(value)
            for value, active in zip(
                source_row["compact_token_ids"],
                source_row["attention_mask"],
                strict=True,
            )
            if bool(active)
        ]
        surrogate_tokenization = dict(surrogate_payload[surrogate])
        surrogate_mean = np.asarray(
            surrogate_tokenization.pop("mean_embedding"),
            dtype=np.float32,
        )

        baseline_model, baseline_graph = load_student(cfg, data, weights)
        baseline = evaluate(
            baseline_model,
            data,
            indices,
            int(args.batch_size),
            f"{source} baseline",
        )
        tf.keras.backend.clear_session()

        surrogate_model, surrogate_graph = load_student(cfg, data, weights)
        patch_prompt_embedding(surrogate_model, source_compact_ids, surrogate_mean)
        surrogate_result = evaluate(
            surrogate_model,
            data,
            indices,
            int(args.batch_size),
            f"{source}->{surrogate}",
        )
        tf.keras.backend.clear_session()

        deltas = {
            key: surrogate_result["metrics"][key] - baseline["metrics"][key]
            for key in baseline["metrics"]
        }
        results.append(
            {
                "source_prompt": source,
                "surrogate_prompt": surrogate,
                "split": args.split,
                "source_prompt_row": {
                    "class_id": int(source_row["class_id"]),
                    "item": str(source_row["item"]),
                    "teacher_token_ids": [int(value) for value in source_row["teacher_token_ids"]],
                    "teacher_tokens": [str(value) for value in source_row["teacher_tokens"]],
                    "compact_token_ids": source_compact_ids,
                },
                "surrogate_tokenization": surrogate_tokenization,
                "method": (
                    "The source prompt's active compact embedding row(s) are replaced "
                    "in-memory by the mean SmolVLM token embedding of the surrogate text. "
                    "The checkpoint, model topology, selected rows, and token IDs fed to "
                    "the model are otherwise unchanged."
                ),
                "weights_graph": {
                    "baseline": baseline_graph,
                    "surrogate": surrogate_graph,
                },
                "baseline": baseline,
                "surrogate": surrogate_result,
                "delta_surrogate_minus_baseline": {
                    key: float(value) for key, value in deltas.items()
                },
            }
        )

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "weights": str(weights),
        "config_name": args.config_name,
        "config_overrides": clean_overrides(args.overrides),
        "split": args.split,
        "batch_size": int(args.batch_size),
        "max_examples_per_prompt": args.max_examples_per_prompt,
        "teacher_model": teacher_model_name,
        "prompt_embeddings": str(absolute_path(cfg.paths.prompt_embeddings)),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({row["source_prompt"]: row["delta_surrogate_minus_baseline"] for row in results}, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
