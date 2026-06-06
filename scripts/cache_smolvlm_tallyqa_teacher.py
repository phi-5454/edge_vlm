from __future__ import annotations

import argparse
from collections import defaultdict
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

from datasets import load_from_disk
import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import torch
from tqdm.auto import tqdm
import transformers
from transformers import AutoModelForImageTextToText, AutoProcessor


DEFAULT_MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"
DEFAULT_DATASET = Path("data/tallyqa_cauldron_target_mobilenet224")
DEFAULT_OUTPUT = Path("artifacts/teacher_cache/smolvlm_tallyqa_target_mobilenet224.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache SmolVLM TallyQA numeric teacher logits.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--image-source",
        choices=["target", "original"],
        default="target",
        help="Use packed 224x224 target images or original Cauldron TallyQA images.",
    )
    parser.add_argument(
        "--source-dataset",
        type=Path,
        default=Path("data/the_cauldron/tallyqa"),
        help="Original Cauldron TallyQA dataset used when --image-source=original.",
    )
    parser.add_argument("--image-processor-backend", choices=["torchvision", "pil"], default="pil")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--answer-min", type=int, default=0)
    parser.add_argument("--answer-max", type=int, default=15)
    parser.add_argument(
        "--run-mode",
        choices=["full", "calibration"],
        default="full",
        help="Use 'calibration' to cache a deterministic balanced subset for model comparison.",
    )
    parser.add_argument(
        "--calibration-examples",
        type=int,
        default=4096,
        help="Target number of examples when --run-mode=calibration.",
    )
    parser.add_argument(
        "--calibration-seed",
        type=int,
        default=20260604,
        help="Seed for deterministic calibration subset selection.",
    )
    parser.add_argument(
        "--calibration-collapse-at",
        type=int,
        default=5,
        help="Answer value where calibration balancing collapses counts into a '<n>+' bucket.",
    )
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--decode-workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--prefetch-batches", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cudnn-benchmark", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--flush-every", type=int, default=1)
    parser.add_argument("--log-timing-every", type=int, default=25)
    parser.add_argument("--synchronize-timing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def device_from_arg(value: str) -> str:
    if value != "auto":
        return value
    return "cuda" if torch.cuda.is_available() else "cpu"


def dtype_from_arg(value: str, device: str) -> torch.dtype:
    if value == "auto":
        return torch.float16 if device == "cuda" else torch.float32
    return getattr(torch, value)


def configure_runtime(args: argparse.Namespace, device: str) -> None:
    if args.require_cuda and device != "cuda":
        raise RuntimeError(
            "--require-cuda was set, but CUDA is not available. "
            "Check the Colab runtime accelerator before starting the cache job."
        )
    torch.set_num_threads(args.cpu_threads)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)
        torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)
        torch.set_float32_matmul_precision("high")


def runtime_metadata(device: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        metadata["cuda_devices"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
                "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
            }
            for index in range(torch.cuda.device_count())
        ]
    return metadata


def cached_model_revision(model_name: str) -> str | None:
    model_cache_name = "models--" + model_name.replace("/", "--")
    cache_roots = []
    if os.environ.get("HF_HUB_CACHE"):
        cache_roots.append(Path(os.environ["HF_HUB_CACHE"]))
    if os.environ.get("HF_HOME"):
        cache_roots.append(Path(os.environ["HF_HOME"]) / "hub")
    cache_roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    for cache_root in cache_roots:
        ref_path = cache_root / model_cache_name / "refs" / "main"
        if ref_path.exists():
            return ref_path.read_text(encoding="utf-8").strip()
    return None


def synchronize_if_requested(device: str, enabled: bool) -> None:
    if enabled and device == "cuda":
        torch.cuda.synchronize()


def load_processor(args: argparse.Namespace) -> AutoProcessor:
    processor_kwargs = {
        "local_files_only": args.local_files_only,
        "trust_remote_code": args.trust_remote_code,
        "backend": args.image_processor_backend,
    }
    try:
        return AutoProcessor.from_pretrained(args.model, **processor_kwargs)
    except AttributeError as exc:
        if "backend" not in str(exc):
            raise
        processor_kwargs.pop("backend")
        return AutoProcessor.from_pretrained(args.model, **processor_kwargs)


@dataclass
class TimingAccumulator:
    batches: int = 0
    records: int = 0
    wait_for_prepared_batch_s: float = 0.0
    move_to_device_s: float = 0.0
    teacher_forward_s: float = 0.0
    continuation_forward_s: float = 0.0
    record_build_write_s: float = 0.0
    progress_update_s: float = 0.0
    loop_s: float = 0.0

    def add_batch(
        self,
        records: int,
        wait_for_prepared_batch_s: float,
        move_to_device_s: float,
        teacher_forward_s: float,
        continuation_forward_s: float,
        record_build_write_s: float,
        progress_update_s: float,
        loop_s: float,
    ) -> None:
        self.batches += 1
        self.records += records
        self.wait_for_prepared_batch_s += wait_for_prepared_batch_s
        self.move_to_device_s += move_to_device_s
        self.teacher_forward_s += teacher_forward_s
        self.continuation_forward_s += continuation_forward_s
        self.record_build_write_s += record_build_write_s
        self.progress_update_s += progress_update_s
        self.loop_s += loop_s

    def summary(self, synchronized: bool) -> dict[str, Any]:
        total = self.loop_s

        def fraction(value: float) -> float | None:
            return value / total if total > 0 else None

        return {
            "synchronized_cuda_timing": synchronized,
            "batches": self.batches,
            "records": self.records,
            "records_per_second": self.records / total if total > 0 else None,
            "seconds": {
                "loop": self.loop_s,
                "wait_for_prepared_batch": self.wait_for_prepared_batch_s,
                "move_to_device": self.move_to_device_s,
                "teacher_forward": self.teacher_forward_s,
                "continuation_forward": self.continuation_forward_s,
                "record_build_write": self.record_build_write_s,
                "progress_update": self.progress_update_s,
            },
            "fractions": {
                "wait_for_prepared_batch": fraction(self.wait_for_prepared_batch_s),
                "move_to_device": fraction(self.move_to_device_s),
                "teacher_forward": fraction(self.teacher_forward_s),
                "continuation_forward": fraction(self.continuation_forward_s),
                "record_build_write": fraction(self.record_build_write_s),
                "progress_update": fraction(self.progress_update_s),
            },
        }


def chat_prompt(processor: Any, question: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }
    ]
    processor_kwargs: dict[str, Any] = {}
    video_processor = getattr(processor, "video_processor", None)
    if video_processor is not None:
        processor_kwargs["num_frames"] = getattr(video_processor, "num_frames", None)
        processor_kwargs["fps"] = getattr(video_processor, "fps", None)
    return processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        processor_kwargs=processor_kwargs,
    )


def encode_processor_batch(
    processor: Any,
    chat_texts: list[str],
    images: list[Image.Image],
) -> Any:
    nested_images = [[image] for image in images]
    call_variants = [
        {"processor_kwargs": {"text_kwargs": {"return_tensors": "pt", "padding": True}}},
        {"text_kwargs": {"return_tensors": "pt", "padding": True}},
        {"return_tensors": "pt", "padding": True},
    ]
    last_error: Exception | None = None
    for kwargs in call_variants:
        try:
            inputs = processor(text=chat_texts, images=nested_images, **kwargs)
        except TypeError as exc:
            last_error = exc
            continue
        input_ids = inputs.get("input_ids") if hasattr(inputs, "get") else None
        if isinstance(input_ids, torch.Tensor):
            return inputs
    if last_error is not None:
        raise last_error
    raise TypeError("Processor did not return tensor input_ids for any supported call form.")



def load_examples(dataset_path: Path) -> list[dict[str, Any]]:
    examples_path = dataset_path / "examples.parquet"
    if not examples_path.exists():
        raise FileNotFoundError(f"{examples_path} not found")
    return pq.read_table(examples_path).to_pylist()


def load_metadata(dataset_path: Path) -> dict[str, Any]:
    metadata_path = dataset_path / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"{metadata_path} not found")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


class Uint8ImageStore:
    def __init__(self, dataset_path: Path, metadata: dict[str, Any]) -> None:
        image_meta = metadata["image"]
        shape = tuple(int(dim) for dim in image_meta["shape"])
        if len(shape) != 4 or shape[1:] != (3, 224, 224):
            raise ValueError(f"Expected CHW MobileNet uint8 image shape [N,3,224,224], got {shape}")
        tensor_path = dataset_path / image_meta.get("tensor_file", "images.uint8.bin")
        index_path = dataset_path / image_meta.get("index_file", "images.index.jsonl")
        if not tensor_path.exists():
            raise FileNotFoundError(f"{tensor_path} not found")
        if not index_path.exists():
            raise FileNotFoundError(f"{index_path} not found")
        self.tensor_path = tensor_path
        self.index_path = index_path
        self.shape = shape
        self.images = np.memmap(tensor_path, dtype=np.uint8, mode="r", shape=shape)
        self.index_rows = self._load_index_rows(index_path)

    @staticmethod
    def _load_index_rows(index_path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with index_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    def get(self, image_index: int) -> tuple[Image.Image, dict[str, Any]]:
        chw = np.asarray(self.images[image_index])
        hwc = np.transpose(chw, (1, 2, 0))
        image = Image.fromarray(hwc, mode="RGB")
        index_row = self.index_rows[image_index]
        return image, {
            "image_source": "tallyqa-target-mobilenet224-uint8",
            "image_index": int(image_index),
            "image_id": index_row.get("image_id"),
            "source_row_index": index_row.get("source_row_index"),
            "image_shape_chw": [3, 224, 224],
            "image_dtype": "uint8",
            "image_layout": "CHW",
            "image_tensor_file": str(self.tensor_path),
            "image_index_file": str(self.index_path),
        }


class OriginalImageStore:
    def __init__(self, source_dataset_path: Path) -> None:
        if not source_dataset_path.exists():
            raise FileNotFoundError(f"{source_dataset_path} not found")
        self.source_dataset_path = source_dataset_path
        self.source_dataset = load_from_disk(str(source_dataset_path))

    def get(self, row: dict[str, Any]) -> tuple[Image.Image, dict[str, Any]]:
        source_row_index = int(row["source_row_index"])
        source_row = self.source_dataset[source_row_index]
        image = source_row["images"][0].convert("RGB")
        return image, {
            "image_source": "original-cauldron-tallyqa",
            "source_dataset": str(self.source_dataset_path),
            "source_row_index": source_row_index,
            "image_slot": 0,
            "image_size": list(image.size),
            "image_mode": image.mode,
        }


def stable_sort_key(seed: int, *parts: object) -> str:
    joined = ":".join([str(seed), *(str(part) for part in parts)])
    return hashlib.blake2b(joined.encode("utf-8"), digest_size=16).hexdigest()


def full_run_indices(dataset_len: int, args: argparse.Namespace) -> list[int]:
    stop = dataset_len if args.end_index is None else min(dataset_len, args.end_index)
    if args.max_examples is not None:
        stop = min(stop, args.start_index + args.max_examples)
    return [
        index
        for index in range(args.start_index, stop)
        if index % args.shard_count == args.shard_index
    ]


def calibration_indices(examples: list[dict[str, Any]], args: argparse.Namespace) -> list[int]:
    stop = len(examples) if args.end_index is None else min(len(examples), args.end_index)
    eligible_indices = list(range(args.start_index, stop))
    if args.max_examples is not None:
        eligible_indices = eligible_indices[: args.max_examples]
    target_count = min(args.calibration_examples, len(eligible_indices))
    buckets: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index in eligible_indices:
        row = examples[index]
        answer = int(row["answer"])
        output_bucket = min(answer, args.calibration_collapse_at)
        buckets[(str(row["student_prompt"]), output_bucket)].append(index)

    for bucket_key, bucket_indices in buckets.items():
        bucket_indices.sort(
            key=lambda index: stable_sort_key(
                args.calibration_seed,
                bucket_key[0],
                bucket_key[1],
                index,
            )
        )

    active_keys = sorted(
        buckets,
        key=lambda key: stable_sort_key(args.calibration_seed, key[0], key[1]),
    )
    selected: list[int] = []
    while active_keys and len(selected) < target_count:
        next_active_keys: list[tuple[str, int]] = []
        for key in active_keys:
            if len(selected) >= target_count:
                next_active_keys.append(key)
                continue
            bucket = buckets[key]
            if bucket:
                selected.append(bucket.pop(0))
            if bucket:
                next_active_keys.append(key)
        active_keys = next_active_keys

    selected = [
        index
        for index in selected
        if index % args.shard_count == args.shard_index
    ]
    return selected


def planned_indices(examples: list[dict[str, Any]], args: argparse.Namespace) -> list[int]:
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.decode_workers <= 0:
        raise ValueError("--decode-workers must be positive")
    if args.prefetch_batches <= 0:
        raise ValueError("--prefetch-batches must be positive")
    if args.cpu_threads <= 0:
        raise ValueError("--cpu-threads must be positive")
    if args.log_timing_every < 0:
        raise ValueError("--log-timing-every must be non-negative")
    if args.shard_count <= 0:
        raise ValueError("--shard-count must be positive")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("--shard-index must be in [0, shard_count)")
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative")
    if args.end_index is not None and args.end_index < args.start_index:
        raise ValueError("--end-index must be greater than or equal to --start-index")
    if args.max_examples is not None and args.max_examples < 0:
        raise ValueError("--max-examples must be non-negative")
    if args.answer_min > args.answer_max:
        raise ValueError("--answer-min must be <= --answer-max")
    if args.calibration_examples <= 0:
        raise ValueError("--calibration-examples must be positive")
    if args.calibration_collapse_at < 0:
        raise ValueError("--calibration-collapse-at must be non-negative")

    if args.run_mode == "calibration":
        return calibration_indices(examples, args)
    return full_run_indices(len(examples), args)


@dataclass
class MetricAccumulator:
    count: int = 0
    correct: float = 0.0
    nll: float = 0.0
    target_probability: float = 0.0
    target_full_vocab_probability: float = 0.0

    def update(self, metrics: dict[str, Any]) -> None:
        self.count += 1
        self.correct += float(metrics["correct"])
        self.nll += float(metrics["nll"])
        self.target_probability += float(metrics["target_probability"])
        self.target_full_vocab_probability += float(metrics["target_full_vocab_probability"])

    def aggregate(self) -> dict[str, Any]:
        if self.count == 0:
            return {
                "records": 0,
                "accuracy": None,
                "mean_nll": None,
                "mean_target_probability": None,
                "mean_target_full_vocab_probability": None,
            }
        return {
            "records": self.count,
            "accuracy": self.correct / self.count,
            "mean_nll": self.nll / self.count,
            "mean_target_probability": self.target_probability / self.count,
            "mean_target_full_vocab_probability": (
                self.target_full_vocab_probability / self.count
            ),
        }


def completed_indices_and_metrics(
    path: Path,
    selected_indices: set[int],
) -> tuple[set[int], MetricAccumulator, dict[str, MetricAccumulator]]:
    completed: set[int] = set()
    overall = MetricAccumulator()
    by_prompt: dict[str, MetricAccumulator] = defaultdict(MetricAccumulator)
    if not path.exists():
        return completed, overall, by_prompt
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_number}") from exc
            dataset_index = int(payload["dataset_index"])
            completed.add(dataset_index)
            if dataset_index not in selected_indices:
                continue
            metrics = payload["teacher_metrics"]["numeric_answer"]
            student_prompt = str(payload["student_prompt"])
            overall.update(metrics)
            by_prompt[student_prompt].update(metrics)
    return completed, overall, by_prompt


def tensor_metadata(tensor: torch.Tensor | None) -> dict[str, Any] | None:
    if tensor is None:
        return None
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
    }


def topk_payload(logits: torch.Tensor, processor: Any, k: int) -> list[dict[str, Any]]:
    values, indices = torch.topk(logits.detach().float().cpu(), k=k)
    tokens = processor.tokenizer.convert_ids_to_tokens(indices.tolist())
    return [
        {
            "rank": rank,
            "token_id": int(token_id),
            "token": token,
            "logit": float(logit),
        }
        for rank, (token_id, token, logit) in enumerate(
            zip(indices.tolist(), tokens, values.tolist(), strict=True),
            start=1,
        )
    ]


def answer_candidates(processor: Any, answer_min: int, answer_max: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for answer in range(answer_min, answer_max + 1):
        text = str(answer)
        token_ids = processor.tokenizer(text, add_special_tokens=False)["input_ids"]
        candidates.append(
            {
                "answer": answer,
                "text": text,
                "token_ids": [int(token_id) for token_id in token_ids],
                "tokens": processor.tokenizer.convert_ids_to_tokens(token_ids),
            }
        )
    return candidates


def candidate_prefixes(candidates: list[dict[str, Any]]) -> list[tuple[int, ...]]:
    prefixes: set[tuple[int, ...]] = set()
    for candidate in candidates:
        token_ids = candidate["token_ids"]
        for length in range(1, len(token_ids)):
            prefixes.add(tuple(int(token_id) for token_id in token_ids[:length]))
    return sorted(prefixes, key=lambda prefix: (len(prefix), prefix))


def continuation_logits_for_prefixes(
    model: Any,
    inputs: dict[str, torch.Tensor],
    sequence_lengths: torch.Tensor,
    prefixes: list[tuple[int, ...]],
) -> dict[tuple[int, ...], torch.Tensor]:
    if not prefixes:
        return {}
    pad_token_id = safe_text_pad_token_id(model)
    continuation_logits: dict[tuple[int, ...], torch.Tensor] = {}
    batch_size = int(inputs["input_ids"].shape[0])
    device = inputs["input_ids"].device
    pixel_attention_mask = inputs.get("pixel_attention_mask")

    for prefix in prefixes:
        prefix_tensor = torch.tensor(prefix, dtype=inputs["input_ids"].dtype, device=device)
        prefix_length = len(prefix)
        max_length = int(sequence_lengths.max().item()) + prefix_length
        extended_input_ids = torch.full(
            (batch_size, max_length),
            pad_token_id,
            dtype=inputs["input_ids"].dtype,
            device=device,
        )
        extended_attention_mask = torch.zeros(
            (batch_size, max_length),
            dtype=inputs["attention_mask"].dtype,
            device=device,
        )
        for batch_offset in range(batch_size):
            sequence_length = int(sequence_lengths[batch_offset].item())
            extended_length = sequence_length + prefix_length
            extended_input_ids[batch_offset, :sequence_length] = inputs["input_ids"][
                batch_offset, :sequence_length
            ]
            extended_input_ids[batch_offset, sequence_length:extended_length] = prefix_tensor
            extended_attention_mask[batch_offset, :extended_length] = 1

        model_inputs = {
            "input_ids": extended_input_ids,
            "attention_mask": extended_attention_mask,
            "pixel_values": inputs["pixel_values"],
        }
        if pixel_attention_mask is not None:
            model_inputs["pixel_attention_mask"] = pixel_attention_mask
        validate_input_ids(model, extended_input_ids, f"continuation prefix {prefix}")
        with torch.inference_mode():
            outputs = model(**model_inputs, use_cache=False)
        logits = []
        for batch_offset in range(batch_size):
            position = int(sequence_lengths[batch_offset].item()) + prefix_length - 1
            logits.append(outputs.logits[batch_offset, position].float())
        continuation_logits[prefix] = torch.stack(logits, dim=0)
    return continuation_logits


def safe_text_pad_token_id(model: Any) -> int:
    embedding = model.get_input_embeddings()
    vocab_size = int(embedding.num_embeddings) if embedding is not None else 0
    candidates = [
        getattr(getattr(model.config, "text_config", None), "pad_token_id", None),
        getattr(model.config, "pad_token_id", None),
        0,
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        token_id = int(candidate)
        if 0 <= token_id < vocab_size:
            return token_id
    raise ValueError(f"Could not find a valid pad token ID for embedding size {vocab_size}.")


def validate_input_ids(model: Any, input_ids: torch.Tensor, context: str) -> None:
    embedding = model.get_input_embeddings()
    if embedding is None:
        return
    vocab_size = int(embedding.num_embeddings)
    min_id = int(input_ids.detach().min().cpu().item())
    max_id = int(input_ids.detach().max().cpu().item())
    if min_id < 0 or max_id >= vocab_size:
        raise ValueError(
            f"{context} input_ids contain IDs outside embedding range: "
            f"min={min_id}, max={max_id}, embedding_rows={vocab_size}. "
            "This would trigger a CUDA device-side assert."
        )


def numeric_answer_metrics(
    next_logits: torch.Tensor,
    continuation_logits: dict[tuple[int, ...], torch.Tensor],
    candidates: list[dict[str, Any]],
    hard_answer: int,
    temperature: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sequence_log_likelihoods: list[float] = []
    token_step_scores: list[list[dict[str, Any]]] = []
    answers = [int(candidate["answer"]) for candidate in candidates]
    if hard_answer not in answers:
        raise ValueError(f"Hard answer {hard_answer} not in configured answer range")

    for candidate in candidates:
        token_ids = [int(token_id) for token_id in candidate["token_ids"]]
        step_scores: list[dict[str, Any]] = []
        step_log_likelihoods: list[float] = []
        for step_index, token_id in enumerate(token_ids):
            logits = (
                next_logits
                if step_index == 0
                else continuation_logits[tuple(token_ids[:step_index])]
            )
            log_probs = torch.log_softmax(logits.float() / temperature, dim=-1)
            probs = torch.softmax(logits.float() / temperature, dim=-1)
            log_likelihood = float(log_probs[token_id].detach().cpu())
            step_log_likelihoods.append(log_likelihood)
            step_scores.append(
                {
                    "step": step_index,
                    "token_id": token_id,
                    "token": candidate["tokens"][step_index],
                    "full_vocab_log_likelihood": log_likelihood,
                    "full_vocab_probability": float(probs[token_id].detach().cpu()),
                }
            )
        sequence_log_likelihoods.append(sum(step_log_likelihoods))
        token_step_scores.append(step_scores)

    sequence_scores = torch.tensor(sequence_log_likelihoods, dtype=torch.float32)
    candidate_log_probs = torch.log_softmax(sequence_scores, dim=0)
    candidate_probs = torch.softmax(sequence_scores, dim=0)
    target_index = answers.index(hard_answer)
    prediction_index = int(torch.argmax(candidate_probs).item())
    prediction = answers[prediction_index]

    scores: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        scores.append(
            {
                **candidate,
                "candidate_log_likelihood": float(candidate_log_probs[index].detach().cpu()),
                "candidate_probability": float(candidate_probs[index].detach().cpu()),
                "full_vocab_sequence_log_likelihood": sequence_log_likelihoods[index],
                "full_vocab_sequence_probability": float(np.exp(sequence_log_likelihoods[index])),
                "token_step_scores": token_step_scores[index],
            }
        )

    metrics = {
        "prediction": prediction,
        "prediction_text": str(prediction),
        "correct": prediction == hard_answer,
        "target_probability": float(candidate_probs[target_index].detach().cpu()),
        "target_full_vocab_probability": float(np.exp(sequence_log_likelihoods[target_index])),
        "nll": float(-candidate_log_probs[target_index].detach().cpu()),
    }
    return metrics, scores


def prepare_batch(
    batch_indices: list[int],
    examples: list[dict[str, Any]],
    image_store: Uint8ImageStore | OriginalImageStore,
    processor: Any,
    decode_executor: ThreadPoolExecutor,
) -> dict[str, Any]:
    rows = [examples[index] for index in batch_indices]
    if isinstance(image_store, OriginalImageStore):
        images_and_identities = list(decode_executor.map(image_store.get, rows))
    else:
        images_and_identities = list(
            decode_executor.map(lambda row: image_store.get(int(row["image_index"])), rows)
        )
    images = [image for image, _ in images_and_identities]
    teacher_prompts = [str(row["teacher_prompt"]) for row in rows]
    chat_texts = [chat_prompt(processor, teacher_prompt) for teacher_prompt in teacher_prompts]
    return {
        "batch_indices": batch_indices,
        "rows": rows,
        "images": images,
        "image_identities": [identity for _, identity in images_and_identities],
        "teacher_prompts": teacher_prompts,
        "chat_texts": chat_texts,
        "inputs": encode_processor_batch(processor, chat_texts, images),
    }


def submit_prepare_batch(
    executor: ThreadPoolExecutor,
    batch_indices: list[int],
    examples: list[dict[str, Any]],
    image_store: Uint8ImageStore | OriginalImageStore,
    processor: Any,
    decode_executor: ThreadPoolExecutor,
) -> Future[dict[str, Any]]:
    return executor.submit(
        prepare_batch,
        batch_indices,
        examples,
        image_store,
        processor,
        decode_executor,
    )


def aggregate_by_prompt(
    by_prompt: dict[str, MetricAccumulator],
    examples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    prompt_to_class: dict[str, dict[str, Any]] = {}
    for row in examples:
        student_prompt = str(row["student_prompt"])
        prompt_to_class.setdefault(
            student_prompt,
            {
                "student_prompt": student_prompt,
                "item": str(row["item"]),
                "item_class_id": int(row["item_class_id"]),
            },
        )
    rows: list[dict[str, Any]] = []
    for student_prompt, accumulator in by_prompt.items():
        rows.append({**prompt_to_class.get(student_prompt, {"student_prompt": student_prompt}), **accumulator.aggregate()})
    return sorted(rows, key=lambda row: (-int(row["records"]), str(row["student_prompt"])))


def main() -> None:
    args = parse_args()
    if args.force and args.output.exists():
        args.output.unlink()
    if args.output.exists() and not args.force and not args.resume:
        raise FileExistsError(f"{args.output} exists. Pass --force to replace it.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output.with_suffix(".manifest.json")

    metadata = load_metadata(args.dataset)
    examples = load_examples(args.dataset)
    selected_indices = planned_indices(examples, args)
    selected_set = set(selected_indices)
    completed, overall_metrics, by_prompt_metrics = completed_indices_and_metrics(
        args.output, selected_set
    ) if args.resume else (set(), MetricAccumulator(), defaultdict(MetricAccumulator))
    resumed_indices = selected_set & completed
    indices = [index for index in selected_indices if index not in completed]

    if args.resume:
        print(
            f"Resume scan: output={args.output} selected={len(selected_indices)} "
            f"completed={len(resumed_indices)} remaining={len(indices)}"
        )
    if args.dry_run:
        device = device_from_arg(args.device)
        configure_runtime(args, device)
        print(
            json.dumps(
                {
                    "dataset": str(args.dataset),
                    "dataset_rows": len(examples),
                    "selected_records": len(selected_indices),
                    "planned_records_this_invocation": len(indices),
                    "skipped_existing_records": len(resumed_indices),
                    "first_index": indices[0] if indices else None,
                    "last_index": indices[-1] if indices else None,
                    "shard_count": args.shard_count,
                    "shard_index": args.shard_index,
                    "batch_size": args.batch_size,
                    "prefetch_batches": args.prefetch_batches,
                    "decode_workers": args.decode_workers,
                    "cpu_threads": args.cpu_threads,
                    "answer_range": [args.answer_min, args.answer_max],
                    "run_mode": args.run_mode,
                    "calibration": {
                        "examples": args.calibration_examples,
                        "seed": args.calibration_seed,
                        "collapse_at": args.calibration_collapse_at,
                    } if args.run_mode == "calibration" else None,
                    "output": str(args.output),
                    "runtime": runtime_metadata(device),
                    "metrics": {
                        "overall": "written to manifest teacher_metrics.numeric_answer",
                        "by_student_prompt": "written to manifest teacher_metrics.by_student_prompt",
                    },
                },
                indent=2,
            )
        )
        return

    device = device_from_arg(args.device)
    dtype = dtype_from_arg(args.torch_dtype, device)
    configure_runtime(args, device)
    processor = load_processor(args)
    processor.tokenizer.padding_side = "right"
    candidates = answer_candidates(processor, args.answer_min, args.answer_max)
    prefixes = candidate_prefixes(candidates)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    model_revision = cached_model_revision(args.model)
    if args.image_source == "original":
        image_store: Uint8ImageStore | OriginalImageStore = OriginalImageStore(args.source_dataset)
    else:
        image_store = Uint8ImageStore(args.dataset, metadata)

    records_written = 0
    selected_examples = len(selected_indices)
    total_examples = len(indices)
    output_mode = "a" if args.resume else "w"
    decode_executor = ThreadPoolExecutor(max_workers=args.decode_workers)
    prefetch_executor = ThreadPoolExecutor(max_workers=args.prefetch_batches)
    timing = TimingAccumulator()

    with args.output.open(output_mode, encoding="utf-8") as handle:
        progress = tqdm(
            total=selected_examples,
            initial=len(resumed_indices),
            desc="Caching TallyQA teacher logits",
            unit="example",
        )
        batch_indices_list = [
            indices[start : start + args.batch_size]
            for start in range(0, total_examples, args.batch_size)
        ]
        pending: deque[Future[dict[str, Any]]] = deque()
        next_batch_to_submit = 0

        def fill_prefetch_queue() -> None:
            nonlocal next_batch_to_submit
            while (
                next_batch_to_submit < len(batch_indices_list)
                and len(pending) < args.prefetch_batches
            ):
                pending.append(
                    submit_prepare_batch(
                        prefetch_executor,
                        batch_indices_list[next_batch_to_submit],
                        examples,
                        image_store,
                        processor,
                        decode_executor,
                    )
                )
                next_batch_to_submit += 1

        fill_prefetch_queue()
        for batch_number in range(len(batch_indices_list)):
            batch_loop_start = time.perf_counter()
            wait_start = time.perf_counter()
            future = pending.popleft()
            prepared = future.result()
            wait_for_prepared_batch_s = time.perf_counter() - wait_start
            fill_prefetch_queue()
            batch_indices = prepared["batch_indices"]
            rows = prepared["rows"]
            images = prepared["images"]
            image_identities = prepared["image_identities"]
            teacher_prompts = prepared["teacher_prompts"]
            chat_texts = prepared["chat_texts"]
            inputs = prepared["inputs"]

            transfer_start = time.perf_counter()
            inputs = {key: value.to(device) for key, value in inputs.items()}
            validate_input_ids(model, inputs["input_ids"], "teacher prompt batch")
            synchronize_if_requested(device, args.synchronize_timing)
            move_to_device_s = time.perf_counter() - transfer_start

            teacher_forward_start = time.perf_counter()
            with torch.inference_mode():
                outputs = model(**inputs, use_cache=False)
            synchronize_if_requested(device, args.synchronize_timing)
            teacher_forward_s = time.perf_counter() - teacher_forward_start

            sequence_lengths = inputs["attention_mask"].sum(dim=1)
            continuation_start = time.perf_counter()
            continuation_logits_by_prefix = continuation_logits_for_prefixes(
                model=model,
                inputs=inputs,
                sequence_lengths=sequence_lengths,
                prefixes=prefixes,
            )
            synchronize_if_requested(device, args.synchronize_timing)
            continuation_forward_s = time.perf_counter() - continuation_start

            record_start = time.perf_counter()
            for batch_offset, dataset_index in enumerate(batch_indices):
                row = rows[batch_offset]
                image = images[batch_offset]
                image_identity = image_identities[batch_offset]
                teacher_prompt = teacher_prompts[batch_offset]
                chat_text = chat_texts[batch_offset]
                sequence_length = int(sequence_lengths[batch_offset].item())
                next_logits = outputs.logits[batch_offset, sequence_length - 1].float()
                hard_answer = int(row["answer"])
                row_continuation_logits = {
                    prefix: logits[batch_offset]
                    for prefix, logits in continuation_logits_by_prefix.items()
                }
                metrics, candidate_scores = numeric_answer_metrics(
                    next_logits=next_logits,
                    continuation_logits=row_continuation_logits,
                    candidates=candidates,
                    hard_answer=hard_answer,
                    temperature=args.temperature,
                )
                student_prompt = str(row["student_prompt"])
                overall_metrics.update(metrics)
                by_prompt_metrics[student_prompt].update(metrics)

                input_ids = inputs["input_ids"][batch_offset, :sequence_length].detach().cpu().tolist()
                attention_mask = (
                    inputs["attention_mask"][batch_offset, :sequence_length].detach().cpu().tolist()
                )
                pixel_attention_mask = inputs.get("pixel_attention_mask")
                record = {
                    "cache_schema_version": 1,
                    "dataset_index": dataset_index,
                    "example_id": row["example_id"],
                    "source_subset": row["source_subset"],
                    "source": row["source"],
                    "source_row_index": int(row["source_row_index"]),
                    "qa_index": int(row["qa_index"]),
                    "answer": hard_answer,
                    "answer_text": row["answer_text"],
                    "teacher_prompt": teacher_prompt,
                    "teacher_prompt_clean": row["teacher_prompt_clean"],
                    "student_prompt": student_prompt,
                    "item": row["item"],
                    "item_class_id": int(row["item_class_id"]),
                    "matched_suffix": row["matched_suffix"],
                    "image_id": row["image_id"],
                    "image_index": int(row["image_index"]),
                    "input_identity": {
                        "filtered_dataset": str(args.dataset),
                        "image_source": image_identity["image_source"],
                        "model": args.model,
                        "model_revision": model_revision,
                        "teacher_prompt_sha256": hashlib.sha256(
                            teacher_prompt.encode("utf-8")
                        ).hexdigest(),
                        "student_prompt_sha256": hashlib.sha256(
                            student_prompt.encode("utf-8")
                        ).hexdigest(),
                        "image_identity": image_identity,
                    },
                    "teacher_input": {
                        "input_ids": input_ids,
                        "tokens": processor.tokenizer.convert_ids_to_tokens(input_ids),
                        "attention_mask": attention_mask,
                        "chat_template_text": chat_text,
                    },
                    "image_preprocessing": {
                        "cached_image_size": list(image.size),
                        "cached_image_mode": image.mode,
                        "image_source": image_identity["image_source"],
                        "image_identity": image_identity,
                        "pixel_values": tensor_metadata(inputs["pixel_values"][batch_offset : batch_offset + 1]),
                        "pixel_attention_mask": tensor_metadata(
                            pixel_attention_mask[batch_offset : batch_offset + 1]
                            if pixel_attention_mask is not None
                            else None
                        ),
                        "image_token_id": int(model.config.image_token_id),
                        "image_token_count": int(
                            (
                                inputs["input_ids"][batch_offset, :sequence_length]
                                == int(model.config.image_token_id)
                            ).sum().item()
                        ),
                    },
                    "teacher_logits": {
                        "temperature": args.temperature,
                        "top_k": topk_payload(next_logits, processor, args.top_k),
                        "numeric_answer_candidates": candidate_scores,
                    },
                    "teacher_metrics": {
                        "numeric_answer": metrics,
                        "metric_definitions": {
                            "accuracy": "argmax over configured numeric answer candidate logits equals normalized integer answer",
                            "nll": "negative log-likelihood of normalized integer answer under softmax over configured numeric answer candidates",
                            "target_probability": "probability of normalized integer answer under softmax over configured numeric answer candidates",
                            "target_full_vocab_probability": "full-vocabulary autoregressive sequence probability of the normalized integer answer",
                        },
                    },
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                records_written += 1
                if args.flush_every > 0 and records_written % args.flush_every == 0:
                    handle.flush()
            record_build_write_s = time.perf_counter() - record_start

            progress_start = time.perf_counter()
            progress.update(len(batch_indices))
            progress_update_s = time.perf_counter() - progress_start
            loop_s = time.perf_counter() - batch_loop_start
            timing.add_batch(
                records=len(batch_indices),
                wait_for_prepared_batch_s=wait_for_prepared_batch_s,
                move_to_device_s=move_to_device_s,
                teacher_forward_s=teacher_forward_s,
                continuation_forward_s=continuation_forward_s,
                record_build_write_s=record_build_write_s,
                progress_update_s=progress_update_s,
                loop_s=loop_s,
            )
            if args.log_timing_every and (batch_number + 1) % args.log_timing_every == 0:
                timing_summary = timing.summary(args.synchronize_timing)
                fractions = timing_summary["fractions"]
                progress.set_postfix(
                    {
                        "prep_wait": f"{100 * (fractions['wait_for_prepared_batch'] or 0):.0f}%",
                        "fwd": f"{100 * (fractions['teacher_forward'] or 0):.0f}%",
                        "cont": f"{100 * (fractions['continuation_forward'] or 0):.0f}%",
                        "record": f"{100 * (fractions['record_build_write'] or 0):.0f}%",
                    }
                )
        progress.close()
    prefetch_executor.shutdown()
    decode_executor.shutdown()

    by_prompt = aggregate_by_prompt(by_prompt_metrics, examples)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output": str(args.output),
        "records_written_this_invocation": records_written,
        "records_already_done": len(resumed_indices),
        "records_in_metric_aggregate": overall_metrics.count,
        "selected_records": selected_examples,
        "planned_records_this_invocation": total_examples,
        "selection": {
            "run_mode": args.run_mode,
            "calibration": {
                "target_examples": args.calibration_examples,
                "seed": args.calibration_seed,
                "collapse_at": args.calibration_collapse_at,
            } if args.run_mode == "calibration" else None,
            "first_index": selected_indices[0] if selected_indices else None,
            "last_index": selected_indices[-1] if selected_indices else None,
            "selected_indices_sha256": hashlib.sha256(
                json.dumps(selected_indices, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        },
        "timing": timing.summary(args.synchronize_timing),
        "teacher_metrics": {
            "numeric_answer": overall_metrics.aggregate(),
            "by_student_prompt": by_prompt,
            "metric_definitions": {
                "accuracy": "argmax over configured numeric answer candidate logits equals normalized integer answer",
                "mean_nll": "mean negative log-likelihood under softmax over configured numeric answer candidates",
                "mean_target_probability": "mean target probability under softmax over configured numeric answer candidates",
                "mean_target_full_vocab_probability": "mean full-vocabulary autoregressive sequence probability of the target answer",
            },
        },
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "model": args.model,
        "model_revision": model_revision,
        "runtime": runtime_metadata(device),
        "torch_dtype": str(dtype).replace("torch.", ""),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "pid": os.getpid(),
        "answer_candidates": candidates,
        "answer_candidate_prefixes": [list(prefix) for prefix in prefixes],
        "source_dataset_metadata": {
            "examples": metadata.get("examples"),
            "classes": metadata.get("classes"),
            "unique_images": metadata.get("unique_images"),
            "output_root": metadata.get("output_root"),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote cache: {args.output}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
