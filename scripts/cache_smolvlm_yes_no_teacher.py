from __future__ import annotations

import argparse
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
import transformers
import pyarrow.parquet as pq
from datasets import load_dataset, load_from_disk
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor


DEFAULT_MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"
DEFAULT_DATASET = Path("data/the_cauldron_yes_no_vsr_token1000_img512_parquet")
DEFAULT_OUTPUT = Path("artifacts/teacher_cache/smolvlm_yes_no_vsr_token1000_img512.jsonl")
ANSWER_VARIANTS = {
    "yes": ["yes", "Yes", " yes"],
    "no": ["no", "No", " no"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache SmolVLM yes/no teacher logits.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--image-source",
        choices=["student-512"],
        default="student-512",
        help="Use the 512x512 padded student images stored in parquet.",
    )
    parser.add_argument("--split", default="combined")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--image-processor-backend", choices=["torchvision", "pil"], default="torchvision")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--decode-workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--variant-batch-size", type=int, default=10, help=argparse.SUPPRESS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--flush-every", type=int, default=1)
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
    return processor.apply_chat_template(messages, add_generation_prompt=True)


def load_examples_dataset(dataset_path: Path, split: str) -> Any:
    parquet_file = dataset_path / f"{split}.parquet"
    if parquet_file.exists():
        return load_dataset("parquet", data_files=str(parquet_file), split="train")
    if dataset_path.suffix == ".parquet":
        return load_dataset("parquet", data_files=str(dataset_path), split="train")
    return load_from_disk(dataset_path)[split]


class ParquetImageStore:
    def __init__(self, dataset_path: Path) -> None:
        images_parquet = dataset_path / "images.parquet"
        if not images_parquet.exists():
            raise FileNotFoundError(
                f"{images_parquet} not found. Build it with "
                "scripts/build_student_img512_parquet_dataset.py."
            )
        self.parquet = pq.ParquetFile(images_parquet)
        self.row_group_starts: list[int] = []
        offset = 0
        for row_group in range(self.parquet.num_row_groups):
            self.row_group_starts.append(offset)
            offset += self.parquet.metadata.row_group(row_group).num_rows
        ids = pq.read_table(images_parquet, columns=["student_image_id"]).column(0).to_pylist()
        self.image_index = {str(image_id): index for index, image_id in enumerate(ids)}
        self.cached_row_group: int | None = None
        self.cached_rows: list[dict[str, Any]] = []

    def get(self, image_id: str) -> dict[str, Any]:
        row_index = self.image_index[image_id]
        row_group = bisect_right(self.row_group_starts, row_index) - 1
        if row_group != self.cached_row_group:
            self.cached_rows = self.parquet.read_row_group(row_group).to_pylist()
            self.cached_row_group = row_group
        local_index = row_index - self.row_group_starts[row_group]
        return self.cached_rows[local_index]


def decode_cache_image(
    row: dict[str, Any],
    image_row: dict[str, Any],
) -> tuple[Image.Image, dict[str, Any]]:
    if "student_image_id" not in row:
        raise KeyError(
            "Dataset rows do not contain student_image_id. Use the parquet dataset "
            "built by scripts/build_student_img512_parquet_dataset.py."
        )
    with Image.open(BytesIO(image_row["image_bytes"])) as loaded:
        image = loaded.convert("RGB")
    return image, {
        "image_source": "student-512",
        "student_image_id": str(row["student_image_id"]),
        "student_image_path": image_row.get("student_image_path"),
        "student_image_sha256": image_row.get("student_image_sha256"),
        "student_image_format": image_row.get("student_image_format"),
        "original_size": image_row.get("original_size"),
        "resized_content_size": image_row.get("resized_content_size"),
        "canvas_size": image_row.get("canvas_size"),
        "scale": image_row.get("scale"),
        "padding": image_row.get("padding"),
        "background_rgb": image_row.get("background_rgb"),
        "image_bytes_source": "images.parquet",
    }


def prepare_batch(
    batch_indices: list[int],
    dataset: Any,
    image_store: ParquetImageStore,
    processor: Any,
    decode_executor: ThreadPoolExecutor,
) -> dict[str, Any]:
    rows = [dataset[dataset_index] for dataset_index in batch_indices]
    image_rows = [image_store.get(str(row["student_image_id"])) for row in rows]
    images_and_identities = list(
        decode_executor.map(
            lambda pair: decode_cache_image(*pair),
            zip(rows, image_rows, strict=True),
        )
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
        "inputs": processor(text=chat_texts, images=images, return_tensors="pt", padding=True),
    }


def planned_indices(dataset_len: int, args: argparse.Namespace) -> list[int]:
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.decode_workers <= 0:
        raise ValueError("--decode-workers must be positive")
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

    stop = dataset_len if args.end_index is None else min(dataset_len, args.end_index)
    if args.max_examples is not None:
        stop = min(stop, args.start_index + args.max_examples)
    return [
        index
        for index in range(args.start_index, stop)
        if index % args.shard_count == args.shard_index
    ]


def completed_indices(path: Path) -> set[int]:
    completed: set[int] = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_number}") from exc
            completed.add(int(payload["dataset_index"]))
    return completed


def tensor_metadata(tensor: torch.Tensor) -> dict[str, Any]:
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


def yes_no_metrics(
    yes_logit: float,
    no_logit: float,
    hard_label: str,
    temperature: float,
) -> dict[str, Any]:
    logits = torch.tensor([yes_logit, no_logit], dtype=torch.float32) / temperature
    log_probs = torch.log_softmax(logits, dim=0)
    probs = torch.softmax(logits, dim=0)
    target_index = 0 if hard_label == "yes" else 1
    target = torch.zeros_like(probs)
    target[target_index] = 1.0
    prediction = "yes" if int(torch.argmax(probs).item()) == 0 else "no"
    diff = probs - target
    return {
        "prediction": prediction,
        "correct": prediction == hard_label,
        "yes_probability": float(probs[0].item()),
        "no_probability": float(probs[1].item()),
        "target_probability": float(probs[target_index].item()),
        "nll": float(-log_probs[target_index].item()),
        "l1_one_hot": float(torch.abs(diff).sum().item()),
        "l2_one_hot": float(torch.linalg.vector_norm(diff, ord=2).item()),
        "squared_l2_one_hot": float(torch.sum(diff * diff).item()),
    }


def empty_metric_sums() -> dict[str, float]:
    return {
        "correct": 0.0,
        "nll": 0.0,
        "l1_one_hot": 0.0,
        "l2_one_hot": 0.0,
        "squared_l2_one_hot": 0.0,
        "target_probability": 0.0,
    }


def update_metric_sums(sums: dict[str, float], metrics: dict[str, Any]) -> None:
    sums["correct"] += float(metrics["correct"])
    sums["nll"] += float(metrics["nll"])
    sums["l1_one_hot"] += float(metrics["l1_one_hot"])
    sums["l2_one_hot"] += float(metrics["l2_one_hot"])
    sums["squared_l2_one_hot"] += float(metrics["squared_l2_one_hot"])
    sums["target_probability"] += float(metrics["target_probability"])


def aggregate_metric_sums(sums: dict[str, float], count: int) -> dict[str, Any]:
    if count == 0:
        return {
            "records": 0,
            "accuracy": None,
            "mean_nll": None,
            "mean_l1_one_hot": None,
            "mean_l2_one_hot": None,
            "mean_squared_l2_one_hot": None,
            "mean_target_probability": None,
        }
    return {
        "records": count,
        "accuracy": sums["correct"] / count,
        "mean_nll": sums["nll"] / count,
        "mean_l1_one_hot": sums["l1_one_hot"] / count,
        "mean_l2_one_hot": sums["l2_one_hot"] / count,
        "mean_squared_l2_one_hot": sums["squared_l2_one_hot"] / count,
        "mean_target_probability": sums["target_probability"] / count,
    }


def entropy_from_log_probs(log_probs: torch.Tensor) -> float:
    probs = torch.exp(log_probs)
    return float((-(probs * log_probs).sum()).detach().cpu())


def sequence_log_likelihoods(
    processor: Any,
    answer_texts: list[str],
    prompt_next_logits: torch.Tensor,
    temperature: float,
) -> list[dict[str, Any]]:
    token_ids = [
        processor.tokenizer(answer_text, add_special_tokens=False)["input_ids"]
        for answer_text in answer_texts
    ]
    if any(len(ids) != 1 for ids in token_ids):
        raise ValueError("Answer variants must tokenize to exactly one token")

    first_log_probs = torch.log_softmax(prompt_next_logits.float() / temperature, dim=-1)
    first_entropy = entropy_from_log_probs(first_log_probs)
    step_logprobs = [
        [float(first_log_probs[ids[0]].detach().cpu())]
        for ids in token_ids
    ]
    step_entropies = [[first_entropy] for _ in token_ids]

    scores: list[dict[str, Any]] = []
    for answer_text, ids, logprobs, entropies in zip(
        answer_texts,
        token_ids,
        step_logprobs,
        step_entropies,
        strict=True,
    ):
        scores.append(
            {
                "text": answer_text,
                "normalized": "yes" if "yes" in answer_text.lower() else "no",
                "token_ids": ids,
                "tokens": processor.tokenizer.convert_ids_to_tokens(ids),
                "log_likelihood": sum(logprobs),
                "mean_log_likelihood": sum(logprobs) / len(logprobs),
                "step_logprobs": logprobs,
                "step_entropies": entropies,
            }
        )
    return scores


def main() -> None:
    args = parse_args()
    if args.force and args.output.exists():
        args.output.unlink()
    if args.output.exists() and not args.force and not args.resume:
        raise FileExistsError(f"{args.output} exists. Pass --force to replace it.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output.with_suffix(".manifest.json")

    dataset = load_examples_dataset(args.dataset, args.split)
    indices = planned_indices(len(dataset), args)
    already_done = completed_indices(args.output) if args.resume else set()
    indices = [index for index in indices if index not in already_done]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dataset": str(args.dataset),
                    "image_source": args.image_source,
                    "split": args.split,
                    "dataset_rows": len(dataset),
                    "planned_records": len(indices),
                    "skipped_existing_records": len(already_done),
                    "first_index": indices[0] if indices else None,
                    "last_index": indices[-1] if indices else None,
                    "shard_count": args.shard_count,
                    "shard_index": args.shard_index,
                    "batch_size": args.batch_size,
                    "output": str(args.output),
                },
                indent=2,
            )
        )
        return

    device = device_from_arg(args.device)
    dtype = dtype_from_arg(args.torch_dtype, device)
    processor = AutoProcessor.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        backend=args.image_processor_backend,
    )
    processor.tokenizer.padding_side = "right"
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    model_revision_path = (
        Path.home()
        / ".cache/huggingface/hub/models--HuggingFaceTB--SmolVLM-256M-Instruct/refs/main"
    )
    model_revision = model_revision_path.read_text().strip() if model_revision_path.exists() else None

    yes_token_id = processor.tokenizer("yes", add_special_tokens=False)["input_ids"][0]
    no_token_id = processor.tokenizer("no", add_special_tokens=False)["input_ids"][0]
    records_written = 0
    metric_sums = empty_metric_sums()
    image_store = ParquetImageStore(args.dataset)

    total_examples = len(indices)
    output_mode = "a" if args.resume else "w"
    decode_executor = ThreadPoolExecutor(max_workers=args.decode_workers)
    prefetch_executor = ThreadPoolExecutor(max_workers=1)
    with args.output.open(output_mode, encoding="utf-8") as handle:
        progress = tqdm(
            total=total_examples,
            desc="Caching teacher logits",
            unit="example",
        )
        batch_starts = list(range(0, total_examples, args.batch_size))
        future = None
        if batch_starts:
            future = prefetch_executor.submit(
                prepare_batch,
                indices[: args.batch_size],
                dataset,
                image_store,
                processor,
                decode_executor,
            )
        for batch_number, batch_start in enumerate(batch_starts):
            assert future is not None
            prepared = future.result()
            next_start = batch_start + args.batch_size
            future = (
                prefetch_executor.submit(
                    prepare_batch,
                    indices[next_start : next_start + args.batch_size],
                    dataset,
                    image_store,
                    processor,
                    decode_executor,
                )
                if batch_number + 1 < len(batch_starts)
                else None
            )
            batch_indices = prepared["batch_indices"]
            rows = prepared["rows"]
            images = prepared["images"]
            image_identities = prepared["image_identities"]
            teacher_prompts = prepared["teacher_prompts"]
            chat_texts = prepared["chat_texts"]
            inputs = prepared["inputs"]
            inputs = {key: value.to(device) for key, value in inputs.items()}

            with torch.inference_mode():
                outputs = model(**inputs, use_cache=False)

            sequence_lengths = inputs["attention_mask"].sum(dim=1)
            for batch_offset, dataset_index in enumerate(batch_indices):
                row = rows[batch_offset]
                image = images[batch_offset]
                image_identity = image_identities[batch_offset]
                teacher_prompt = teacher_prompts[batch_offset]
                chat_text = chat_texts[batch_offset]
                sequence_length = int(sequence_lengths[batch_offset].item())
                next_logits = outputs.logits[batch_offset, sequence_length - 1].float()
                scaled_logits = next_logits / args.temperature
                log_probs = torch.log_softmax(scaled_logits, dim=-1)
                probs = torch.softmax(scaled_logits, dim=-1)
                entropy = float((-(probs * log_probs).sum()).detach().cpu())
                yes_logit = float(next_logits[yes_token_id].detach().cpu())
                no_logit = float(next_logits[no_token_id].detach().cpu())
                standalone_metrics = yes_no_metrics(
                    yes_logit=yes_logit,
                    no_logit=no_logit,
                    hard_label=str(row["answer"]),
                    temperature=args.temperature,
                )
                answer_variants = [
                    variant for variants in ANSWER_VARIANTS.values() for variant in variants
                ]
                answer_variant_scores = sequence_log_likelihoods(
                    processor=processor,
                    answer_texts=answer_variants,
                    prompt_next_logits=next_logits,
                    temperature=args.temperature,
                )
                input_ids = inputs["input_ids"][batch_offset, :sequence_length].detach().cpu().tolist()
                attention_mask = (
                    inputs["attention_mask"][batch_offset, :sequence_length].detach().cpu().tolist()
                )
                record = {
                    "cache_schema_version": 1,
                    "dataset_index": dataset_index,
                    "source_subset": row["source_subset"],
                    "source": row["source"],
                    "original_index": int(row["original_index"]),
                    "qa_index": int(row["qa_index"]),
                    "hard_label": row["answer"],
                    "teacher_prompt": teacher_prompt,
                    "student_prompt": row["student_prompt"],
                    "removed_last_line": row["removed_last_line"],
                    "input_identity": {
                        "filtered_dataset": str(args.dataset),
                        "image_source": args.image_source,
                        "split": args.split,
                        "model": args.model,
                        "model_revision": model_revision,
                        "teacher_prompt_sha256": hashlib.sha256(
                            teacher_prompt.encode("utf-8")
                        ).hexdigest(),
                        "student_prompt_sha256": hashlib.sha256(
                            str(row["student_prompt"]).encode("utf-8")
                        ).hexdigest(),
                        "image_sha256": image_identity["student_image_sha256"],
                        "image_identity": image_identity,
                    },
                    "teacher_input": {
                        "input_ids": input_ids,
                        "tokens": processor.tokenizer.convert_ids_to_tokens(input_ids),
                        "attention_mask": attention_mask,
                        "chat_template_text": chat_text,
                    },
                    "image_preprocessing": {
                        "original_image_size": list(image.size),
                        "original_image_mode": image.mode,
                        "image_source": args.image_source,
                        "image_identity": image_identity,
                        "pixel_values": tensor_metadata(inputs["pixel_values"][batch_offset : batch_offset + 1]),
                        "pixel_attention_mask": tensor_metadata(
                            inputs["pixel_attention_mask"][batch_offset : batch_offset + 1]
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
                        "standalone": {
                            "yes": {
                                "token_id": int(yes_token_id),
                                "token": processor.tokenizer.convert_ids_to_tokens([yes_token_id])[0],
                                "logit": yes_logit,
                                "log_likelihood": float(log_probs[yes_token_id].detach().cpu()),
                            },
                            "no": {
                                "token_id": int(no_token_id),
                                "token": processor.tokenizer.convert_ids_to_tokens([no_token_id])[0],
                                "logit": no_logit,
                                "log_likelihood": float(log_probs[no_token_id].detach().cpu()),
                            },
                            "yes_minus_no_logit": yes_logit - no_logit,
                            "entropy": entropy,
                        },
                        "answer_variant_sequences": answer_variant_scores,
                    },
                    "teacher_metrics": {
                        "standalone_yes_no": standalone_metrics,
                        "metric_definitions": {
                            "accuracy": "argmax over standalone yes/no logits equals hard label",
                            "l1_one_hot": "L1 distance between standalone yes/no softmax and one-hot hard label",
                            "l2_one_hot": "L2 distance between standalone yes/no softmax and one-hot hard label",
                            "nll": "negative log-likelihood of hard label under standalone yes/no softmax",
                        },
                    },
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                records_written += 1
                update_metric_sums(metric_sums, standalone_metrics)
                if args.flush_every > 0 and records_written % args.flush_every == 0:
                    handle.flush()
            progress.update(len(batch_indices))
        progress.close()
    prefetch_executor.shutdown()
    decode_executor.shutdown()

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output": str(args.output),
        "records": records_written,
        "records_already_done": len(already_done),
        "planned_records_this_invocation": total_examples,
        "image_source": args.image_source,
        "teacher_metrics": {
            "standalone_yes_no": aggregate_metric_sums(metric_sums, records_written),
            "metric_definitions": {
                "accuracy": "argmax over standalone yes/no logits equals hard label",
                "mean_l1_one_hot": "mean L1 distance between standalone yes/no softmax and one-hot hard label",
                "mean_l2_one_hot": "mean L2 distance between standalone yes/no softmax and one-hot hard label",
                "mean_nll": "mean negative log-likelihood of hard label under standalone yes/no softmax",
            },
        },
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "model": args.model,
        "model_revision": model_revision,
        "device": device,
        "torch_dtype": str(dtype).replace("torch.", ""),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "pid": os.getpid(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "answer_variants": ANSWER_VARIANTS,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote cache: {args.output}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
