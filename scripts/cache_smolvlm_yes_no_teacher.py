from __future__ import annotations

import argparse
from bisect import bisect_right
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
    "yes": ["yes", "Yes", " yes", "Yes.", "yes."],
    "no": ["no", "No", " no", "No.", "no."],
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
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--variant-batch-size", type=int, default=10)
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


def image_sha256(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return hashlib.sha256(buf.getvalue()).hexdigest()


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


def load_cache_image(
    row: dict[str, Any],
    args: argparse.Namespace,
    image_store: ParquetImageStore,
) -> tuple[Image.Image, dict[str, Any]]:
    if "student_image_id" not in row:
        raise KeyError(
            "Dataset rows do not contain student_image_id. Use the parquet dataset "
            "built by scripts/build_student_img512_parquet_dataset.py."
        )
    image_row = image_store.get(str(row["student_image_id"]))
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


def planned_indices(dataset_len: int, args: argparse.Namespace) -> list[int]:
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.variant_batch_size <= 0:
        raise ValueError("--variant-batch-size must be positive")
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


def sequence_log_likelihoods(
    model: Any,
    processor: Any,
    chat_text: str,
    image: Image.Image,
    answer_texts: list[str],
    prompt_len: int,
    device: str,
    temperature: float,
    batch_size: int,
) -> list[dict[str, Any]]:
    scores: list[dict[str, Any]] = []
    for batch_start in range(0, len(answer_texts), batch_size):
        batch_answers = answer_texts[batch_start : batch_start + batch_size]
        full_inputs = processor(
            text=[chat_text + answer_text for answer_text in batch_answers],
            images=[image] * len(batch_answers),
            return_tensors="pt",
            padding=True,
        )
        full_inputs = {key: value.to(device) for key, value in full_inputs.items()}
        with torch.inference_mode():
            outputs = model(**full_inputs, use_cache=False)

        for batch_offset, answer_text in enumerate(batch_answers):
            input_ids = full_inputs["input_ids"][batch_offset]
            seq_len = int(full_inputs["attention_mask"][batch_offset].sum().item())
            answer_token_ids = input_ids[prompt_len:seq_len].detach().cpu().tolist()
            step_logprobs: list[float] = []
            step_entropies: list[float] = []
            for position in range(prompt_len, seq_len):
                logits = outputs.logits[batch_offset, position - 1].float() / temperature
                log_probs = torch.log_softmax(logits, dim=-1)
                probs = torch.softmax(logits, dim=-1)
                token_id = int(input_ids[position].item())
                step_logprobs.append(float(log_probs[token_id].detach().cpu()))
                step_entropies.append(float((-(probs * log_probs).sum()).detach().cpu()))

            scores.append(
                {
                    "text": answer_text,
                    "normalized": "yes" if "yes" in answer_text.lower() else "no",
                    "token_ids": answer_token_ids,
                    "tokens": processor.tokenizer.convert_ids_to_tokens(answer_token_ids),
                    "log_likelihood": sum(step_logprobs),
                    "mean_log_likelihood": (
                        sum(step_logprobs) / len(step_logprobs) if step_logprobs else None
                    ),
                    "step_logprobs": step_logprobs,
                    "step_entropies": step_entropies,
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
    with args.output.open(output_mode, encoding="utf-8") as handle:
        for dataset_index in tqdm(
            indices,
            total=total_examples,
            desc="Caching teacher logits",
            unit="example",
        ):
            row = dataset[dataset_index]
            image, image_identity = load_cache_image(row, args, image_store)
            teacher_prompt = str(row["teacher_prompt"])
            chat_text = chat_prompt(processor, teacher_prompt)
            inputs = processor(text=chat_text, images=[image], return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}

            with torch.inference_mode():
                outputs = model(**inputs, use_cache=False)

            next_logits = outputs.logits[0, -1].float()
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

            prompt_len = int(inputs["input_ids"].shape[1])
            answer_variants = [
                variant for variants in ANSWER_VARIANTS.values() for variant in variants
            ]
            answer_variant_scores = sequence_log_likelihoods(
                model=model,
                processor=processor,
                chat_text=chat_text,
                image=image,
                answer_texts=answer_variants,
                prompt_len=prompt_len,
                device=device,
                temperature=args.temperature,
                batch_size=args.variant_batch_size,
            )

            input_ids = inputs["input_ids"][0].detach().cpu().tolist()
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
                    "image_sha256_png": image_sha256(image),
                    "image_identity": image_identity,
                },
                "teacher_input": {
                    "input_ids": input_ids,
                    "tokens": processor.tokenizer.convert_ids_to_tokens(input_ids),
                    "attention_mask": inputs["attention_mask"][0].detach().cpu().tolist(),
                    "chat_template_text": chat_text,
                },
                "image_preprocessing": {
                    "original_image_size": list(image.size),
                    "original_image_mode": image.mode,
                    "image_source": args.image_source,
                    "image_identity": image_identity,
                    "pixel_values": tensor_metadata(inputs["pixel_values"]),
                    "pixel_attention_mask": tensor_metadata(inputs["pixel_attention_mask"]),
                    "image_token_id": int(model.config.image_token_id),
                    "image_token_count": int(
                        (inputs["input_ids"] == int(model.config.image_token_id)).sum().item()
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
            tqdm.write(
                f"cached {records_written}/{total_examples}: "
                f"{row['source_subset']}#{row['original_index']} qa={row['qa_index']}"
            )

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
