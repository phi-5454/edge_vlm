from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
import transformers
from datasets import load_from_disk
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor


DEFAULT_MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"
DEFAULT_DATASET = Path("data/the_cauldron_yes_no_vsr_token1000")
DEFAULT_SOURCE_ROOT = Path("data/the_cauldron")
DEFAULT_OUTPUT = Path("artifacts/teacher_cache/smolvlm_yes_no_vsr_token1000.jsonl")
ANSWER_VARIANTS = {
    "yes": ["yes", "Yes", " yes", "Yes.", "yes."],
    "no": ["no", "No", " no", "No.", "no."],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache SmolVLM yes/no teacher logits.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--split", default="combined")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-examples", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
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


def first_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, list):
        for item in value:
            try:
                return first_image(item)
            except ValueError:
                pass
    raise ValueError("No image found in source row")


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


def load_source_row(source_cache: dict[str, Any], source_root: Path, subset: str, index: int) -> dict[str, Any]:
    if subset not in source_cache:
        source_cache[subset] = load_from_disk(source_root / subset)
    return source_cache[subset][index]


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


def sequence_log_likelihood(
    model: Any,
    processor: Any,
    chat_text: str,
    image: Image.Image,
    answer_text: str,
    prompt_len: int,
    device: str,
    temperature: float,
) -> dict[str, Any]:
    full_inputs = processor(text=chat_text + answer_text, images=[image], return_tensors="pt")
    full_inputs = {key: value.to(device) for key, value in full_inputs.items()}
    with torch.inference_mode():
        outputs = model(**full_inputs, use_cache=False)

    input_ids = full_inputs["input_ids"][0]
    answer_token_ids = input_ids[prompt_len:].detach().cpu().tolist()
    step_logprobs: list[float] = []
    step_entropies: list[float] = []
    for position in range(prompt_len, input_ids.shape[0]):
        logits = outputs.logits[0, position - 1].float() / temperature
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        token_id = int(input_ids[position].item())
        step_logprobs.append(float(log_probs[token_id].detach().cpu()))
        step_entropies.append(float((-(probs * log_probs).sum()).detach().cpu()))

    return {
        "text": answer_text,
        "normalized": "yes" if "yes" in answer_text.lower() else "no",
        "token_ids": answer_token_ids,
        "tokens": processor.tokenizer.convert_ids_to_tokens(answer_token_ids),
        "log_likelihood": sum(step_logprobs),
        "mean_log_likelihood": sum(step_logprobs) / len(step_logprobs) if step_logprobs else None,
        "step_logprobs": step_logprobs,
        "step_entropies": step_entropies,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"{args.output} exists. Pass --force to replace it.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output.with_suffix(".manifest.json")

    device = device_from_arg(args.device)
    dtype = dtype_from_arg(args.torch_dtype, device)
    processor = AutoProcessor.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    dataset_dict = load_from_disk(args.dataset)
    dataset = dataset_dict[args.split]
    end_index = min(len(dataset), args.start_index + args.max_examples)
    source_cache: dict[str, Any] = {}
    model_revision_path = (
        Path.home()
        / ".cache/huggingface/hub/models--HuggingFaceTB--SmolVLM-256M-Instruct/refs/main"
    )
    model_revision = model_revision_path.read_text().strip() if model_revision_path.exists() else None

    yes_token_id = processor.tokenizer("yes", add_special_tokens=False)["input_ids"][0]
    no_token_id = processor.tokenizer("no", add_special_tokens=False)["input_ids"][0]
    records_written = 0

    total_examples = end_index - args.start_index
    with args.output.open("w", encoding="utf-8") as handle:
        for dataset_index in tqdm(
            range(args.start_index, end_index),
            total=total_examples,
            desc="Caching teacher logits",
            unit="example",
        ):
            row = dataset[dataset_index]
            source_row = load_source_row(
                source_cache,
                args.source_root,
                row["source_subset"],
                int(row["original_index"]),
            )
            image = first_image(source_row.get("images") or source_row.get("image"))
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

            answer_variant_scores = []
            prompt_len = int(inputs["input_ids"].shape[1])
            answer_variants = [
                variant for variants in ANSWER_VARIANTS.values() for variant in variants
            ]
            for variant in tqdm(
                answer_variants,
                desc="Scoring answer variants",
                unit="variant",
                leave=False,
            ):
                answer_variant_scores.append(
                    sequence_log_likelihood(
                        model=model,
                        processor=processor,
                        chat_text=chat_text,
                        image=image,
                        answer_text=variant,
                        prompt_len=prompt_len,
                        device=device,
                        temperature=args.temperature,
                    )
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
                    "source_root": str(args.source_root),
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
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            records_written += 1
            tqdm.write(
                f"cached {records_written}/{total_examples}: "
                f"{row['source_subset']}#{row['original_index']} qa={row['qa_index']}"
            )

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output": str(args.output),
        "records": records_written,
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
        "answer_variants": ANSWER_VARIANTS,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote cache: {args.output}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
