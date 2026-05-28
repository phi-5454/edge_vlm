from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


def _first_image(value: Any) -> Image.Image | None:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, list):
        for item in value:
            image = _first_image(item)
            if image is not None:
                return image
    return None


def _message_text(message: Any) -> str:
    if isinstance(message, str):
        return message
    if isinstance(message, dict):
        content = message.get("content", "")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            return " ".join(parts).strip()
        if isinstance(content, str):
            return content.strip()
        if isinstance(message.get("text"), str):
            return message["text"].strip()
    return ""


def _extract_qa(sample: dict[str, Any]) -> tuple[str, str | None]:
    texts = sample.get("texts")
    if isinstance(texts, list):
        user_parts: list[str] = []
        assistant_parts: list[str] = []
        for message in texts:
            role = message.get("role") if isinstance(message, dict) else None
            if isinstance(message, dict) and isinstance(message.get("user"), str):
                user_parts.append(message["user"].strip())
                assistant = message.get("assistant")
                if isinstance(assistant, str):
                    assistant_parts.append(assistant.strip())
                continue
            text = _message_text(message)
            if not text:
                continue
            if role in {"assistant", "gpt"}:
                assistant_parts.append(text)
            else:
                user_parts.append(text)
        if user_parts:
            return user_parts[0], assistant_parts[0] if assistant_parts else None
        if texts:
            return _message_text(texts[0]), _message_text(texts[1]) if len(texts) > 1 else None

    for key in ("question", "query", "prompt", "instruction"):
        if isinstance(sample.get(key), str):
            answer = sample.get("answer")
            return sample[key], answer if isinstance(answer, str) else None

    return "Describe this image briefly.", None


def _chat_prompt(processor: Any, question: str) -> str:
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


def _device(value: str) -> str:
    if value != "auto":
        return value
    return "cuda" if torch.cuda.is_available() else "cpu"


def _dtype(value: str) -> torch.dtype | str:
    if value == "auto":
        return "auto"
    return getattr(torch, value)


def run_smolvlm_smoke(cfg: DictConfig) -> Path:
    resolved = OmegaConf.to_container(cfg, resolve=True)
    output = Path(str(cfg.output))
    output.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        str(cfg.dataset_name),
        str(cfg.dataset_config),
        split=str(cfg.split),
        streaming=True,
    )
    processor = AutoProcessor.from_pretrained(
        str(cfg.model_name),
        trust_remote_code=bool(cfg.trust_remote_code),
    )
    device = _device(str(cfg.device))
    model = AutoModelForImageTextToText.from_pretrained(
        str(cfg.model_name),
        torch_dtype=_dtype(str(cfg.torch_dtype)),
        trust_remote_code=bool(cfg.trust_remote_code),
    ).to(device)
    model.eval()

    records: list[dict[str, Any]] = []
    scanned = 0
    for sample in dataset:
        scanned += 1
        image = _first_image(sample.get("images") or sample.get("image"))
        if image is None:
            if scanned >= int(cfg.max_streamed_samples):
                break
            continue

        question, reference_answer = _extract_qa(sample)
        prompt = _chat_prompt(processor, question)
        inputs = processor(text=prompt, images=[image], return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=int(cfg.max_new_tokens),
                do_sample=False,
            )
        prompt_tokens = inputs["input_ids"].shape[1]
        answer = processor.batch_decode(
            generated_ids[:, prompt_tokens:],
            skip_special_tokens=True,
        )[0].strip()

        records.append(
            {
                "index_in_stream": scanned - 1,
                "question": question,
                "reference_answer": reference_answer,
                "model_answer": answer,
                "image_size": list(image.size),
            }
        )
        print(json.dumps(records[-1], ensure_ascii=False))
        if len(records) >= int(cfg.num_examples):
            break
        if scanned >= int(cfg.max_streamed_samples):
            break

    payload = {
        "config": resolved,
        "device": device,
        "torch_cuda_available": torch.cuda.is_available(),
        "scanned_samples": scanned,
        "examples": records,
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output
