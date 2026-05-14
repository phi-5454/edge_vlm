from __future__ import annotations

import json
import time
from pathlib import Path

from PIL import Image
import torch


def run_profile(model_name: str, output_path: Path, steps: int, warmup: int) -> None:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    output_path.parent.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(model_name, trust_remote_code=True).eval()

    prompt = "<image>\nDescribe the image for an embedded vision benchmark."
    image = Image.new("RGB", (224, 224), color="black")
    inputs = processor(text=[prompt], images=[image], return_tensors="pt", padding=True)

    records = []
    with torch.inference_mode():
        for index in range(warmup + steps):
            start = time.perf_counter()
            _ = model.generate(**inputs, max_new_tokens=16)
            elapsed_ms = (time.perf_counter() - start) * 1000
            if index >= warmup:
                records.append(
                    {
                        "model": model_name,
                        "step": index - warmup,
                        "latency_ms": elapsed_ms,
                        "device": "cpu",
                        "max_new_tokens": 16,
                    }
                )

    with output_path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
