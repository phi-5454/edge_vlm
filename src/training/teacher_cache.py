from __future__ import annotations

from pathlib import Path
from typing import Any

import lightning as L
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import CacheTeacherConfig
from training.datamodule import CauldronDataModule


def cache_teacher(config: CacheTeacherConfig) -> None:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    L.seed_everything(config.seed)
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(config.device)

    processor = AutoProcessor.from_pretrained(
        config.teacher.model_name,
        trust_remote_code=config.teacher.trust_remote_code,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        config.teacher.model_name,
        trust_remote_code=config.teacher.trust_remote_code,
    ).to(device)
    model.eval()

    data = CauldronDataModule(config.data, processor=processor, batch_size=1)
    data.setup("fit")
    embeddings = []
    dataloader = DataLoader(
        data.train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=data._collate,
    )
    for batch in tqdm(dataloader, total=len(dataloader), desc="Caching teacher embeddings"):
        teacher_inputs = {
            key: value.to(device)
            for key, value in batch.items()
            if key != "labels" and isinstance(value, torch.Tensor)
        }
        with torch.inference_mode():
            embedding = teacher_embedding(
                model,
                teacher_inputs,
                normalize=True,
            )
        embeddings.append(embedding.cpu().to(torch.float16))

    torch.save(
        {
            "teacher_model": config.teacher.model_name,
            "dataset_name": config.data.dataset_name,
            "dataset_config": config.data.dataset_config,
            "split": config.data.train_split,
            "max_samples": config.data.max_samples,
            "embeddings": torch.cat(embeddings, dim=0),
        },
        output_path,
    )
    print(f"Wrote teacher cache to {output_path}")


def teacher_embedding(
    model: Any,
    teacher_inputs: dict[str, torch.Tensor],
    normalize: bool,
) -> torch.Tensor:
    outputs = model(**teacher_inputs, output_hidden_states=True, return_dict=True)
    hidden_states = outputs.hidden_states[-1]
    attention_mask = teacher_inputs.get("attention_mask")
    if attention_mask is None:
        pooled = hidden_states.mean(dim=1)
    else:
        weights = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        pooled = (hidden_states * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
    return F.normalize(pooled, dim=-1) if normalize else pooled


def load_teacher_cache(path: str | Path) -> torch.Tensor:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    embeddings = payload["embeddings"] if isinstance(payload, dict) else payload
    if not isinstance(embeddings, torch.Tensor):
        raise TypeError(f"Teacher cache at {path} does not contain tensor embeddings.")
    return embeddings.float()


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
