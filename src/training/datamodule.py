from __future__ import annotations

from typing import Any

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset

from config import DataConfig


class CauldronDataModule(L.LightningDataModule):
    """Loads The Cauldron and prepares batches with a Hugging Face processor."""

    def __init__(self, config: DataConfig, processor: Any, batch_size: int) -> None:
        super().__init__()
        self.config = config
        self.processor = processor
        self.batch_size = batch_size
        self.train_dataset = None
        self.val_dataset = None
        self.teacher_embeddings = None

    def setup(self, stage: str | None = None) -> None:
        from datasets import load_dataset

        kwargs = {}
        if self.config.dataset_config:
            kwargs["name"] = self.config.dataset_config

        self.train_dataset = load_dataset(
            self.config.dataset_name,
            split=self.config.train_split,
            **kwargs,
        )
        if self.config.max_samples:
            self.train_dataset = self.train_dataset.select(range(self.config.max_samples))

        if self.config.val_split:
            self.val_dataset = load_dataset(
                self.config.dataset_name,
                split=self.config.val_split,
                **kwargs,
            )

    def train_dataloader(self) -> DataLoader:
        dataset = self.train_dataset
        if self.teacher_embeddings is not None:
            dataset = TeacherCacheDataset(dataset, self.teacher_embeddings)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=self._collate,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            return DataLoader([])
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=self._collate,
        )

    def _collate(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        teacher_embeddings = [
            example.pop("teacher_embedding") for example in examples if "teacher_embedding" in example
        ]
        texts = [self._with_image_token(self._extract_text(example)) for example in examples]
        images = [self._extract_image(example) for example in examples]
        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        batch["labels"] = batch["input_ids"].clone()
        if teacher_embeddings:
            batch["teacher_embedding"] = torch.stack(teacher_embeddings)
        return batch

    def _extract_text(self, example: dict[str, Any]) -> str:
        value = example.get(self.config.text_column)
        if isinstance(value, str):
            return value
        if isinstance(value, list) and value:
            first = value[0]
            if isinstance(first, str):
                return first
            if isinstance(first, dict):
                return first.get("text") or first.get("content") or str(first)
        return str(value)

    def _extract_image(self, example: dict[str, Any]) -> Any:
        value = example.get(self.config.image_column)
        if isinstance(value, list) and value:
            return value[0]
        return value

    def _with_image_token(self, text: str) -> str:
        if self.config.image_token in text:
            return text
        return f"{self.config.image_token}\n{text}"


class TeacherCacheDataset(Dataset):
    def __init__(self, dataset: Dataset, teacher_embeddings: torch.Tensor) -> None:
        self.dataset = dataset
        self.teacher_embeddings = teacher_embeddings
        if len(dataset) > len(teacher_embeddings):
            raise ValueError(
                f"Dataset has {len(dataset)} samples but teacher cache has "
                f"{len(teacher_embeddings)} embeddings."
            )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = dict(self.dataset[index])
        item["teacher_embedding"] = self.teacher_embeddings[index]
        return item
