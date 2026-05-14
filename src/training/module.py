from __future__ import annotations

from typing import Any

import lightning as L
import torch

from config import ModelConfig


class SmolVlmModule(L.LightningModule):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = None

    def setup(self, stage: str | None = None) -> None:
        if self.model is not None:
            return
        from transformers import AutoModelForImageTextToText

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
        )

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        outputs = self.model(**batch)
        self.log("train/loss", outputs.loss, prog_bar=True, on_step=True, on_epoch=True)
        return outputs.loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        outputs = self.model(**batch)
        self.log("val/loss", outputs.loss, prog_bar=True, on_epoch=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
