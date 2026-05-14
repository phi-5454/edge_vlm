from __future__ import annotations

from pathlib import Path
from typing import Any

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from config import DistillConfig, config_to_dict
from models.tiny_vlm import TinyVLM, TinyVlmConfig
from training.callbacks import ModelSizeLogger
from training.datamodule import CauldronDataModule
from training.teacher_cache import load_teacher_cache, teacher_embedding
from training.train import load_tracking_env


class TinyVlmDistillationModule(L.LightningModule):
    def __init__(self, config: DistillConfig) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.student = TinyVLM(TinyVlmConfig(**config_to_dict(config.student)))
        self.teacher = None

    def setup(self, stage: str | None = None) -> None:
        if self.config.teacher_cache.enabled:
            return
        if self.teacher is not None:
            return
        from transformers import AutoModelForImageTextToText

        self.teacher = AutoModelForImageTextToText.from_pretrained(
            self.config.teacher.model_name,
            trust_remote_code=self.config.teacher.trust_remote_code,
        )
        self.teacher.eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad = False
        teacher_dim = self._teacher_hidden_size()
        if teacher_dim and teacher_dim != self.config.student.teacher_dim:
            self.config.student.teacher_dim = teacher_dim
            self.student.teacher_projection = nn.Linear(
                self.config.student.fusion_hidden_dim,
                teacher_dim,
            )

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        student_batch = self._student_batch(batch)
        student_outputs = self.student(**student_batch)
        with torch.inference_mode():
            teacher_targets = self._teacher_targets(batch)
        if teacher_targets.shape[-1] != student_outputs["teacher_embeds"].shape[-1]:
            self._resize_student_teacher_projection(teacher_targets.shape[-1])
            student_outputs = self.student(**student_batch)

        teacher_loss = F.mse_loss(student_outputs["teacher_embeds"], teacher_targets)
        contrastive_loss = self._contrastive_loss(
            student_outputs["image_embeds"],
            student_outputs["text_embeds"],
        )
        loss = (
            self.config.distillation.teacher_embedding_loss_weight * teacher_loss
            + self.config.distillation.image_text_contrastive_loss_weight * contrastive_loss
        )
        self.log("distill/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("distill/teacher_embedding_loss", teacher_loss, on_step=True, on_epoch=True)
        self.log("distill/image_text_contrastive_loss", contrastive_loss, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.student.parameters(),
            lr=self.config.teacher.learning_rate,
            weight_decay=self.config.teacher.weight_decay,
        )

    def _student_batch(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        max_tokens = self.config.student.max_text_tokens
        student_batch = {
            "pixel_values": self._student_pixel_values(batch["pixel_values"]),
            "input_ids": batch["input_ids"][:, :max_tokens],
        }
        if "attention_mask" in batch:
            student_batch["attention_mask"] = batch["attention_mask"][:, :max_tokens]
        return student_batch

    def _student_pixel_values(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim == 4:
            return pixel_values
        if pixel_values.ndim == 5:
            return pixel_values[:, 0]
        raise ValueError(f"Expected 4D or 5D pixel_values, got shape {tuple(pixel_values.shape)}")

    def _teacher_targets(self, batch: dict[str, Any]) -> torch.Tensor:
        if "teacher_embedding" in batch:
            return batch["teacher_embedding"].to(self.device)
        teacher_inputs = {
            key: value
            for key, value in batch.items()
            if key != "labels" and isinstance(value, torch.Tensor)
        }
        return teacher_embedding(
            self.teacher,
            teacher_inputs,
            normalize=self.config.distillation.normalize_teacher_targets,
        )

    def _contrastive_loss(self, image_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)
        logits = image_embeds @ text_embeds.T / self.config.distillation.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

    def _teacher_hidden_size(self) -> int | None:
        for config_name in ("text_config", "language_config"):
            nested_config = getattr(self.teacher.config, config_name, None)
            hidden_size = getattr(nested_config, "hidden_size", None)
            if hidden_size:
                return int(hidden_size)
        hidden_size = getattr(self.teacher.config, "hidden_size", None)
        return int(hidden_size) if hidden_size else None

    def _resize_student_teacher_projection(self, teacher_dim: int) -> None:
        self.config.student.teacher_dim = teacher_dim
        self.student.teacher_projection = nn.Linear(
            self.config.student.fusion_hidden_dim,
            teacher_dim,
        ).to(self.device)


def distill(config: DistillConfig) -> None:
    from transformers import AutoProcessor

    L.seed_everything(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    load_tracking_env(config.tracking.env_file)

    processor = AutoProcessor.from_pretrained(
        config.teacher.model_name,
        trust_remote_code=config.teacher.trust_remote_code,
    )
    student_vocab_size = len(processor.tokenizer) if hasattr(processor, "tokenizer") else None
    if student_vocab_size:
        config.student.vocab_size = max(config.student.vocab_size, student_vocab_size)

    model = TinyVlmDistillationModule(config)
    data = CauldronDataModule(config.data, processor=processor, batch_size=config.batch_size)
    if config.teacher_cache.enabled:
        data.teacher_embeddings = load_teacher_cache(config.teacher_cache.path)

    logger = WandbLogger(
        project=config.tracking.project,
        entity=config.tracking.entity,
        name=config.tracking.run_name,
        tags=config.tracking.tags,
        offline=config.tracking.offline,
        log_model=False,
        config=config_to_dict(config),
    )

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            monitor="distill/loss_epoch",
            save_top_k=1,
            mode="min",
        ),
    ]
    if config.size_logging.enabled:
        callbacks.append(ModelSizeLogger(config.size_logging, model_attribute="student"))

    trainer = L.Trainer(
        default_root_dir=output_dir,
        max_epochs=config.max_epochs,
        precision=config.precision,
        accumulate_grad_batches=config.accumulate_grad_batches,
        limit_val_batches=0 if config.data.val_split is None else None,
        log_every_n_steps=config.log_every_n_steps,
        logger=logger,
        callbacks=callbacks,
    )
    trainer.fit(model, datamodule=data)
