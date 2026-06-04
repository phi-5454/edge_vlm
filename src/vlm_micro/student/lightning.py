from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightning as L
import torch
from torch import nn

from vlm_micro.student.model import StudentBaseline


@dataclass
class BinaryTotals:
    true_positive: int = 0
    true_negative: int = 0
    false_positive: int = 0
    false_negative: int = 0

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        predicted = logits.detach() >= 0
        expected = labels.detach() >= 0.5
        self.true_positive += int((predicted & expected).sum().cpu())
        self.true_negative += int((~predicted & ~expected).sum().cpu())
        self.false_positive += int((predicted & ~expected).sum().cpu())
        self.false_negative += int((~predicted & expected).sum().cpu())

    def metrics(self) -> dict[str, float]:
        total = self.true_positive + self.true_negative + self.false_positive + self.false_negative
        predicted_positive = self.true_positive + self.false_positive
        actual_positive = self.true_positive + self.false_negative
        precision = self.true_positive / predicted_positive if predicted_positive else 0.0
        recall = self.true_positive / actual_positive if actual_positive else 0.0
        return {
            "accuracy": (self.true_positive + self.true_negative) / total if total else 0.0,
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        }


class StudentBaselineModule(L.LightningModule):
    def __init__(
        self,
        model: StudentBaseline,
        alpha: float,
        learning_rate: float,
        weight_decay: float,
        warmup_start_learning_rate: float | None = None,
        warmup_steps: int = 0,
        distill_kind: str = "mse_logit",
        temperature: float = 1.0,
    ):
        super().__init__()
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1.")
        if distill_kind not in {"mse_logit", "soft_bce"}:
            raise ValueError("distill_kind must be 'mse_logit' or 'soft_bce'.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if warmup_start_learning_rate is not None and not 0 < warmup_start_learning_rate <= learning_rate:
            raise ValueError("warmup_start_learning_rate must be positive and at most learning_rate.")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative.")
        if warmup_steps and warmup_start_learning_rate is None:
            raise ValueError("warmup_start_learning_rate is required when warmup_steps is positive.")
        self.model = model
        self.save_hyperparameters(ignore=["model"])
        self._totals = {stage: BinaryTotals() for stage in ("train", "val", "test")}

    def forward(self, token_ids: torch.Tensor, attention_mask: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        return self.model(token_ids, attention_mask, images)

    def _distill_loss(self, logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        if torch.isnan(teacher_logits).any():
            raise ValueError("Teacher cache is incomplete, but alpha < 1 requires every teacher logit.")
        if self.hparams.distill_kind == "mse_logit":
            return nn.functional.mse_loss(logits, teacher_logits)
        temperature = float(self.hparams.temperature)
        teacher_probability = torch.sigmoid(teacher_logits / temperature)
        return nn.functional.binary_cross_entropy_with_logits(
            logits / temperature, teacher_probability
        ) * temperature**2

    def _shared_step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        logits = self(batch["token_ids"], batch["attention_mask"], batch["images"])
        hard_loss = nn.functional.binary_cross_entropy_with_logits(logits, batch["labels"])
        if self.hparams.alpha < 1:
            distill_loss = self._distill_loss(logits, batch["teacher_logits"])
        else:
            distill_loss = torch.zeros((), device=logits.device)
        loss = (1 - self.hparams.alpha) * distill_loss + self.hparams.alpha * hard_loss
        batch_size = batch["labels"].shape[0]
        self.log(f"{stage}/loss", loss, on_step=stage == "train", on_epoch=True, batch_size=batch_size)
        self.log(f"{stage}/hard_loss", hard_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log(f"{stage}/distill_loss", distill_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self._totals[stage].update(logits, batch["labels"])
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_index: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_index: int) -> None:
        self._shared_step(batch, "val")

    def test_step(self, batch: dict[str, torch.Tensor], batch_index: int) -> None:
        self._shared_step(batch, "test")

    def _log_epoch_metrics(self, stage: str) -> None:
        metrics = self._totals[stage].metrics()
        self.log_dict({f"{stage}/{name}": value for name, value in metrics.items()}, sync_dist=True)
        self._totals[stage] = BinaryTotals()

    def on_train_epoch_end(self) -> None:
        self._log_epoch_metrics("train")

    def on_validation_epoch_end(self) -> None:
        self._log_epoch_metrics("val")

    def on_test_epoch_end(self) -> None:
        self._log_epoch_metrics("test")

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        if not self.hparams.warmup_steps:
            return optimizer
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=self.hparams.warmup_start_learning_rate / self.hparams.learning_rate,
            end_factor=1.0,
            total_iters=self.hparams.warmup_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


@dataclass
class MulticlassTotals:
    correct: int = 0
    total: int = 0

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        predicted = torch.argmax(logits.detach(), dim=1)
        expected = labels.detach()
        self.correct += int((predicted == expected).sum().cpu())
        self.total += int(expected.numel())

    def metrics(self) -> dict[str, float]:
        return {"accuracy": self.correct / self.total if self.total else 0.0}


class TallyQAStudentModule(L.LightningModule):
    def __init__(
        self,
        model: StudentBaseline,
        alpha: float,
        beta: float,
        learning_rate: float,
        weight_decay: float,
        warmup_start_learning_rate: float | None = None,
        warmup_steps: int = 0,
        temperature: float = 2.0,
    ):
        super().__init__()
        if alpha < 0:
            raise ValueError("alpha must be non-negative.")
        if beta < 0:
            raise ValueError("beta must be non-negative.")
        if alpha == 0 and beta == 0:
            raise ValueError("At least one of alpha or beta must be positive.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if warmup_start_learning_rate is not None and not 0 < warmup_start_learning_rate <= learning_rate:
            raise ValueError("warmup_start_learning_rate must be positive and at most learning_rate.")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative.")
        if warmup_steps and warmup_start_learning_rate is None:
            raise ValueError("warmup_start_learning_rate is required when warmup_steps is positive.")
        self.model = model
        self.save_hyperparameters(ignore=["model"])
        self._totals = {stage: MulticlassTotals() for stage in ("train", "val", "test")}

    def forward(self, token_ids: torch.Tensor, attention_mask: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        return self.model(token_ids, attention_mask, images)

    def _distill_loss(self, logits: torch.Tensor, teacher_probs: torch.Tensor) -> torch.Tensor:
        if torch.isnan(teacher_probs).any():
            raise ValueError("Teacher cache is incomplete, but beta > 0 requires every teacher distribution.")
        temperature = float(self.hparams.temperature)
        teacher_probs = teacher_probs.clamp_min(1e-8)
        teacher_probs = teacher_probs / teacher_probs.sum(dim=1, keepdim=True)
        student_log_probs = nn.functional.log_softmax(logits / temperature, dim=1)
        return (
            nn.functional.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
            * temperature**2
        )

    def _shared_step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        logits = self(batch["token_ids"], batch["attention_mask"], batch["images"])
        hard_loss = nn.functional.cross_entropy(logits, batch["labels"])
        if self.hparams.beta > 0:
            distill_loss = self._distill_loss(logits, batch["teacher_probs"])
        else:
            distill_loss = torch.zeros((), device=logits.device)
        loss = self.hparams.alpha * hard_loss + self.hparams.beta * distill_loss
        batch_size = batch["labels"].shape[0]
        self.log(f"{stage}/loss", loss, on_step=stage == "train", on_epoch=True, batch_size=batch_size)
        self.log(f"{stage}/ce_loss", hard_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log(f"{stage}/kl_loss", distill_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self._totals[stage].update(logits, batch["labels"])
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_index: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_index: int) -> None:
        self._shared_step(batch, "val")

    def test_step(self, batch: dict[str, torch.Tensor], batch_index: int) -> None:
        self._shared_step(batch, "test")

    def _log_epoch_metrics(self, stage: str) -> None:
        metrics = self._totals[stage].metrics()
        self.log_dict({f"{stage}/{name}": value for name, value in metrics.items()}, sync_dist=True)
        self._totals[stage] = MulticlassTotals()

    def on_train_epoch_end(self) -> None:
        self._log_epoch_metrics("train")

    def on_validation_epoch_end(self) -> None:
        self._log_epoch_metrics("val")

    def on_test_epoch_end(self) -> None:
        self._log_epoch_metrics("test")

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        if not self.hparams.warmup_steps:
            return optimizer
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=self.hparams.warmup_start_learning_rate / self.hparams.learning_rate,
            end_factor=1.0,
            total_iters=self.hparams.warmup_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
