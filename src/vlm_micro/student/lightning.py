from __future__ import annotations

from dataclasses import dataclass
import textwrap
from typing import Any

import lightning as L
import matplotlib.pyplot as plt
import torch
import wandb
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
        self._log_learning_rate(stage, batch_size)
        self.log(
            f"{stage}/loss",
            loss,
            on_step=stage == "train",
            on_epoch=True,
            batch_size=batch_size,
            prog_bar=True,
        )
        self.log(
            f"{stage}/hard_loss",
            hard_loss,
            on_step=stage == "train",
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}/distill_loss",
            distill_loss,
            on_step=stage == "train",
            on_epoch=True,
            batch_size=batch_size,
        )
        self._totals[stage].update(logits, batch["labels"])
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_index: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def on_train_epoch_start(self) -> None:
        datamodule = getattr(getattr(self, "trainer", None), "datamodule", None)
        if datamodule is not None and hasattr(datamodule, "set_train_epoch"):
            datamodule.set_train_epoch(int(self.current_epoch))

    def _log_learning_rate(self, stage: str, batch_size: int) -> None:
        trainer = getattr(self, "_trainer", None)
        if stage != "train" or trainer is None or not trainer.optimizers:
            return
        learning_rate = float(trainer.optimizers[0].param_groups[0]["lr"])
        self.log(
            "train/lr",
            learning_rate,
            on_step=True,
            on_epoch=False,
            batch_size=batch_size,
            prog_bar=True,
        )

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
    within_one: int = 0
    total: int = 0

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        predicted = torch.argmax(logits.detach(), dim=1)
        expected = labels.detach()
        self.correct += int((predicted == expected).sum().cpu())
        self.within_one += int((predicted - expected).abs().le(1).sum().cpu())
        self.total += int(expected.numel())

    def metrics(self) -> dict[str, float]:
        return {
            "accuracy": self.correct / self.total if self.total else 0.0,
            "within_1_accuracy": self.within_one / self.total if self.total else 0.0,
        }


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
        image_learning_rate_scale: float = 1.0,
        temperature: float = 2.0,
        class_weights: list[float] | None = None,
        target_distribution: str = "hard",
        local_soft_sigma: float = 1.0,
        local_soft_radius: int = 1,
        validation_plot_samples: int = 4,
        validation_plot_every_n_epochs: int = 1,
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
        if image_learning_rate_scale <= 0:
            raise ValueError("image_learning_rate_scale must be positive.")
        if warmup_start_learning_rate is not None and not 0 < warmup_start_learning_rate <= learning_rate:
            raise ValueError("warmup_start_learning_rate must be positive and at most learning_rate.")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative.")
        if warmup_steps and warmup_start_learning_rate is None:
            raise ValueError("warmup_start_learning_rate is required when warmup_steps is positive.")
        if class_weights is not None:
            if not class_weights:
                raise ValueError("class_weights must not be empty.")
            if any(weight < 0 for weight in class_weights):
                raise ValueError("class_weights must be non-negative.")
        if target_distribution not in {"hard", "local_soft"}:
            raise ValueError("target_distribution must be 'hard' or 'local_soft'.")
        if local_soft_sigma <= 0:
            raise ValueError("local_soft_sigma must be positive.")
        if local_soft_radius < 0:
            raise ValueError("local_soft_radius must be non-negative.")
        if validation_plot_samples < 0:
            raise ValueError("validation_plot_samples must be non-negative.")
        if validation_plot_every_n_epochs <= 0:
            raise ValueError("validation_plot_every_n_epochs must be positive.")
        self.model = model
        weights = torch.tensor(class_weights or [], dtype=torch.float32)
        self.register_buffer("class_weights", weights, persistent=False)
        self.save_hyperparameters(ignore=["model"])
        self._totals = {stage: MulticlassTotals() for stage in ("train", "val", "test")}
        self.num_outputs = int(getattr(model, "num_outputs", 0))
        if self.num_outputs <= 0 and hasattr(model, "logits"):
            self.num_outputs = int(model.logits.shape[-1])
        if self.num_outputs <= 0:
            raise ValueError("TallyQAStudentModule could not infer the number of output classes.")
        self._confusions = {
            stage: torch.zeros((self.num_outputs, self.num_outputs), dtype=torch.long)
            for stage in ("val", "test")
        }
        self._validation_plot_rows: list[dict[str, Any]] = []

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

    def _target_distribution(self, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
        if self.hparams.target_distribution == "hard":
            return nn.functional.one_hot(labels, num_classes=num_classes).float()

        class_ids = torch.arange(num_classes, device=labels.device).unsqueeze(0)
        distances = (class_ids - labels.unsqueeze(1)).abs()
        radius = int(self.hparams.local_soft_radius)
        sigma = float(self.hparams.local_soft_sigma)
        targets = torch.exp(-(distances.float() ** 2) / (2 * sigma**2))
        if radius > 0:
            targets = targets.masked_fill(distances > radius, 0.0)
        targets = targets / targets.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return targets

    def _shared_step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        logits = self(batch["token_ids"], batch["attention_mask"], batch["images"])
        target_distribution = self._target_distribution(batch["labels"], logits.shape[1])
        log_probs = nn.functional.log_softmax(logits, dim=1)
        per_example_hard_loss = -(target_distribution * log_probs).sum(dim=1)
        unweighted_hard_loss = per_example_hard_loss.mean()
        if self.class_weights.numel():
            if self.class_weights.numel() != logits.shape[1]:
                raise ValueError("class_weights length must match the number of output classes.")
            weights = self.class_weights.to(per_example_hard_loss.device)[batch["labels"]]
            hard_loss = (per_example_hard_loss * weights).mean()
        else:
            hard_loss = unweighted_hard_loss
        if self.hparams.beta > 0:
            distill_loss = self._distill_loss(logits, batch["teacher_probs"])
        else:
            distill_loss = torch.zeros((), device=logits.device)
        loss = self.hparams.alpha * hard_loss + self.hparams.beta * distill_loss
        batch_size = batch["labels"].shape[0]
        self._log_learning_rate(stage, batch_size)
        self.log(
            f"{stage}/loss",
            loss,
            on_step=stage == "train",
            on_epoch=True,
            batch_size=batch_size,
            prog_bar=True,
        )
        self.log(
            f"{stage}/ce_loss",
            hard_loss,
            on_step=stage == "train",
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}/ce_loss_unweighted",
            unweighted_hard_loss,
            on_step=stage == "train",
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}/kl_loss",
            distill_loss,
            on_step=stage == "train",
            on_epoch=True,
            batch_size=batch_size,
        )
        self._totals[stage].update(logits, batch["labels"])
        self._update_confusion(stage, logits, batch["labels"])
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_index: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def _log_learning_rate(self, stage: str, batch_size: int) -> None:
        trainer = getattr(self, "_trainer", None)
        if stage != "train" or trainer is None or not trainer.optimizers:
            return
        learning_rate = float(trainer.optimizers[0].param_groups[0]["lr"])
        image_learning_rate = next(
            (
                float(group["lr"])
                for group in trainer.optimizers[0].param_groups
                if group.get("name") == "image_features"
            ),
            None,
        )
        self.log(
            "train/lr",
            learning_rate,
            on_step=True,
            on_epoch=False,
            batch_size=batch_size,
            prog_bar=True,
        )
        if image_learning_rate is not None:
            self.log(
                "train/image_lr",
                image_learning_rate,
                on_step=True,
                on_epoch=False,
                batch_size=batch_size,
                prog_bar=False,
            )

    def validation_step(self, batch: dict[str, torch.Tensor], batch_index: int) -> None:
        self._shared_step(batch, "val")
        self._collect_validation_plots(batch)

    def test_step(self, batch: dict[str, torch.Tensor], batch_index: int) -> None:
        self._shared_step(batch, "test")

    def _log_epoch_metrics(self, stage: str) -> None:
        metrics = self._totals[stage].metrics()
        self.log_dict({f"{stage}/{name}": value for name, value in metrics.items()}, sync_dist=True)
        self._totals[stage] = MulticlassTotals()

    def _update_confusion(
        self,
        stage: str,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        if stage not in self._confusions:
            return
        predictions = torch.argmax(logits.detach(), dim=1).cpu()
        expected = labels.detach().cpu()
        confusion = self._confusions[stage]
        for true_label, predicted_label in zip(expected.tolist(), predictions.tolist(), strict=True):
            confusion[int(true_label), int(predicted_label)] += 1

    def on_train_epoch_end(self) -> None:
        self._log_epoch_metrics("train")

    def on_validation_epoch_end(self) -> None:
        self._log_epoch_metrics("val")
        self._log_confusion_matrix("val")
        self._log_validation_plots()
        self._validation_plot_rows = []

    def on_test_epoch_end(self) -> None:
        self._log_epoch_metrics("test")
        self._log_confusion_matrix("test")

    def configure_optimizers(self) -> Any:
        image_parameters = [
            parameter
            for parameter in self.model.image_features.parameters()
            if parameter.requires_grad
        ]
        image_parameter_ids = {id(parameter) for parameter in image_parameters}
        main_parameters = [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad and id(parameter) not in image_parameter_ids
        ]
        parameter_groups: list[dict[str, Any]] = [
            {
                "params": main_parameters,
                "lr": self.hparams.learning_rate,
                "name": "main",
            }
        ]
        if image_parameters:
            parameter_groups.append(
                {
                    "params": image_parameters,
                    "lr": self.hparams.learning_rate
                    * self.hparams.image_learning_rate_scale,
                    "name": "image_features",
                }
            )
        optimizer = torch.optim.AdamW(
            parameter_groups,
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

    def _collect_validation_plots(self, batch: dict[str, torch.Tensor]) -> None:
        max_samples = int(self.hparams.validation_plot_samples)
        if max_samples == 0 or len(self._validation_plot_rows) >= max_samples:
            return
        if (int(self.current_epoch) + 1) % int(self.hparams.validation_plot_every_n_epochs) != 0:
            return
        if not hasattr(self.model, "image_features"):
            return
        remaining = max_samples - len(self._validation_plot_rows)
        images = batch["images"][:remaining]
        with torch.no_grad():
            query = self.model.encode_query(
                batch["token_ids"][:remaining],
                batch["attention_mask"][:remaining],
            )
            features_device = self.model.encode_image_features(images, query)
            features = features_device.detach().float().cpu()
            if getattr(self.model, "image_token_mode", "spatial") == "pooled":
                pooled_features = self.model.image_pool(features_device)
                projected_tokens = self.model.image_projection(
                    torch.flatten(pooled_features, 1)
                ).unsqueeze(1)
                projected_activation = projected_tokens.mean(dim=2).view(images.shape[0], 1, 1)
            else:
                feature_height, feature_width = features_device.shape[-2:]
                projected_tokens = self.model.image_projection(
                    features_device.flatten(2).transpose(1, 2)
                )
                projected_activation = projected_tokens.mean(dim=2).view(
                    images.shape[0],
                    feature_height,
                    feature_width,
                )
            projected_activation = projected_activation.detach().float().cpu()
            logits = self(batch["token_ids"][:remaining], batch["attention_mask"][:remaining], images)
            predictions = torch.argmax(logits.detach().cpu(), dim=1)
        for index in range(images.shape[0]):
            self._validation_plot_rows.append(
                {
                    "dataset_index": int(batch["dataset_index"][index].detach().cpu()),
                    "image": images[index].detach().float().cpu(),
                    "activation": features[index].mean(dim=0),
                    "projected_activation": projected_activation[index],
                    "label": int(batch["labels"][index].detach().cpu()),
                    "prediction": int(predictions[index]),
                    "student_prompt": str(batch.get("student_prompts", [""] * images.shape[0])[index]),
                }
            )

    @staticmethod
    def _denormalized_image(image: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=image.dtype).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=image.dtype).view(3, 1, 1)
        return (image * std + mean).clamp(0, 1)

    @staticmethod
    def _normalized_activation(activation: torch.Tensor) -> torch.Tensor:
        activation = activation.float()
        minimum = activation.min()
        maximum = activation.max()
        return (activation - minimum) / (maximum - minimum).clamp_min(1e-8)

    def _validation_plot(self, row: dict[str, Any]) -> wandb.Image:
        image = self._denormalized_image(row["image"]).permute(1, 2, 0).numpy()
        activation = self._normalized_activation(row["activation"]).numpy()
        projected_activation = self._normalized_activation(row["projected_activation"]).numpy()
        fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2))
        axes[0].imshow(image)
        title = (
            f"idx={row['dataset_index']} true={row['label']} pred={row['prediction']} "
            f"prompt={row['student_prompt']}"
        )
        axes[0].set_title(
            "\n".join(textwrap.wrap(title, width=42)),
            fontsize=9,
        )
        axes[0].axis("off")
        axes[1].imshow(activation, cmap="magma")
        axes[1].set_title("mean image encoder activation", fontsize=9)
        axes[1].axis("off")
        axes[2].imshow(projected_activation, cmap="magma")
        axes[2].set_title("mean projected image tokens", fontsize=9)
        axes[2].axis("off")
        fig.tight_layout()
        image_payload = wandb.Image(fig)
        plt.close(fig)
        return image_payload

    def _class_labels(self) -> list[str]:
        if self.num_outputs == 6:
            return ["0", "1", "2", "3", "4", "5+"]
        return [str(index) for index in range(self.num_outputs)]

    def _confusion_matrix_plot(self, stage: str, confusion: torch.Tensor) -> wandb.Image:
        counts = confusion.numpy()
        row_totals = counts.sum(axis=1, keepdims=True)
        normalized = counts / row_totals.clip(min=1)
        labels = self._class_labels()
        fig, ax = plt.subplots(figsize=(6.2, 5.4))
        cmap = plt.get_cmap("magma")
        image = ax.imshow(normalized, cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(range(len(labels)), labels=labels)
        ax.set_yticks(range(len(labels)), labels=labels)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title(f"{stage} output confusion matrix")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Row fraction")
        for row in range(counts.shape[0]):
            for col in range(counts.shape[1]):
                value = normalized[row, col]
                red, green, blue, _alpha = cmap(float(value))
                luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                ax.text(
                    col,
                    row,
                    str(int(counts[row, col])),
                    ha="center",
                    va="center",
                    color="black" if luminance > 0.5 else "white",
                    fontsize=8,
                )
        fig.tight_layout()
        image_payload = wandb.Image(fig)
        plt.close(fig)
        return image_payload

    def _log_confusion_matrix(self, stage: str) -> None:
        if stage not in self._confusions or self.logger is None:
            return
        if int(getattr(self, "global_rank", 0)) != 0:
            return
        confusion = self._confusions[stage].clone()
        self._confusions[stage].zero_()
        if int(confusion.sum()) == 0:
            return
        experiment = getattr(self.logger, "experiment", None)
        if experiment is None or not hasattr(experiment, "log"):
            return
        experiment.log(
            {
                f"{stage}_plots/confusion_matrix": self._confusion_matrix_plot(stage, confusion),
                "trainer/epoch": int(self.current_epoch),
            }
        )

    def _log_validation_plots(self) -> None:
        if not self._validation_plot_rows or self.logger is None:
            return
        if int(getattr(self, "global_rank", 0)) != 0:
            return
        experiment = getattr(self.logger, "experiment", None)
        if experiment is None or not hasattr(experiment, "log"):
            return
        experiment.log(
            {
                "validation_plots/image_encoding": [
                    self._validation_plot(row) for row in self._validation_plot_rows
                ],
                "trainer/epoch": int(self.current_epoch),
            }
        )
