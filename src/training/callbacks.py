from __future__ import annotations

from typing import Any

import lightning as L
import torch

from config import SampleLoggingConfig, SizeLoggingConfig


def model_size_metrics(model: torch.nn.Module, target_quantized_bits: int | None) -> dict[str, float | int]:
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    buffer_count = sum(buffer.numel() for buffer in model.buffers())
    parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    buffer_bytes = sum(buffer.numel() * buffer.element_size() for buffer in model.buffers())
    metrics: dict[str, float | int] = {
        "model/parameters": parameter_count,
        "model/trainable_parameters": trainable_parameter_count,
        "model/buffers": buffer_count,
        "model/uncompressed_parameter_bytes": parameter_bytes,
        "model/uncompressed_buffer_bytes": buffer_bytes,
        "model/uncompressed_total_bytes": parameter_bytes + buffer_bytes,
        "model/uncompressed_total_mib": (parameter_bytes + buffer_bytes) / 1024**2,
    }
    if target_quantized_bits is not None:
        quantized_parameter_bytes = (parameter_count * target_quantized_bits + 7) // 8
        metrics.update(
            {
                "model/target_quantized_bits": target_quantized_bits,
                "model/estimated_quantized_parameter_bytes": quantized_parameter_bytes,
                "model/estimated_quantized_parameter_mib": quantized_parameter_bytes / 1024**2,
                "model/estimated_compression_ratio": parameter_bytes / quantized_parameter_bytes
                if quantized_parameter_bytes
                else 0.0,
            }
        )
    return metrics


class ModelSizeLogger(L.Callback):
    """Logs model size metrics once the model has been materialized."""

    def __init__(self, config: SizeLoggingConfig) -> None:
        self.config = config

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if not self.config.enabled:
            return
        metrics = model_size_metrics(pl_module.model, self.config.target_quantized_bits)
        trainer.logger.log_metrics(metrics, step=0)
        if hasattr(trainer.logger, "experiment") and hasattr(trainer.logger.experiment, "summary"):
            trainer.logger.experiment.summary.update(metrics)


class WandbGenerationLogger(L.Callback):
    """Logs occasional decoded model generations to W&B."""

    def __init__(self, processor: Any, config: SampleLoggingConfig) -> None:
        self.processor = processor
        self.config = config

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: dict[str, Any],
        batch_idx: int,
    ) -> None:
        if not self.config.enabled:
            return
        if trainer.global_step == 0 or trainer.global_step % self.config.every_n_train_steps != 0:
            return
        if not hasattr(trainer.logger, "experiment"):
            return

        rows = self._generate_rows(pl_module, batch, trainer.global_step)
        if not rows:
            return

        import wandb

        table = wandb.Table(columns=["step", "prompt", "generation"])
        for row in rows:
            table.add_data(row["step"], row["prompt"], row["generation"])
        trainer.logger.experiment.log({"samples/generations": table}, step=trainer.global_step)

    def _generate_rows(
        self,
        pl_module: L.LightningModule,
        batch: dict[str, Any],
        step: int,
    ) -> list[dict[str, Any]]:
        sample_count = min(self.config.num_samples, batch["input_ids"].shape[0])
        inputs = {
            key: value[:sample_count]
            for key, value in batch.items()
            if key != "labels" and isinstance(value, torch.Tensor)
        }

        was_training = pl_module.training
        pl_module.eval()
        with torch.inference_mode():
            generated_ids = pl_module.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
            )
        if was_training:
            pl_module.train()

        prompts = self.processor.batch_decode(inputs["input_ids"], skip_special_tokens=False)
        generations = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return [
            {
                "step": step,
                "prompt": prompt,
                "generation": generation,
            }
            for prompt, generation in zip(prompts, generations, strict=True)
        ]
