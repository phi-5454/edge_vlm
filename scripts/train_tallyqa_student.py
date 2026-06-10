from __future__ import annotations

import html
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ["MPLBACKEND"] = "Agg"

import hydra
import lightning as L
import torch
import wandb
from dotenv import load_dotenv
from hydra.utils import to_absolute_path
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from torchinfo import summary as model_summary

from vlm_micro.student.data import TallyQAStudentDataModule
from vlm_micro.student.lightning import TallyQAStudentModule
from vlm_micro.student.model import StudentBaseline, architecture_report


def absolute_path(value: str) -> Path:
    return Path(to_absolute_path(value))


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def make_report(
    cfg: DictConfig,
    data: TallyQAStudentDataModule,
    model: StudentBaseline,
    report_path: Path,
) -> dict[str, Any]:
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "dataset": {
            "root": str(absolute_path(cfg.paths.dataset_root)),
            "prompt_embeddings": str(absolute_path(cfg.paths.prompt_embeddings)),
            "teacher_cache": (
                str(absolute_path(cfg.paths.teacher_cache)) if cfg.paths.teacher_cache else None
            ),
            "split_strategy": "deterministic 70/10/20 hash split grouped by image_id",
            "split_sizes": data.split_sizes(),
            "full_split_sizes": data.full_split_sizes(),
            "teacher_cache_coverage": data.cache_coverage(),
            "classes": int(cfg.model.num_outputs),
            "collapse_at": int(cfg.data.collapse_at),
            "prompt_embedding_rows": list(data.embedding_rows.shape),
            "prompt_token_shape": list(data.prompt_token_ids.shape),
        },
        "model": architecture_report(model),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def print_model_summary(model: StudentBaseline, cfg: DictConfig) -> None:
    print(
        model_summary(
            model,
            input_data=[
                torch.zeros((1, 4), dtype=torch.long),
                torch.ones((1, 4), dtype=torch.bool),
                torch.zeros((1, 3, int(cfg.model.image_size), int(cfg.model.image_size))),
            ],
            depth=4,
            col_names=("input_size", "output_size", "num_params", "trainable"),
            row_settings=("var_names",),
            verbose=0,
        )
    )


def class_weights_from_config(cfg: DictConfig, data: TallyQAStudentDataModule) -> list[float] | None:
    explicit_weights = cfg.distillation.get("class_weights", None)
    weight_mode = cfg.distillation.get("class_weight_mode", None)
    if explicit_weights is not None and weight_mode is not None:
        raise ValueError("Use either distillation.class_weights or class_weight_mode, not both.")
    if explicit_weights is not None:
        return [float(weight) for weight in explicit_weights]
    if weight_mode is None:
        return None
    if str(weight_mode) != "balanced":
        raise ValueError("distillation.class_weight_mode must be null or 'balanced'.")
    counts = data.label_counts("train")
    total = sum(counts.values())
    num_classes = int(cfg.model.num_outputs)
    if total <= 0 or any(counts[class_id] <= 0 for class_id in range(num_classes)):
        raise ValueError(f"Cannot compute balanced class weights from counts: {counts}")
    return [total / (num_classes * counts[class_id]) for class_id in range(num_classes)]


@hydra.main(version_base=None, config_path="../conf", config_name="tallyqa_student")
def main(cfg: DictConfig) -> None:
    L.seed_everything(int(cfg.seed), workers=True)
    load_dotenv(absolute_path(cfg.paths.wandb_env_file), override=False)

    beta = float(cfg.distillation.beta)
    require_teacher_cache = bool(cfg.data.get("require_teacher_cache", True))
    if beta > 0 and not require_teacher_cache:
        raise ValueError("data.require_teacher_cache=false requires distillation.beta=0.")
    teacher_cache = (
        absolute_path(cfg.paths.teacher_cache)
        if require_teacher_cache and cfg.paths.teacher_cache
        else None
    )
    missing_teacher_policy = (
        str(cfg.data.missing_teacher_policy) if require_teacher_cache else "keep"
    )
    prompt_class_filter_csv = (
        absolute_path(cfg.data.prompt_class_filter_csv)
        if cfg.data.get("prompt_class_filter_csv", None)
        else None
    )
    prompt_class_names_file = (
        absolute_path(cfg.data.prompt_class_names_file)
        if cfg.data.get("prompt_class_names_file", None)
        else None
    )
    curriculum_schedule = (
        absolute_path(cfg.data.curriculum_schedule)
        if cfg.data.get("curriculum_schedule", None)
        else None
    )
    data = TallyQAStudentDataModule(
        dataset_root=absolute_path(cfg.paths.dataset_root),
        prompt_embeddings=absolute_path(cfg.paths.prompt_embeddings),
        teacher_cache=teacher_cache,
        batch_size=int(cfg.data.batch_size),
        num_workers=int(cfg.data.num_workers),
        seed=int(cfg.seed),
        tensor_cache_size=int(cfg.data.tensor_cache_size),
        prefetch_factor=int(cfg.data.prefetch_factor),
        persistent_workers=bool(cfg.data.persistent_workers),
        pin_memory=bool(cfg.data.pin_memory),
        group_train_by_image=bool(cfg.data.group_train_by_image),
        shuffle_train=bool(cfg.data.get("shuffle_train", True)),
        train_sampling=str(cfg.data.get("train_sampling", "natural")),
        prompt_class_sampling_temperature=float(
            cfg.data.get("prompt_class_sampling_temperature", 0.5)
        ),
        train_epoch_size=(
            int(cfg.data.train_epoch_size)
            if cfg.data.get("train_epoch_size", None) is not None
            else None
        ),
        shuffle_block_size=int(cfg.data.shuffle_block_size),
        train_example_limit=(
            int(cfg.data.train_example_limit)
            if cfg.data.get("train_example_limit", None) is not None
            else None
        ),
        missing_teacher_policy=missing_teacher_policy,
        collapse_at=int(cfg.data.collapse_at),
        num_classes=int(cfg.model.num_outputs),
        prompt_class_filter_csv=prompt_class_filter_csv,
        min_prompt_accuracy=(
            float(cfg.data.min_prompt_accuracy)
            if cfg.data.get("min_prompt_accuracy", None) is not None
            else None
        ),
        prompt_class_names=(
            str(cfg.data.prompt_class_names)
            if cfg.data.get("prompt_class_names", None) is not None
            else None
        ),
        prompt_class_names_file=prompt_class_names_file,
        curriculum_schedule=curriculum_schedule,
        teacher_probability_temperature=float(
            cfg.data.get("teacher_probability_temperature", 1.0)
        ),
    )
    if beta > 0 and missing_teacher_policy == "keep":
        raise ValueError(
            f"beta={beta} cannot use data.missing_teacher_policy=keep because "
            "uncached prompts do not have distillation targets. Use filter."
        )

    model = StudentBaseline(
        embedding_rows=data.embedding_rows,
        freeze_embeddings=bool(cfg.model.freeze_embeddings),
        freeze_image_features=bool(cfg.model.get("freeze_image_features", False)),
        image_pretrained=bool(cfg.model.image_pretrained),
        query_dim=int(cfg.model.query_dim),
        image_dim=int(cfg.model.image_dim),
        fusion_dim=int(cfg.model.fusion_dim),
        fusion_depth=int(cfg.model.fusion_depth),
        fusion_heads=int(cfg.model.fusion_heads),
        fusion_mlp_ratio=int(cfg.model.fusion_mlp_ratio),
        dropout=float(cfg.model.dropout),
        image_backbone=str(cfg.model.image_backbone),
        image_feature_cutoff=cfg.model.get("image_feature_cutoff", "auto"),
        image_film_at=cfg.model.get("image_film_at", None),
        image_token_mode=str(cfg.model.get("image_token_mode", "spatial")),
        fusion_mode=str(cfg.model.get("fusion_mode", "transformer")),
        use_prompt_identity=bool(cfg.model.get("use_prompt_identity", True)),
        use_image_positional_embeddings=bool(
            cfg.model.get("use_image_positional_embeddings", True)
        ),
        image_position_tokens=int(cfg.model.get("image_position_tokens", 196)),
        zero_image_tokens=bool(cfg.model.get("zero_image_tokens", False)),
        zero_query_token=bool(cfg.model.get("zero_query_token", False)),
        num_outputs=int(cfg.model.num_outputs),
    )
    class_weights = class_weights_from_config(cfg, data)
    validation_plot_samples = (
        int(cfg.validation_plots.samples) if bool(cfg.validation_plots.enabled) else 0
    )
    module = TallyQAStudentModule(
        model=model,
        alpha=float(cfg.distillation.alpha),
        beta=beta,
        learning_rate=float(cfg.optimizer.learning_rate),
        warmup_start_learning_rate=float(cfg.optimizer.warmup_start_learning_rate),
        warmup_steps=int(cfg.optimizer.warmup_steps),
        lr_schedule=str(cfg.optimizer.get("lr_schedule", "warmup")),
        lr_decay_start_fraction=float(cfg.optimizer.get("lr_decay_start_fraction", 0.5)),
        lr_decay_start_step=(
            int(cfg.optimizer.lr_decay_start_step)
            if cfg.optimizer.get("lr_decay_start_step", None) is not None
            else None
        ),
        lr_final_learning_rate=(
            float(cfg.optimizer.lr_final_learning_rate)
            if cfg.optimizer.get("lr_final_learning_rate", None) is not None
            else None
        ),
        weight_decay=float(cfg.optimizer.weight_decay),
        image_learning_rate_scale=float(cfg.optimizer.get("image_learning_rate_scale", 1.0)),
        temperature=float(cfg.distillation.temperature),
        class_weights=class_weights,
        kl_class_weights=(
            [float(weight) for weight in cfg.distillation.kl_class_weights]
            if cfg.distillation.get("kl_class_weights", None) is not None
            else None
        ),
        target_distribution=str(cfg.distillation.target_distribution),
        local_soft_sigma=float(cfg.distillation.local_soft_sigma),
        local_soft_radius=int(cfg.distillation.local_soft_radius),
        validation_plot_samples=validation_plot_samples,
        validation_plot_every_n_epochs=int(cfg.validation_plots.every_n_epochs),
    )

    run_name = str(cfg.experiment.run_name)
    report_path = absolute_path(cfg.paths.report_dir) / f"{run_name}_architecture.json"
    report = make_report(cfg, data, model, report_path)
    print(json.dumps(report["dataset"], indent=2))
    print(json.dumps(report["model"]["parameter_counts"], indent=2))
    print_model_summary(model, cfg)
    print(f"Full architecture report: {report_path}")

    logger = WandbLogger(
        project=str(cfg.wandb.project),
        entity=str(cfg.wandb.entity) if cfg.wandb.entity else None,
        name=run_name,
        tags=list(cfg.wandb.tags),
        offline=str(cfg.wandb.mode) != "online",
        log_model=str(cfg.wandb.mode) == "online",
    )
    logger.log_hyperparams(
        {
            **OmegaConf.to_container(cfg, resolve=True),
            "parameter_counts": report["model"]["parameter_counts"],
            "split_sizes": data.split_sizes(),
        }
    )
    logger.experiment.log(
        {"model/architecture": wandb.Html(f"<pre>{html.escape(str(model))}</pre>")}
    )
    if bool(cfg.wandb.get("watch", {}).get("enabled", False)):
        logger.watch(
            module,
            log=str(cfg.wandb.watch.get("log", "all")),
            log_freq=int(cfg.wandb.watch.get("log_freq", 100)),
            log_graph=bool(cfg.wandb.watch.get("log_graph", False)),
        )

    checkpoint_monitor = str(cfg.trainer.early_stopping.get("monitor", "val/loss"))
    checkpoint_mode = str(cfg.trainer.early_stopping.get("mode", "min"))
    checkpoint = ModelCheckpoint(
        dirpath=absolute_path(cfg.paths.checkpoint_dir) / run_name,
        monitor=checkpoint_monitor,
        mode=checkpoint_mode,
        save_top_k=1,
        filename="{epoch:02d}",
        auto_insert_metric_name=False,
    )
    callbacks: list[Any] = [checkpoint]
    if bool(cfg.trainer.get("early_stopping", {}).get("enabled", False)):
        callbacks.append(
            EarlyStopping(
                monitor=checkpoint_monitor,
                mode=checkpoint_mode,
                patience=int(cfg.trainer.early_stopping.get("patience", 3)),
                min_delta=float(cfg.trainer.early_stopping.get("min_delta", 0.0)),
                verbose=True,
            )
        )
    trainer = L.Trainer(
        accelerator=str(cfg.trainer.accelerator),
        devices=cfg.trainer.devices,
        precision=str(cfg.trainer.precision),
        max_epochs=int(cfg.trainer.max_epochs),
        log_every_n_steps=int(cfg.trainer.log_every_n_steps),
        check_val_every_n_epoch=int(cfg.trainer.get("check_val_every_n_epoch", 1)),
        gradient_clip_val=cfg.trainer.get("gradient_clip_val", None),
        gradient_clip_algorithm=str(cfg.trainer.get("gradient_clip_algorithm", "norm")),
        limit_train_batches=cfg.trainer.get("limit_train_batches", None),
        limit_val_batches=cfg.trainer.get("limit_val_batches", None),
        limit_test_batches=cfg.trainer.get("limit_test_batches", None),
        reload_dataloaders_every_n_epochs=int(
            cfg.trainer.get("reload_dataloaders_every_n_epochs", 0)
        ),
        callbacks=callbacks,
        logger=logger,
    )
    trainer.fit(module, datamodule=data)
    test_results = trainer.test(module, datamodule=data, ckpt_path="best", weights_only=False)
    result_path = absolute_path(cfg.paths.report_dir) / f"{run_name}_results.json"
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": run_name,
        "best_model_monitor": checkpoint_monitor,
        "best_model_mode": checkpoint_mode,
        "best_model_path": checkpoint.best_model_path,
        "best_model_score": (
            float(checkpoint.best_model_score.detach().cpu())
            if checkpoint.best_model_score is not None
            else None
        ),
        "split_sizes": data.split_sizes(),
        "full_split_sizes": data.full_split_sizes(),
        "teacher_cache_coverage": data.cache_coverage(),
        "test_results": test_results,
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    logger.experiment.save(str(report_path), policy="now")
    logger.experiment.save(str(result_path), policy="now")
    wandb.finish()


if __name__ == "__main__":
    main()
