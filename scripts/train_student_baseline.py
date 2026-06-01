from __future__ import annotations

import html
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Colab shells can inherit Jupyter's inline backend, which is invalid outside
# the notebook kernel. Training does not render interactive figures.
os.environ["MPLBACKEND"] = "Agg"

import hydra
import lightning as L
import torch
import wandb
from dotenv import load_dotenv
from hydra.utils import to_absolute_path
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

from vlm_micro.student.data import StudentDataModule
from vlm_micro.student.lightning import StudentBaselineModule
from vlm_micro.student.model import StudentBaseline, architecture_report, load_teacher_embedding_rows


def absolute_path(value: str) -> Path:
    return Path(to_absolute_path(value))


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_or_create_embedding_rows(
    cfg: DictConfig,
    teacher_token_ids: tuple[int, ...],
) -> torch.Tensor:
    path = absolute_path(cfg.paths.compact_embeddings)
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if tuple(payload["teacher_token_ids"]) != teacher_token_ids:
            raise ValueError(f"{path} was built for a different compact token vocabulary.")
        return payload["embedding_rows"]

    rows = load_teacher_embedding_rows(
        model_name=str(cfg.teacher.model_name),
        teacher_token_ids=teacher_token_ids,
        local_files_only=bool(cfg.teacher.local_files_only),
        trust_remote_code=bool(cfg.teacher.trust_remote_code),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "teacher_model": str(cfg.teacher.model_name),
            "teacher_token_ids": teacher_token_ids,
            "embedding_rows": rows,
        },
        path,
    )
    return rows


def make_report(
    cfg: DictConfig,
    data: StudentDataModule,
    model: StudentBaseline,
    report_path: Path,
) -> dict[str, Any]:
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "dataset": {
            "root": str(absolute_path(cfg.paths.dataset_root)),
            "prompt_normalization": "student_prompt with final non-empty teacher prompt line removed",
            "tokenization": "pretokenized with the SmolVLM teacher tokenizer during dataset construction",
            "split_strategy": "deterministic 70/10/20 hash split grouped by student_image_id",
            "split_sizes": data.split_sizes(),
            "full_split_sizes": data.full_split_sizes(),
            "teacher_cache_coverage": data.cache_coverage(),
            "compact_vocabulary_rows": len(data.vocabulary.teacher_token_ids),
            "compact_embedding_rows_including_padding": data.vocabulary.size,
            "teacher_cache_records": len(data.teacher_targets),
        },
        "model": architecture_report(model),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


@hydra.main(version_base=None, config_path="../conf", config_name="student_baseline")
def main(cfg: DictConfig) -> None:
    L.seed_everything(int(cfg.seed), workers=True)
    load_dotenv(absolute_path(cfg.paths.wandb_env_file), override=False)

    teacher_cache = absolute_path(cfg.paths.teacher_cache) if cfg.paths.teacher_cache else None
    data = StudentDataModule(
        dataset_root=absolute_path(cfg.paths.dataset_root),
        teacher_cache=teacher_cache,
        batch_size=int(cfg.data.batch_size),
        num_workers=int(cfg.data.num_workers),
        image_size=int(cfg.model.image_size),
        seed=int(cfg.seed),
        row_group_cache_size=int(cfg.data.row_group_cache_size),
        tensor_cache_size=int(cfg.data.tensor_cache_size),
        prefetch_factor=int(cfg.data.prefetch_factor),
        persistent_workers=bool(cfg.data.persistent_workers),
        pin_memory=bool(cfg.data.pin_memory),
        group_train_by_image=bool(cfg.data.group_train_by_image),
        shuffle_block_size=int(cfg.data.shuffle_block_size),
        missing_teacher_policy=str(cfg.data.missing_teacher_policy),
    )
    alpha = float(cfg.distillation.alpha)
    if alpha < 1 and str(cfg.data.missing_teacher_policy) == "keep":
        raise ValueError(
            f"alpha={alpha} cannot use data.missing_teacher_policy=keep because "
            "uncached prompts do not have distillation targets. Use filter."
        )

    embedding_rows = load_or_create_embedding_rows(cfg, data.vocabulary.teacher_token_ids)
    model = StudentBaseline(
        embedding_rows=embedding_rows,
        freeze_embeddings=bool(cfg.model.freeze_embeddings),
        image_pretrained=bool(cfg.model.image_pretrained),
        query_dim=int(cfg.model.query_dim),
        image_dim=int(cfg.model.image_dim),
        fusion_dim=int(cfg.model.fusion_dim),
        fusion_depth=int(cfg.model.fusion_depth),
        fusion_heads=int(cfg.model.fusion_heads),
        fusion_mlp_ratio=int(cfg.model.fusion_mlp_ratio),
        dropout=float(cfg.model.dropout),
    )
    module = StudentBaselineModule(
        model=model,
        alpha=alpha,
        learning_rate=float(cfg.optimizer.learning_rate),
        warmup_start_learning_rate=float(cfg.optimizer.warmup_start_learning_rate),
        warmup_steps=int(cfg.optimizer.warmup_steps),
        weight_decay=float(cfg.optimizer.weight_decay),
        distill_kind=str(cfg.distillation.kind),
        temperature=float(cfg.distillation.temperature),
    )

    run_name = str(cfg.experiment.run_name)
    report_path = absolute_path(cfg.paths.report_dir) / f"{run_name}_architecture.json"
    report = make_report(cfg, data, model, report_path)
    print(json.dumps(report["dataset"], indent=2))
    print(json.dumps(report["model"]["parameter_counts"], indent=2))
    print(report["model"]["architecture"])

    logger = WandbLogger(
        project=str(cfg.wandb.project),
        entity=str(cfg.wandb.entity) if cfg.wandb.entity else None,
        name=run_name,
        tags=list(cfg.wandb.tags),
        offline=str(cfg.wandb.mode) != "online",
        log_model=True,
    )
    logger.log_hyperparams(
        {
            **OmegaConf.to_container(cfg, resolve=True),
            "parameter_counts": report["model"]["parameter_counts"],
            "compact_vocabulary_rows": len(data.vocabulary.teacher_token_ids),
            "split_sizes": data.split_sizes(),
        }
    )
    logger.experiment.log(
        {"model/architecture": wandb.Html(f"<pre>{html.escape(str(model))}</pre>")}
    )

    checkpoint = ModelCheckpoint(
        dirpath=absolute_path(cfg.paths.checkpoint_dir) / run_name,
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        filename="{epoch:02d}",
        auto_insert_metric_name=False,
    )
    trainer = L.Trainer(
        accelerator=str(cfg.trainer.accelerator),
        devices=cfg.trainer.devices,
        precision=str(cfg.trainer.precision),
        max_epochs=int(cfg.trainer.max_epochs),
        log_every_n_steps=int(cfg.trainer.log_every_n_steps),
        callbacks=[checkpoint],
        logger=logger,
    )
    trainer.fit(module, datamodule=data)
    trainer.test(module, datamodule=data, ckpt_path="best")
    logger.experiment.save(str(report_path), policy="now")
    wandb.finish()


if __name__ == "__main__":
    main()
