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
import matplotlib.pyplot as plt
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


def format_count(value: int | float) -> str:
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(int(value))


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


def model_summary_text(model: StudentBaseline, cfg: DictConfig, depth: int = 4) -> str:
    return str(
        model_summary(
            model,
            input_data=[
                torch.zeros((1, 4), dtype=torch.long),
                torch.ones((1, 4), dtype=torch.bool),
                torch.zeros((1, 3, int(cfg.model.image_size), int(cfg.model.image_size))),
            ],
            depth=depth,
            col_names=("input_size", "output_size", "num_params", "trainable"),
            row_settings=("var_names",),
            verbose=0,
        )
    )


def parameter_rows(model: torch.nn.Module) -> list[dict[str, Any]]:
    rows = []
    for name, parameter in model.named_parameters():
        top_level = name.split(".", 1)[0]
        rows.append(
            {
                "name": name,
                "top_level": top_level,
                "shape": list(parameter.shape),
                "parameters": int(parameter.numel()),
                "trainable": bool(parameter.requires_grad),
            }
        )
    return rows


def top_level_parameter_rows(model: torch.nn.Module) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in parameter_rows(model):
        group = grouped.setdefault(
            row["top_level"],
            {"module": row["top_level"], "total": 0, "trainable": 0, "frozen": 0},
        )
        group["total"] += row["parameters"]
        if row["trainable"]:
            group["trainable"] += row["parameters"]
        else:
            group["frozen"] += row["parameters"]
    return sorted(grouped.values(), key=lambda item: item["total"], reverse=True)


def write_parameter_chart(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    labels = [row["module"] for row in rows]
    trainable = [row["trainable"] for row in rows]
    frozen = [row["frozen"] for row in rows]
    height = max(3.5, 0.42 * len(rows) + 1.4)
    fig, ax = plt.subplots(figsize=(9, height))
    y_positions = list(range(len(rows)))
    ax.barh(y_positions, frozen, label="frozen", color="#9aa4b2")
    ax.barh(y_positions, trainable, left=frozen, label="trainable", color="#2f80ed")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Parameters")
    ax.set_title("Model Parameters by Top-Level Module")
    ax.xaxis.set_major_formatter(lambda value, _pos: format_count(value))
    ax.legend(loc="lower right")
    for y_pos, row in zip(y_positions, rows, strict=True):
        ax.text(
            row["total"],
            y_pos,
            f" {format_count(row['total'])}",
            va="center",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_model_report_bundle(
    model: StudentBaseline,
    cfg: DictConfig,
    report: dict[str, Any],
    report_path: Path,
) -> dict[str, Path]:
    base = report_path.with_suffix("")
    summary_path = base.with_name(base.name + "_summary.txt")
    markdown_path = base.with_name(base.name + "_readable.md")
    html_path = base.with_name(base.name + "_readable.html")
    chart_path = base.with_name(base.name + "_parameters.png")
    parameters_path = base.with_name(base.name + "_parameters.json")

    summary = model_summary_text(model, cfg)
    rows = top_level_parameter_rows(model)
    detailed_rows = parameter_rows(model)
    parameters_path.write_text(json.dumps(detailed_rows, indent=2) + "\n", encoding="utf-8")
    write_parameter_chart(rows, chart_path)

    parameter_counts = report["model"]["parameter_counts"]
    overview_rows = [
        ["total", format_count(parameter_counts["total"]), str(parameter_counts["total"])],
        ["trainable", format_count(parameter_counts["trainable"]), str(parameter_counts["trainable"])],
        ["frozen", format_count(parameter_counts["frozen"]), str(parameter_counts["frozen"])],
    ]
    module_rows = [
        [
            str(row["module"]),
            format_count(row["total"]),
            format_count(row["trainable"]),
            format_count(row["frozen"]),
        ]
        for row in rows
    ]
    model_info = report["model"]
    dataset_info = report["dataset"]
    config_lines = [
        f"- backbone: `{model_info.get('image_backbone')}`",
        f"- image feature cutoff: `{model_info.get('image_feature_cutoff')}`",
        f"- image token mode: `{model_info.get('image_token_mode')}`",
        f"- fusion mode: `{model_info.get('fusion_mode')}`",
        f"- fusion depth / heads / MLP ratio: `{cfg.model.fusion_depth}` / `{cfg.model.fusion_heads}` / `{cfg.model.fusion_mlp_ratio}`",
        f"- FiLM: `{model_info.get('image_film_indices')}` at `{model_info.get('image_film_position')}`",
        f"- prompt identity: `{model_info.get('use_prompt_identity')}`",
        f"- image positional embeddings: `{model_info.get('use_image_positional_embeddings')}` over `{model_info.get('image_position_tokens')}` tokens",
        f"- zero image / zero query: `{model_info.get('zero_image_tokens')}` / `{model_info.get('zero_query_token')}`",
        f"- dataset split sizes: `{dataset_info.get('split_sizes')}`",
    ]
    markdown = "\n\n".join(
        [
            f"# {cfg.experiment.run_name} Model Report",
            "## Architecture",
            "\n".join(config_lines),
            "## Parameter Totals",
            markdown_table(["group", "compact", "exact"], overview_rows),
            "## Top-Level Modules",
            markdown_table(["module", "total", "trainable", "frozen"], module_rows),
            "## Torchinfo",
            f"```text\n{summary}\n```",
            "## Raw Module Tree",
            f"```text\n{model}\n```",
        ]
    )
    summary_path.write_text(summary + "\n", encoding="utf-8")
    markdown_path.write_text(markdown + "\n", encoding="utf-8")
    html_body = (
        "<html><body>"
        f"<h1>{html.escape(str(cfg.experiment.run_name))} Model Report</h1>"
        "<h2>Parameter Totals</h2>"
        f"<pre>{html.escape(markdown_table(['group', 'compact', 'exact'], overview_rows))}</pre>"
        "<h2>Top-Level Modules</h2>"
        f"<pre>{html.escape(markdown_table(['module', 'total', 'trainable', 'frozen'], module_rows))}</pre>"
        "<h2>Torchinfo</h2>"
        f"<pre>{html.escape(summary)}</pre>"
        "<h2>Raw Module Tree</h2>"
        f"<pre>{html.escape(str(model))}</pre>"
        "</body></html>"
    )
    html_path.write_text(html_body, encoding="utf-8")
    return {
        "summary": summary_path,
        "markdown": markdown_path,
        "html": html_path,
        "parameter_chart": chart_path,
        "parameters": parameters_path,
    }


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
        prompt_class_sampling_end_temperature=(
            float(cfg.data.prompt_class_sampling_end_temperature)
            if cfg.data.get("prompt_class_sampling_end_temperature", None) is not None
            else None
        ),
        prompt_class_sampling_decay_steps=(
            int(cfg.data.prompt_class_sampling_decay_steps)
            if cfg.data.get("prompt_class_sampling_decay_steps", None) is not None
            else None
        ),
        prompt_class_sampling_ramp_start_step=(
            int(cfg.data.prompt_class_sampling_ramp_start_step)
            if cfg.data.get("prompt_class_sampling_ramp_start_step", None) is not None
            else None
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
        max_epochs=int(cfg.trainer.max_epochs),
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
        image_film_position=str(cfg.model.get("image_film_position", "post_block")),
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
        beta_ramp_start_step=(
            int(cfg.distillation.beta_ramp_start_step)
            if cfg.distillation.get("beta_ramp_start_step", None) is not None
            else None
        ),
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
            else class_weights
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
    readable_report_paths = write_model_report_bundle(model, cfg, report, report_path)
    print("Dataset:")
    print(json.dumps(report["dataset"], indent=2))
    print("Parameter counts:")
    print(json.dumps(report["model"]["parameter_counts"], indent=2))
    print(readable_report_paths["summary"].read_text(encoding="utf-8"))
    print(f"Full architecture report: {report_path}")
    print(f"Readable architecture report: {readable_report_paths['markdown']}")
    print(f"Parameter chart: {readable_report_paths['parameter_chart']}")

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
        {
            "model/architecture": wandb.Html(
                readable_report_paths["html"].read_text(encoding="utf-8")
            ),
            "model/parameter_chart": wandb.Image(str(readable_report_paths["parameter_chart"])),
        }
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
    for path in readable_report_paths.values():
        logger.experiment.save(str(path), policy="now")
    logger.experiment.save(str(result_path), policy="now")
    wandb.finish()


if __name__ == "__main__":
    main()
