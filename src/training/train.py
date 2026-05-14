from __future__ import annotations

from pathlib import Path

import lightning as L
from dotenv import load_dotenv
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from config import TrainConfig, config_to_dict
from training.callbacks import ModelSizeLogger, WandbGenerationLogger
from training.datamodule import CauldronDataModule
from training.module import SmolVlmModule


def load_tracking_env(env_file: str | None) -> None:
    if not env_file:
        return
    path = Path(env_file).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"W&B env file not found: {path}")
    load_dotenv(path, override=False)


def train(config: TrainConfig) -> None:
    from transformers import AutoProcessor

    L.seed_everything(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    load_tracking_env(config.tracking.env_file)

    processor = AutoProcessor.from_pretrained(
        config.model.model_name,
        trust_remote_code=config.model.trust_remote_code,
    )
    model = SmolVlmModule(config.model)
    data = CauldronDataModule(config.data, processor=processor, batch_size=config.batch_size)

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
            monitor="train/loss_epoch",
            save_top_k=1,
            mode="min",
        ),
    ]
    if config.size_logging.enabled:
        callbacks.append(ModelSizeLogger(config.size_logging))
    if config.sample_logging.enabled:
        callbacks.append(WandbGenerationLogger(processor, config.sample_logging))

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
