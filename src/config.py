from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from omegaconf import DictConfig, OmegaConf


@dataclass(slots=True)
class DataConfig:
    dataset_name: str = "HuggingFaceM4/the_cauldron"
    dataset_config: str | None = None
    train_split: str = "train"
    val_split: str | None = None
    text_column: str = "texts"
    image_column: str = "images"
    image_token: str = "<image>"
    max_samples: int | None = None
    num_workers: int = 4


@dataclass(slots=True)
class ModelConfig:
    model_name: str = "HuggingFaceTB/SmolVLM-256M-Instruct"
    trust_remote_code: bool = True
    learning_rate: float = 2e-5
    weight_decay: float = 0.01


@dataclass(slots=True)
class TrackingConfig:
    project: str = "edge-vlm"
    entity: str | None = None
    run_name: str | None = None
    tags: list[str] = field(default_factory=lambda: ["smolvlm", "cauldron", "microcontroller"])
    offline: bool = False
    env_file: str | None = "../wandb_api_key.env"


@dataclass(slots=True)
class SampleLoggingConfig:
    enabled: bool = True
    every_n_train_steps: int = 50
    num_samples: int = 1
    max_new_tokens: int = 32


@dataclass(slots=True)
class SizeLoggingConfig:
    enabled: bool = True
    target_quantized_bits: int | None = 8


@dataclass(slots=True)
class TrainConfig:
    seed: int = 42
    output_dir: str = "artifacts/runs"
    batch_size: int = 1
    max_epochs: int = 1
    precision: str = "bf16-mixed"
    accumulate_grad_batches: int = 8
    log_every_n_steps: int = 10
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    sample_logging: SampleLoggingConfig = field(default_factory=SampleLoggingConfig)
    size_logging: SizeLoggingConfig = field(default_factory=SizeLoggingConfig)


def _asdict(config: Any) -> dict[str, Any]:
    if not hasattr(config, "__dataclass_fields__"):
        return config
    return {key: _asdict(getattr(config, key)) for key in config.__dataclass_fields__}


def train_config_from_hydra(cfg: DictConfig) -> TrainConfig:
    values = OmegaConf.to_container(cfg.train, resolve=True)
    return TrainConfig(
        seed=values["seed"],
        output_dir=values["output_dir"],
        batch_size=values["batch_size"],
        max_epochs=values["max_epochs"],
        precision=values["precision"],
        accumulate_grad_batches=values["accumulate_grad_batches"],
        log_every_n_steps=values["log_every_n_steps"],
        data=DataConfig(**values["data"]),
        model=ModelConfig(**values["model"]),
        tracking=TrackingConfig(**values["tracking"]),
        sample_logging=SampleLoggingConfig(**values["sample_logging"]),
        size_logging=SizeLoggingConfig(**values["size_logging"]),
    )


def config_to_dict(config: Any) -> dict[str, Any]:
    return _asdict(config)
