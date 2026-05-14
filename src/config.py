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
class TinyVlmConfig:
    vocab_size: int = 49_152
    max_text_tokens: int = 64
    image_size: int = 224
    text_width: int = 128
    text_layers: int = 2
    text_heads: int = 4
    projection_dim: int = 128
    fusion_hidden_dim: int = 256
    teacher_dim: int = 576
    num_answer_classes: int = 0
    pretrained_vision: bool = True
    freeze_vision: bool = False


@dataclass(slots=True)
class DistillationConfig:
    teacher_embedding_loss_weight: float = 1.0
    image_text_contrastive_loss_weight: float = 0.1
    temperature: float = 0.07
    normalize_teacher_targets: bool = True


@dataclass(slots=True)
class TeacherCacheConfig:
    enabled: bool = False
    path: str = "artifacts/teacher_cache/cauldron_rendered_text.pt"
    batch_size: int = 1


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


@dataclass(slots=True)
class DistillConfig:
    seed: int = 42
    output_dir: str = "artifacts/runs/tiny-vlm-distill"
    batch_size: int = 1
    max_epochs: int = 1
    precision: str = "bf16-mixed"
    accumulate_grad_batches: int = 8
    log_every_n_steps: int = 10
    data: DataConfig = field(default_factory=DataConfig)
    teacher: ModelConfig = field(default_factory=ModelConfig)
    student: TinyVlmConfig = field(default_factory=TinyVlmConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    teacher_cache: TeacherCacheConfig = field(default_factory=TeacherCacheConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    size_logging: SizeLoggingConfig = field(default_factory=SizeLoggingConfig)


@dataclass(slots=True)
class CacheTeacherConfig:
    seed: int = 42
    output_path: str = "artifacts/teacher_cache/cauldron_rendered_text.pt"
    device: str = "auto"
    data: DataConfig = field(default_factory=DataConfig)
    teacher: ModelConfig = field(default_factory=ModelConfig)


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


def distill_config_from_hydra(cfg: DictConfig) -> DistillConfig:
    values = OmegaConf.to_container(cfg.distill, resolve=True)
    return DistillConfig(
        seed=values["seed"],
        output_dir=values["output_dir"],
        batch_size=values["batch_size"],
        max_epochs=values["max_epochs"],
        precision=values["precision"],
        accumulate_grad_batches=values["accumulate_grad_batches"],
        log_every_n_steps=values["log_every_n_steps"],
        data=DataConfig(**values["data"]),
        teacher=ModelConfig(**values["teacher"]),
        student=TinyVlmConfig(**values["student"]),
        distillation=DistillationConfig(**values["distillation"]),
        teacher_cache=TeacherCacheConfig(**values["teacher_cache"]),
        tracking=TrackingConfig(**values["tracking"]),
        size_logging=SizeLoggingConfig(**values["size_logging"]),
    )


def cache_teacher_config_from_hydra(cfg: DictConfig) -> CacheTeacherConfig:
    values = OmegaConf.to_container(cfg.cache_teacher, resolve=True)
    return CacheTeacherConfig(
        seed=values["seed"],
        output_path=values["output_path"],
        device=values["device"],
        data=DataConfig(**values["data"]),
        teacher=ModelConfig(**values["teacher"]),
    )


def config_to_dict(config: Any) -> dict[str, Any]:
    return _asdict(config)
