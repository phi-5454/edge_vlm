from omegaconf import OmegaConf

from config import DataConfig, train_config_from_hydra
from training.callbacks import model_size_metrics
from training.datamodule import CauldronDataModule


def test_train_config_from_hydra() -> None:
    cfg = OmegaConf.create(
        {
            "train": {
                "seed": 42,
                "output_dir": "artifacts/runs",
                "batch_size": 2,
                "max_epochs": 1,
                "precision": "bf16-mixed",
                "accumulate_grad_batches": 8,
                "log_every_n_steps": 10,
                "data": {
                    "dataset_name": "HuggingFaceM4/the_cauldron",
                    "dataset_config": "rendered_text",
                    "train_split": "train",
                    "val_split": None,
                    "text_column": "texts",
                    "image_column": "images",
                    "image_token": "<image>",
                    "max_samples": 4,
                    "num_workers": 4,
                },
                "model": {
                    "model_name": "HuggingFaceTB/SmolVLM-256M-Instruct",
                    "trust_remote_code": True,
                    "learning_rate": 0.00002,
                    "weight_decay": 0.01,
                },
                "tracking": {
                    "project": "edge-vlm",
                    "entity": None,
                    "run_name": "smoke",
                    "env_file": "../wandb_api_key.env",
                    "tags": ["smolvlm"],
                    "offline": True,
                },
                "sample_logging": {
                    "enabled": True,
                    "every_n_train_steps": 2,
                    "num_samples": 1,
                    "max_new_tokens": 8,
                },
                "size_logging": {
                    "enabled": True,
                    "target_quantized_bits": 8,
                },
            }
        }
    )

    config = train_config_from_hydra(cfg)

    assert config.batch_size == 2
    assert config.data.max_samples == 4
    assert config.data.dataset_name == "HuggingFaceM4/the_cauldron"
    assert config.data.dataset_config == "rendered_text"
    assert config.data.image_token == "<image>"
    assert config.tracking.env_file == "../wandb_api_key.env"
    assert config.tracking.offline is True
    assert config.sample_logging.every_n_train_steps == 2
    assert config.sample_logging.max_new_tokens == 8
    assert config.size_logging.target_quantized_bits == 8


def test_cauldron_text_includes_image_token() -> None:
    datamodule = CauldronDataModule(DataConfig(), processor=None, batch_size=1)

    assert datamodule._with_image_token("Describe this.") == "<image>\nDescribe this."
    assert datamodule._with_image_token("<image>\nDescribe this.") == "<image>\nDescribe this."


def test_model_size_metrics_include_quantized_estimate() -> None:
    import torch

    model = torch.nn.Linear(4, 2)

    metrics = model_size_metrics(model, target_quantized_bits=8)

    assert metrics["model/parameters"] == 10
    assert metrics["model/uncompressed_parameter_bytes"] == 40
    assert metrics["model/estimated_quantized_parameter_bytes"] == 10
    assert metrics["model/estimated_compression_ratio"] == 4
