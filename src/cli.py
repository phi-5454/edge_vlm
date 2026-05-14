from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from config import cache_teacher_config_from_hydra, distill_config_from_hydra, train_config_from_hydra


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.command == "train":
        from training.train import train

        train(train_config_from_hydra(cfg))
    elif cfg.command == "distill":
        from training.distill import distill

        distill(distill_config_from_hydra(cfg))
    elif cfg.command == "cache_teacher":
        from training.teacher_cache import cache_teacher

        cache_teacher(cache_teacher_config_from_hydra(cfg))
    elif cfg.command == "profile":
        from profiling.benchmark import run_profile

        run_profile(
            model_name=cfg.profile.model,
            output_path=Path(cfg.profile.output),
            steps=cfg.profile.steps,
            warmup=cfg.profile.warmup,
        )
    elif cfg.command == "decision":
        from docs import create_decision

        print(create_decision(cfg.decision.slug))
    else:
        raise ValueError(f"Unsupported command: {cfg.command}")
