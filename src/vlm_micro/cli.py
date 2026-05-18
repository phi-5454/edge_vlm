from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf


@dataclass(frozen=True)
class ArtifactSummary:
    path: str
    exists: bool
    bytes: int | None
    sha256: str | None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_artifacts(paths: Iterable[str]) -> list[ArtifactSummary]:
    summaries: list[ArtifactSummary] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            summaries.append(ArtifactSummary(raw_path, False, None, None))
            continue
        summaries.append(
            ArtifactSummary(
                path=raw_path,
                exists=True,
                bytes=path.stat().st_size,
                sha256=_sha256(path),
            )
        )
    return summaries


def write_artifact_report(cfg: DictConfig, cli_files: list[str]) -> Path:
    files = cli_files or list(cfg.artifact_report.files)
    output = Path(cfg.artifact_report.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "files": [asdict(summary) for summary in summarize_artifacts(files)],
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output


def record_decision(cfg: DictConfig) -> Path:
    slug = str(cfg.decision.slug).strip().lower().replace(" ", "-")
    if not slug or slug == "unnamed-decision":
        raise ValueError("Set decision.slug=<short-name>.")
    directory = Path("docs/decisions")
    directory.mkdir(parents=True, exist_ok=True)
    existing = sorted(directory.glob("[0-9][0-9][0-9][0-9]-*.md"))
    number = len(existing) + 1
    path = directory / f"{number:04d}-{slug}.md"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(
        "\n".join(
            [
                f"# {number:04d} {slug.replace('-', ' ').title()}",
                "",
                f"Date: {date.today().isoformat()}",
                "",
                "## Status",
                "",
                "Proposed",
                "",
                "## Context",
                "",
                "TODO",
                "",
                "## Decision",
                "",
                "TODO",
                "",
                "## Evidence",
                "",
                "- W&B run:",
                "- Artifact report:",
                "- Profiling log:",
                "",
                "## Consequences",
                "",
                "TODO",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def run(cfg: DictConfig, positional: list[str] | None = None) -> None:
    positional = positional or []
    command = str(cfg.command)
    if command == "artifact-report":
        output = write_artifact_report(cfg, positional)
        print(f"Wrote {output}")
    elif command == "record-decision":
        output = record_decision(cfg)
        print(f"Wrote {output}")
    else:
        raise ValueError(f"Unknown command: {command}")


def main() -> None:
    config_dir = Path.cwd() / "conf"
    args = sys.argv[1:]
    command_names = {"artifact-report", "record-decision"}
    command_override: list[str] = []
    if args and args[0] in command_names:
        command_override = [f"command={args.pop(0)}"]

    hydra_overrides: list[str] = []
    positional: list[str] = []
    for arg in args:
        if "=" in arg or arg.startswith(("+", "~")):
            hydra_overrides.append(arg)
        else:
            positional.append(arg)

    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="config", overrides=command_override + hydra_overrides)
    run(cfg, positional)


if __name__ == "__main__":
    main()
