from __future__ import annotations

from datetime import date
from pathlib import Path


def create_decision(slug: str, directory: Path = Path("docs/decisions")) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    existing = sorted(directory.glob("[0-9][0-9][0-9][0-9]-*.md"))
    next_id = len(existing) + 1
    path = directory / f"{next_id:04d}-{slug}.md"
    if path.exists():
        return path
    path.write_text(
        "\n".join(
            [
                f"# {slug.replace('-', ' ').title()}",
                "",
                f"Date: {date.today().isoformat()}",
                "",
                "## Context",
                "",
                "What constraint, measurement, or problem motivated this decision?",
                "",
                "## Decision",
                "",
                "What will change?",
                "",
                "## Evidence",
                "",
                "- Profiling run:",
                "- Training run:",
                "- Relevant benchmark:",
                "",
                "## Consequences",
                "",
                "What improves, what gets worse, and what should be revisited?",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path
