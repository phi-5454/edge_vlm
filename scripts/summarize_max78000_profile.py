#!/usr/bin/env python3
"""Summarize MAX78000 ai8xize and board serial profiling logs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any


INTEGER = r"([0-9][0-9,]*)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthesis-log", type=Path, default=Path("artifacts/profiles/max78000/ai8xize.log"))
    parser.add_argument("--serial-jsonl", type=Path, default=Path("artifacts/profiles/max78000/preview.jsonl"))
    parser.add_argument("--generated-project", type=Path, default=Path("artifacts/exports/max78000/generated"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/profiles/max78000/report.json"))
    return parser.parse_args()


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def int_match(pattern: str, text: str) -> int | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match is None:
        return None
    return int(match.group(1).replace(",", ""))


def float_match(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match is None:
        return None
    return float(match.group(1))


def parse_synthesis_log(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    text = path.read_text(encoding="utf-8", errors="replace")
    hardware = re.search(
        rf"Hardware:\s*{INTEGER}\s*ops\s*\({INTEGER}\s*macc;\s*{INTEGER}\s*comp;\s*{INTEGER}\s*add;\s*{INTEGER}\s*mul;\s*{INTEGER}\s*bitwise\)",
        text,
        flags=re.IGNORECASE,
    )
    software = re.search(
        rf"Software:\s*{INTEGER}\s*ops\s*\({INTEGER}\s*macc;\s*{INTEGER}\s*comp\)",
        text,
        flags=re.IGNORECASE,
    )
    weight = re.search(
        rf"Weight memory:\s*{INTEGER}\s*bytes out of\s*{INTEGER}\s*bytes total\s*\(([0-9.]+)%\)",
        text,
        flags=re.IGNORECASE,
    )
    bias = re.search(
        rf"Bias memory:\s*{INTEGER}\s*bytes out of\s*{INTEGER}\s*bytes total\s*\(([0-9.]+)%\)",
        text,
        flags=re.IGNORECASE,
    )
    layers: list[dict[str, int]] = []
    for match in re.finditer(
        rf"(?:Layer|L)(?:ayer)?\s*([0-9]+):?\s*{INTEGER}\s*ops\s*\({INTEGER}\s*macc;\s*{INTEGER}\s*comp;\s*{INTEGER}\s*add;\s*{INTEGER}\s*mul;\s*{INTEGER}\s*bitwise\)",
        text,
        flags=re.IGNORECASE,
    ):
        layers.append(
            {
                "layer": int(match.group(1)),
                "ops": int(match.group(2).replace(",", "")),
                "macc": int(match.group(3).replace(",", "")),
                "comp": int(match.group(4).replace(",", "")),
                "add": int(match.group(5).replace(",", "")),
                "mul": int(match.group(6).replace(",", "")),
                "bitwise": int(match.group(7).replace(",", "")),
            }
        )
    return {
        "path": str(path),
        "exists": True,
        "sha256": sha256(path),
        "hardware_ops": (
            {
                "ops": int(hardware.group(1).replace(",", "")),
                "macc": int(hardware.group(2).replace(",", "")),
                "comp": int(hardware.group(3).replace(",", "")),
                "add": int(hardware.group(4).replace(",", "")),
                "mul": int(hardware.group(5).replace(",", "")),
                "bitwise": int(hardware.group(6).replace(",", "")),
            }
            if hardware
            else None
        ),
        "software_ops": (
            {
                "ops": int(software.group(1).replace(",", "")),
                "macc": int(software.group(2).replace(",", "")),
                "comp": int(software.group(3).replace(",", "")),
            }
            if software
            else None
        ),
        "weight_memory": (
            {
                "used_bytes": int(weight.group(1).replace(",", "")),
                "total_bytes": int(weight.group(2).replace(",", "")),
                "used_percent": float(weight.group(3)),
            }
            if weight
            else None
        ),
        "bias_memory": (
            {
                "used_bytes": int(bias.group(1).replace(",", "")),
                "total_bytes": int(bias.group(2).replace(",", "")),
                "used_percent": float(bias.group(3)),
            }
            if bias
            else None
        ),
        "latency_cycles": int_match(rf"(?:latency|cnn cycles)[^0-9]*{INTEGER}\s*(?:cycles)?", text),
        "energy_uj": float_match(r"energy[^0-9]*([0-9.]+)\s*u?j", text),
        "layers": layers,
    }


def generated_project_sizes(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    files = [file for file in path.rglob("*") if file.is_file()]
    by_suffix: dict[str, int] = {}
    for file in files:
        suffix = file.suffix or "<none>"
        by_suffix[suffix] = by_suffix.get(suffix, 0) + file.stat().st_size
    selected = {}
    for name in ("cnn.c", "cnn.h", "weights.h", "main.c", "log.txt"):
        candidate = next((file for file in files if file.name == name), None)
        if candidate is not None:
            selected[name] = {
                "path": str(candidate),
                "bytes": candidate.stat().st_size,
                "sha256": sha256(candidate),
            }
    return {
        "path": str(path),
        "exists": True,
        "file_count": len(files),
        "total_bytes": sum(file.stat().st_size for file in files),
        "bytes_by_suffix": by_suffix,
        "selected_files": selected,
    }


def parse_serial_jsonl(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    timings: dict[str, list[float]] = {}
    records = 0
    predictions: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records += 1
            row = json.loads(line)
            timing = row.get("timing", {})
            for key, value in timing.items():
                if isinstance(value, int | float):
                    timings.setdefault(key, []).append(float(value))
            primary = row.get("primary_detection") or row.get("prediction")
            if primary is None and row.get("detections"):
                primary = row["detections"][0]
            if isinstance(primary, dict):
                label = str(primary.get("name") or primary.get("id") or primary.get("class_id"))
                predictions[label] = predictions.get(label, 0) + 1
    timing_summary = {
        key: {
            "count": len(values),
            "min": min(values),
            "mean": sum(values) / len(values),
            "max": max(values),
        }
        for key, values in timings.items()
        if values
    }
    return {
        "path": str(path),
        "exists": True,
        "sha256": sha256(path),
        "records": records,
        "timing": timing_summary,
        "prediction_counts": predictions,
    }


def main() -> None:
    args = parse_args()
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
        "synthesis_log": parse_synthesis_log(args.synthesis_log),
        "generated_project": generated_project_sizes(args.generated_project),
        "serial_preview": parse_serial_jsonl(args.serial_jsonl),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
