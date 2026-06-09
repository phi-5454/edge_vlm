#!/usr/bin/env python3
"""Capture Coral Micro serial detection output as raw log plus JSONL."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


PREFIXES = ("VLM_MICRO_DETECTION ", "VLM_MICRO_READY ", "VLM_MICRO_ERROR ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="Serial device, for example /dev/ttyACM0.")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--raw-log", type=Path, default=Path("artifacts/profiles/coral/serial.log"))
    parser.add_argument(
        "--jsonl", type=Path, default=Path("artifacts/profiles/coral/detections.jsonl")
    )
    parser.add_argument(
        "--summary", type=Path, default=Path("artifacts/profiles/coral/report.json")
    )
    return parser.parse_args()


def prefixed_payload(line: str) -> tuple[str, dict[str, Any]] | None:
    for prefix in PREFIXES:
        if line.startswith(prefix):
            event = prefix.strip().removeprefix("VLM_MICRO_").lower()
            return event, json.loads(line[len(prefix) :])
    return None


def main() -> None:
    args = parse_args()
    try:
        import serial
    except ImportError as exc:
        raise SystemExit(
            "pyserial is required. Install with `uv sync --extra coral` or "
            "`uv pip install pyserial`."
        ) from exc

    args.raw_log.parent.mkdir(parents=True, exist_ok=True)
    args.jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(timezone.utc)
    detections = 0
    errors = 0
    ready: dict[str, Any] | None = None
    invoke_us: list[float] = []
    capture_us: list[float] = []

    with (
        serial.Serial(args.port, args.baud, timeout=args.timeout_s) as ser,
        args.raw_log.open("a", encoding="utf-8") as raw,
        args.jsonl.open("a", encoding="utf-8") as jsonl,
    ):
        while detections < args.frames:
            raw_bytes = ser.readline()
            if not raw_bytes:
                raise TimeoutError(f"Timed out waiting for serial output on {args.port}.")
            line = raw_bytes.decode("utf-8", errors="replace").strip()
            raw.write(line + "\n")
            raw.flush()
            parsed = prefixed_payload(line)
            if parsed is None:
                print(line)
                continue

            event, payload = parsed
            record = {
                "captured_at_utc": datetime.now(timezone.utc).isoformat(),
                "event": event,
                **payload,
            }
            jsonl.write(json.dumps(record, sort_keys=True) + "\n")
            jsonl.flush()
            print(json.dumps(record, sort_keys=True))

            if event == "ready":
                ready = payload
            elif event == "error":
                errors += 1
            elif event == "detection":
                detections += 1
                invoke_us.append(float(payload["invoke_us"]))
                capture_us.append(float(payload["capture_us"]))

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "started_at_utc": started_at.isoformat(),
        "port": args.port,
        "baud": args.baud,
        "frames_requested": args.frames,
        "frames_captured": detections,
        "errors": errors,
        "ready": ready,
        "raw_log": str(args.raw_log),
        "jsonl": str(args.jsonl),
        "invoke_us_mean": sum(invoke_us) / len(invoke_us) if invoke_us else None,
        "invoke_us_min": min(invoke_us) if invoke_us else None,
        "invoke_us_max": max(invoke_us) if invoke_us else None,
        "capture_us_mean": sum(capture_us) / len(capture_us) if capture_us else None,
        "capture_us_min": min(capture_us) if capture_us else None,
        "capture_us_max": max(capture_us) if capture_us else None,
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.summary}", file=sys.stderr)


if __name__ == "__main__":
    main()
