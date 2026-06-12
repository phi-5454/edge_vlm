#!/usr/bin/env python3
"""Capture one Coral Micro on-board self-test latency run."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import serial
from serial.tools import list_ports


PREFIX_READY = "VLM_MICRO_READY "
PREFIX_RESULT = "VLM_MICRO_SELFTEST_RESULT "
PREFIX_SUMMARY = "VLM_MICRO_SELFTEST_SUMMARY "
PREFIX_BEGIN = "VLM_MICRO_SELFTEST_BEGIN "
PREFIX_ERROR = "VLM_MICRO_ERROR "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--port",
        default="auto",
        help="Serial device, or 'auto' to use the Coral Micro VID:PID 18D1:9308 port.",
    )
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-log", type=Path, required=True)
    parser.add_argument(
        "--min-measured-iterations",
        type=int,
        default=100,
        help="Require at least this many non-warmup result records.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_prefixed(line: str, prefix: str) -> dict[str, Any] | None:
    if not line.startswith(prefix):
        return None
    return json.loads(line[len(prefix) :])


def serial_ports_text() -> list[str]:
    rows = []
    for port in list_ports.comports():
        rows.append(
            f"{port.device} {port.description or 'n/a'} "
            f"VID:PID={port.vid:04X}:{port.pid:04X}" if port.vid and port.pid
            else f"{port.device} {port.description or 'n/a'}"
        )
    return rows


def resolve_port(port: str) -> str:
    if port != "auto":
        return port
    candidates = [
        item
        for item in list_ports.comports()
        if item.vid == 0x18D1 and item.pid in {0x9308, 0x93FF}
    ]
    if not candidates:
        raise RuntimeError(
            "Could not find a Coral Micro serial port. "
            f"Available ports: {serial_ports_text()}"
        )
    if len(candidates) > 1:
        devices = ", ".join(item.device for item in candidates)
        raise RuntimeError(
            "Multiple Coral Micro serial ports found; pass --port explicitly. "
            f"Candidates: {devices}"
        )
    return str(candidates[0].device)


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [record for record in results if not bool(record.get("warmup", False))]
    invoke_values = [float(record["invoke_us"]) for record in measured]
    copy_values = [float(record.get("copy_us", 0.0)) for record in measured]
    return {
        "measured_iterations": len(measured),
        "invoke_us": {
            "min": min(invoke_values) if invoke_values else None,
            "mean": statistics.fmean(invoke_values) if invoke_values else None,
            "median": statistics.median(invoke_values) if invoke_values else None,
            "max": max(invoke_values) if invoke_values else None,
        },
        "copy_us": {
            "min": min(copy_values) if copy_values else None,
            "mean": statistics.fmean(copy_values) if copy_values else None,
            "median": statistics.median(copy_values) if copy_values else None,
            "max": max(copy_values) if copy_values else None,
        },
    }


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"{args.output} exists; rerun with --force.")
    if args.raw_log.exists() and not args.force:
        raise FileExistsError(f"{args.raw_log} exists; rerun with --force.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.raw_log.parent.mkdir(parents=True, exist_ok=True)
    args.port = resolve_port(str(args.port))

    started_at = time.time()
    ready: dict[str, Any] | None = None
    begin: dict[str, Any] | None = None
    summary: dict[str, Any] | None = None
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    recent_lines: list[str] = []

    try:
        with serial.Serial(args.port, args.baud, timeout=1.0) as ser:
            with args.raw_log.open("w", encoding="utf-8") as raw:
                while time.time() - started_at < args.timeout_s:
                    raw_bytes = ser.readline()
                    if not raw_bytes:
                        continue
                    line = raw_bytes.decode("utf-8", errors="replace").strip()
                    raw.write(line + "\n")
                    raw.flush()
                    recent_lines.append(line)
                    recent_lines = recent_lines[-20:]

                    if (payload := parse_prefixed(line, PREFIX_READY)) is not None:
                        ready = payload
                        print(json.dumps({"event": "ready", **payload}))
                        continue
                    if (payload := parse_prefixed(line, PREFIX_BEGIN)) is not None:
                        begin = payload
                        results = []
                        print(json.dumps({"event": "selftest_begin", **payload}))
                        continue
                    if (payload := parse_prefixed(line, PREFIX_RESULT)) is not None:
                        results.append(payload)
                        if not payload.get("warmup", False):
                            measured_seen = sum(
                                1 for record in results if not record.get("warmup", False)
                            )
                            print(
                                json.dumps(
                                    {
                                        "event": "selftest_progress",
                                        "measured_seen": measured_seen,
                                        "invoke_us": payload.get("invoke_us"),
                                    }
                                )
                            )
                        continue
                    if (payload := parse_prefixed(line, PREFIX_SUMMARY)) is not None:
                        summary = payload
                        measured_seen = sum(
                            1 for record in results if not record.get("warmup", False)
                        )
                        if measured_seen >= args.min_measured_iterations:
                            break
                        print(
                            json.dumps(
                                {
                                    "event": "summary_ignored",
                                    "measured_seen": measured_seen,
                                    "required": args.min_measured_iterations,
                                }
                            )
                        )
                        continue
                    if (payload := parse_prefixed(line, PREFIX_ERROR)) is not None:
                        errors.append(payload)
                        print(json.dumps({"event": "board_error", **payload}))
    except serial.SerialException as exc:
        raise RuntimeError(
            f"Serial failure on {args.port}: {exc}\n"
            f"Available ports: {serial_ports_text()}"
        ) from exc

    measured_seen = sum(1 for record in results if not record.get("warmup", False))
    if summary is None or measured_seen < args.min_measured_iterations:
        raise TimeoutError(
            f"Timed out after {args.timeout_s}s waiting for "
            f"{args.min_measured_iterations} measured self-test iterations on {args.port}.\n"
            f"Raw log: {args.raw_log}\n"
            f"Available ports: {serial_ports_text()}\n"
            f"Recent lines: {recent_lines}"
        )

    report = {
        "created_at_unix": time.time(),
        "model_name": args.model_name,
        "port": args.port,
        "baud": args.baud,
        "ready": ready,
        "begin": begin,
        "board_summary": summary,
        "host_summary": summarize_results(results),
        "errors": errors,
        "results": results,
        "raw_log": str(args.raw_log),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "wrote_report", "output": str(args.output)}))


if __name__ == "__main__":
    main()
