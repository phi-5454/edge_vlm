#!/usr/bin/env python3
"""Browser preview for MAX78000 serial camera/model output."""

from __future__ import annotations

import argparse
import base64
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import threading
import time
from typing import Any
from urllib.parse import parse_qs, urlparse

from PIL import Image, ImageDraw, ImageEnhance, ImageFont


SERIAL_PREFIXES = (
    "VLM_MAX78000_PREVIEW",
    "VLM_MAX78000_FRAME",
    "VLM_MICRO_MAX78000_PREVIEW",
    "VLM_MICRO_MAX78000_FRAME",
)

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MAX78000 Preview</title>
  <style>
    body { margin: 0; font-family: system-ui, sans-serif; background: #15171a; color: #f5f7fa; }
    main { max-width: 980px; margin: 0 auto; padding: 20px; }
    header { display: flex; justify-content: space-between; gap: 16px; align-items: baseline; }
    img { width: min(100%, 900px); height: auto; background: #050607; }
    .panel { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; margin: 14px 0; }
    .metric { background: #23272d; border: 1px solid #343a43; padding: 10px 12px; border-radius: 6px; min-width: 0; }
    .label { color: #aab2bf; font-size: 12px; }
    .value { font-size: 18px; margin-top: 4px; overflow-wrap: anywhere; }
    button { background: #2f7dd1; color: white; border: 0; border-radius: 6px; padding: 8px 12px; cursor: pointer; }
    button.paused { background: #686f7a; }
    pre { white-space: pre-wrap; background: #0d0f12; padding: 12px; border-radius: 6px; overflow: auto; }
  </style>
</head>
<body>
<main>
  <header>
    <h1>MAX78000 Preview</h1>
    <button id="toggle">Pause</button>
  </header>
  <div class="panel">
    <div class="metric"><div class="label">Frame</div><div class="value" id="frame">-</div></div>
    <div class="metric"><div class="label">Prediction</div><div class="value" id="prediction">-</div></div>
    <div class="metric"><div class="label">Score</div><div class="value" id="score">-</div></div>
    <div class="metric"><div class="label">Inference us</div><div class="value" id="invoke_us">-</div></div>
    <div class="metric"><div class="label">Serial age ms</div><div class="value" id="age_ms">-</div></div>
  </div>
  <img id="preview" alt="MAX78000 camera preview">
  <pre id="json"></pre>
</main>
<script>
let paused = false;
let frame = 0;
const intervalMs = Number(new URLSearchParams(location.search).get("interval_ms") || "500");
const toggle = document.getElementById("toggle");
toggle.onclick = () => {
  paused = !paused;
  toggle.textContent = paused ? "Resume" : "Pause";
  toggle.className = paused ? "paused" : "";
};
async function tick() {
  if (!paused) {
    const response = await fetch(`/frame.jpg?frame=${frame++}`, { cache: "no-store" });
    const meta = JSON.parse(response.headers.get("X-Preview-Metadata") || "{}");
    document.getElementById("preview").src = URL.createObjectURL(await response.blob());
    document.getElementById("frame").textContent = meta.frame_id ?? "-";
    const primary = meta.primary_detection || meta.prediction || null;
    document.getElementById("prediction").textContent = primary ? (primary.name || primary.id || primary.class_id || primary.label || "-") : "-";
    document.getElementById("score").textContent = primary && primary.score != null ? Number(primary.score).toFixed(3) : "-";
    document.getElementById("invoke_us").textContent = meta.timing && meta.timing.inference_us != null ? meta.timing.inference_us : "-";
    document.getElementById("age_ms").textContent = meta.host_age_ms != null ? meta.host_age_ms.toFixed(0) : "-";
    document.getElementById("json").textContent = JSON.stringify(meta, null, 2);
  }
  setTimeout(tick, intervalMs);
}
tick();
</script>
</body>
</html>
"""


@dataclass
class PreviewState:
    jsonl: Path
    jpeg_quality: int
    display_brightness_gain: float
    labels: dict[int, str]
    label_offset: int
    show_label_names: bool
    raw_log: Path | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    latest: dict[str, Any] | None = None
    history: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=8))

    def __post_init__(self) -> None:
        self.jsonl.parent.mkdir(parents=True, exist_ok=True)
        if self.raw_log is not None:
            self.raw_log.parent.mkdir(parents=True, exist_ok=True)

    def update(self, record: dict[str, Any]) -> None:
        record = normalize_record(record, self.labels, self.label_offset, self.show_label_names)
        record["received_at_utc"] = datetime.now(timezone.utc).isoformat()
        with self.lock:
            self.latest = record
            self.history.append(record)
        with self.jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", help="Serial device, for example /dev/ttyACM0.")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--timeout-s", type=float, default=0.25)
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=8775)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--display-brightness-gain", type=float, default=1.0)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--label-offset", type=int, default=0)
    parser.add_argument("--show-label-names", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--jsonl", type=Path, default=Path("artifacts/profiles/max78000/preview.jsonl"))
    parser.add_argument("--raw-log", type=Path, default=Path("artifacts/profiles/max78000/preview_serial.log"))
    parser.add_argument(
        "--replay-jsonl",
        type=Path,
        default=None,
        help="Replay a captured preview JSONL instead of opening a serial port.",
    )
    parser.add_argument("--replay-interval-s", type=float, default=0.5)
    return parser.parse_args()


def load_labels(path: Path | None) -> dict[int, str]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(class_id): str(name) for class_id, name in payload.items()}


def prefixed_json(line: str) -> dict[str, Any] | None:
    stripped = line.strip()
    for prefix in SERIAL_PREFIXES:
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix) :].strip()
            break
    if not stripped.startswith("{"):
        return None
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def serial_reader(state: PreviewState, port: str, baud: int, timeout_s: float) -> None:
    try:
        import serial
    except ImportError as error:
        raise SystemExit("pyserial is required. Install with `uv sync --extra max78000`.") from error

    with serial.Serial(port, baud, timeout=timeout_s) as ser:
        while True:
            raw = ser.readline()
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace").strip()
            if state.raw_log is not None:
                with state.raw_log.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
            record = prefixed_json(line)
            if record is not None:
                state.update(record)


def replay_reader(state: PreviewState, path: Path, interval_s: float) -> None:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise SystemExit(f"{path} does not contain preview records.")
    while True:
        for row in rows:
            state.update(row)
            time.sleep(interval_s)


def decode_image(record: dict[str, Any]) -> Image.Image:
    image = record.get("image", record)
    width = int(image["width"])
    height = int(image["height"])
    encoding = str(image.get("encoding", image.get("format", "rgb565le"))).lower()
    data_b64 = image.get("data_b64", image.get("base64_data"))
    if not data_b64:
        raise ValueError("preview record does not contain image.data_b64/base64_data")
    payload = base64.b64decode(str(data_b64))
    if encoding in {"jpeg", "jpg"}:
        return Image.open(BytesIO(payload)).convert("RGB")
    if encoding in {"rgb888", "rgb24"}:
        return Image.frombytes("RGB", (width, height), payload)
    if encoding in {"gray8", "l"}:
        return Image.frombytes("L", (width, height), payload).convert("RGB")
    if encoding in {"rgb565le", "rgb565"}:
        return Image.frombytes("RGB", (width, height), payload, "raw", "BGR;16")
    if encoding == "rgb565be":
        return Image.frombytes("RGB", (width, height), payload, "raw", "BGR;16B")
    raise ValueError(f"Unsupported image encoding: {encoding}")


def normalize_record(
    record: dict[str, Any],
    labels: dict[int, str],
    label_offset: int,
    show_label_names: bool,
) -> dict[str, Any]:
    normalized = dict(record)
    detections = list(normalized.get("detections", []))
    if "detection" in normalized and normalized["detection"] is not None:
        detections = [normalized["detection"], *detections]
    for detection in detections:
        raw_id = detection.get("raw_id", detection.get("id", detection.get("class_id")))
        if raw_id is None:
            continue
        raw_id = int(raw_id)
        label_id = raw_id + label_offset
        detection["raw_id"] = raw_id
        detection["label_id"] = label_id
        detection["name"] = labels.get(label_id) if show_label_names else None
    if detections:
        normalized["detections"] = detections
        normalized["primary_detection"] = detections[0]
    return normalized


def draw_preview(
    record: dict[str, Any],
    jpeg_quality: int,
    display_brightness_gain: float,
) -> tuple[bytes, dict[str, Any]]:
    if display_brightness_gain <= 0:
        raise ValueError("--display-brightness-gain must be positive.")
    image = decode_image(record)
    if display_brightness_gain != 1.0:
        image = ImageEnhance.Brightness(image).enhance(display_brightness_gain)
    width, height = image.size
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for detection in record.get("detections", []):
        box = detection.get("box", detection)
        if not all(key in box for key in ("xmin", "ymin", "xmax", "ymax")):
            continue
        left = float(box["xmin"])
        top = float(box["ymin"])
        right = float(box["xmax"])
        bottom = float(box["ymax"])
        if max(left, top, right, bottom) <= 1.5:
            left *= width
            right *= width
            top *= height
            bottom *= height
        draw.rectangle([left, top, right, bottom], outline=(255, 64, 64), width=3)
        name = detection.get("name")
        label = (
            f'{name} raw={detection.get("raw_id")} score={float(detection.get("score", 0.0)):.3f}'
            if name
            else f'raw_id={detection.get("raw_id", detection.get("id", "-"))} score={float(detection.get("score", 0.0)):.3f}'
        )
        text_box = draw.textbbox((left, top), label, font=font)
        draw.rectangle(text_box, fill=(255, 64, 64))
        draw.text((left, top), label, fill=(255, 255, 255), font=font)

    metadata = {key: value for key, value in record.items() if key != "image"}
    received_at = metadata.get("received_at_utc")
    if received_at:
        received = datetime.fromisoformat(str(received_at))
        metadata["host_age_ms"] = (datetime.now(timezone.utc) - received).total_seconds() * 1000
    metadata["width"] = width
    metadata["height"] = height
    metadata["display_brightness_gain"] = display_brightness_gain
    output = BytesIO()
    image.save(output, format="JPEG", quality=jpeg_quality)
    return output.getvalue(), metadata


def placeholder(jpeg_quality: int) -> tuple[bytes, dict[str, Any]]:
    image = Image.new("RGB", (320, 240), (10, 12, 15))
    draw = ImageDraw.Draw(image)
    draw.text((18, 104), "Waiting for MAX78000 preview serial records", fill=(230, 235, 242))
    output = BytesIO()
    image.save(output, format="JPEG", quality=jpeg_quality)
    return output.getvalue(), {"status": "waiting_for_serial"}


class Handler(BaseHTTPRequestHandler):
    state: PreviewState

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(INDEX_HTML.encode("utf-8"))
            return
        if parsed.path != "/frame.jpg":
            self.send_error(404)
            return
        _ = parse_qs(parsed.query)
        with self.state.lock:
            record = dict(self.state.latest) if self.state.latest is not None else None
        try:
            if record is None:
                image, metadata = placeholder(self.state.jpeg_quality)
            else:
                image, metadata = draw_preview(
                    record,
                    self.state.jpeg_quality,
                    self.state.display_brightness_gain,
                )
        except Exception as exc:
            self.send_error(502, explain=str(exc))
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Preview-Metadata", json.dumps(metadata))
        self.end_headers()
        self.wfile.write(image)

    def log_message(self, fmt: str, *args: Any) -> None:
        return


def main() -> None:
    args = parse_args()
    if args.replay_jsonl is None and not args.port:
        raise SystemExit("Provide --port or --replay-jsonl.")
    state = PreviewState(
        jsonl=args.jsonl,
        raw_log=args.raw_log if args.replay_jsonl is None else None,
        jpeg_quality=args.jpeg_quality,
        display_brightness_gain=args.display_brightness_gain,
        labels=load_labels(args.labels),
        label_offset=args.label_offset,
        show_label_names=args.show_label_names,
    )
    if args.replay_jsonl is not None:
        thread = threading.Thread(
            target=replay_reader,
            args=(state, args.replay_jsonl, args.replay_interval_s),
            daemon=True,
        )
    else:
        thread = threading.Thread(
            target=serial_reader,
            args=(state, args.port, args.baud, args.timeout_s),
            daemon=True,
        )
    thread.start()

    class BoundHandler(Handler):
        pass

    BoundHandler.state = state
    server = ThreadingHTTPServer((args.listen_host, args.listen_port), BoundHandler)
    print(f"Preview: http://{args.listen_host}:{args.listen_port}")
    print(f"JSONL log: {args.jsonl}")
    if args.replay_jsonl is not None:
        print(f"Replay JSONL: {args.replay_jsonl}")
    else:
        print(f"Serial: {args.port} @ {args.baud}")
        print(f"Raw serial log: {args.raw_log}")
    server.serve_forever()


if __name__ == "__main__":
    main()
