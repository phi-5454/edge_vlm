#!/usr/bin/env python3
"""Browser preview for Coral Micro camera detection RPC output."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import threading
from typing import Any
from urllib.parse import parse_qs, urlparse

from PIL import Image, ImageDraw, ImageEnhance, ImageFont
import requests


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Coral Detection Preview</title>
  <style>
    body { margin: 0; font-family: system-ui, sans-serif; background: #15171a; color: #f5f7fa; }
    main { max-width: 980px; margin: 0 auto; padding: 20px; }
    header { display: flex; justify-content: space-between; gap: 16px; align-items: baseline; }
    img { width: min(100%, 900px); height: auto; image-rendering: auto; background: #050607; }
    .panel { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 14px 0; }
    .metric { background: #23272d; border: 1px solid #343a43; padding: 10px 12px; border-radius: 6px; }
    .label { color: #aab2bf; font-size: 12px; }
    .value { font-size: 18px; margin-top: 4px; }
    button { background: #2f7dd1; color: white; border: 0; border-radius: 6px; padding: 8px 12px; cursor: pointer; }
    button.paused { background: #686f7a; }
    pre { white-space: pre-wrap; background: #0d0f12; padding: 12px; border-radius: 6px; overflow: auto; }
  </style>
</head>
<body>
<main>
  <header>
    <h1>Coral Detection Preview</h1>
    <button id="toggle">Pause</button>
  </header>
  <div class="panel">
    <div class="metric"><div class="label">Frame</div><div class="value" id="frame">-</div></div>
    <div class="metric"><div class="label">Score</div><div class="value" id="score">-</div></div>
    <div class="metric"><div class="label">Class ID</div><div class="value" id="class_id">-</div></div>
    <div class="metric"><div class="label">RPC ms</div><div class="value" id="rpc_ms">-</div></div>
  </div>
  <img id="preview" alt="Coral camera detection preview">
  <pre id="json"></pre>
</main>
<script>
let paused = false;
let frame = 0;
const intervalMs = Number(new URLSearchParams(location.search).get("interval_ms") || "1000");
const toggle = document.getElementById("toggle");
toggle.onclick = () => {
  paused = !paused;
  toggle.textContent = paused ? "Resume" : "Pause";
  toggle.className = paused ? "paused" : "";
};
async function tick() {
  if (!paused) {
    const response = await fetch(`/frame.jpg?frame=${frame++}`, { cache: "no-store" });
    const meta = JSON.parse(response.headers.get("X-Detection-Metadata") || "{}");
    document.getElementById("preview").src = URL.createObjectURL(await response.blob());
    document.getElementById("frame").textContent = meta.frame_id ?? "-";
    document.getElementById("score").textContent = meta.detection ? meta.detection.score.toFixed(3) : "none";
    document.getElementById("class_id").textContent = meta.detection ? meta.detection.id : "none";
    document.getElementById("rpc_ms").textContent = meta.rpc_ms ? meta.rpc_ms.toFixed(1) : "-";
    document.getElementById("json").textContent = JSON.stringify(meta, null, 2);
  }
  setTimeout(tick, intervalMs);
}
tick();
</script>
</body>
</html>
"""


class PreviewState:
    def __init__(
        self,
        coral_host: str,
        jsonl: Path,
        timeout_s: float,
        jpeg_quality: int,
        display_brightness_gain: float,
        labels: dict[int, str],
        label_offset: int,
        show_label_names: bool,
    ):
        self.coral_host = coral_host
        self.jsonl = jsonl
        self.timeout_s = timeout_s
        self.jpeg_quality = jpeg_quality
        self.display_brightness_gain = display_brightness_gain
        self.labels = labels
        self.label_offset = label_offset
        self.show_label_names = show_label_names
        self.lock = threading.Lock()
        self.frame_id = 0
        self.jsonl.parent.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coral-host", default="10.10.10.1")
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=8765)
    parser.add_argument("--timeout-s", type=float, default=10.0)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument(
        "--display-brightness-gain",
        type=float,
        default=1.0,
        help="Preview-only brightness multiplier. Does not affect board inference.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("conf/labels/coco_detection_labels.json"),
        help="JSON object mapping detection class id strings to names.",
    )
    parser.add_argument(
        "--label-offset",
        type=int,
        default=0,
        help="Add this offset to raw model ids before label lookup.",
    )
    parser.add_argument(
        "--show-label-names",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Display label names from --labels. Raw ids are always logged.",
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=Path("artifacts/profiles/coral/preview_detections.jsonl"),
    )
    return parser.parse_args()


def load_labels(path: Path) -> dict[int, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(class_id): str(name) for class_id, name in payload.items()}


def rpc_detect(coral_host: str, timeout_s: float) -> tuple[dict[str, Any], float]:
    started = datetime.now(timezone.utc)
    response = requests.post(
        f"http://{coral_host}:80/jsonrpc",
        json={"method": "detect_from_camera", "jsonrpc": "2.0", "id": 0},
        timeout=timeout_s,
    )
    response.raise_for_status()
    elapsed_ms = (datetime.now(timezone.utc) - started).total_seconds() * 1000
    payload = response.json()
    if "error" in payload:
        raise RuntimeError(payload["error"])
    return payload["result"], elapsed_ms


def draw_detection(
    result: dict[str, Any],
    frame_id: int,
    rpc_ms: float,
    labels: dict[int, str],
    label_offset: int,
    show_label_names: bool,
    display_brightness_gain: float,
    jpeg_quality: int,
) -> tuple[bytes, dict[str, Any]]:
    width = int(result["width"])
    height = int(result["height"])
    image = Image.frombytes("RGB", (width, height), base64.b64decode(result["base64_data"]))
    if display_brightness_gain <= 0:
        raise ValueError("--display-brightness-gain must be positive.")
    if display_brightness_gain != 1.0:
        image = ImageEnhance.Brightness(image).enhance(display_brightness_gain)
    draw = ImageDraw.Draw(image)
    detection = result.get("detection")
    if detection:
        raw_class_id = int(detection["id"])
        label_id = raw_class_id + label_offset
        detection["raw_id"] = raw_class_id
        detection["label_id"] = label_id
        detection["name"] = labels.get(label_id)
        left = float(detection["xmin"]) * width
        top = float(detection["ymin"]) * height
        right = float(detection["xmax"]) * width
        bottom = float(detection["ymax"]) * height
        draw.rectangle([left, top, right, bottom], outline=(255, 64, 64), width=3)
        label_name = detection["name"] if show_label_names and detection["name"] else None
        label = (
            f'{label_name} raw={raw_class_id} score={float(detection["score"]):.3f}'
            if label_name
            else f'raw_id={raw_class_id} label_id={label_id} score={float(detection["score"]):.3f}'
        )
        try:
            font = ImageFont.load_default()
        except OSError:
            font = None
        bbox = draw.textbbox((left, top), label, font=font)
        draw.rectangle(bbox, fill=(255, 64, 64))
        draw.text((left, top), label, fill=(255, 255, 255), font=font)

    metadata = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "frame_id": frame_id,
        "width": width,
        "height": height,
        "rpc_ms": rpc_ms,
        "display_brightness_gain": display_brightness_gain,
        "label_offset": label_offset,
        "show_label_names": show_label_names,
        "detection": detection,
    }
    output = BytesIO()
    image.save(output, format="JPEG", quality=jpeg_quality)
    return output.getvalue(), metadata


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
            frame_id = self.state.frame_id
            self.state.frame_id += 1
        try:
            result, rpc_ms = rpc_detect(self.state.coral_host, self.state.timeout_s)
            image, metadata = draw_detection(
                result,
                frame_id,
                rpc_ms,
                self.state.labels,
                self.state.label_offset,
                self.state.show_label_names,
                self.state.display_brightness_gain,
                self.state.jpeg_quality,
            )
            with self.state.jsonl.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metadata, sort_keys=True) + "\n")
        except Exception as exc:
            self.send_error(502, explain=str(exc))
            return

        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Detection-Metadata", json.dumps(metadata))
        self.end_headers()
        self.wfile.write(image)

    def log_message(self, fmt: str, *args: Any) -> None:
        return


def main() -> None:
    args = parse_args()
    labels = load_labels(args.labels)
    state = PreviewState(
        args.coral_host,
        args.jsonl,
        args.timeout_s,
        args.jpeg_quality,
        args.display_brightness_gain,
        labels,
        args.label_offset,
        args.show_label_names,
    )

    class BoundHandler(Handler):
        pass

    BoundHandler.state = state
    server = ThreadingHTTPServer((args.listen_host, args.listen_port), BoundHandler)
    print(f"Preview: http://{args.listen_host}:{args.listen_port}")
    print(f"Coral RPC host: {args.coral_host}")
    print(f"Labels: {args.labels}")
    print(f"Label offset: {args.label_offset}")
    print(f"Show label names: {args.show_label_names}")
    print(f"Display brightness gain: {args.display_brightness_gain}")
    print(f"JSONL log: {args.jsonl}")
    server.serve_forever()


if __name__ == "__main__":
    main()
