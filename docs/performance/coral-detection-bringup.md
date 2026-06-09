# Coral Detection Bringup Runbook

This runbook documents the generic TFLite-onward test path for Coral Dev Board
Micro. It starts from an already compiled Edge TPU `.tflite` model, runs live
camera inference on the board, and captures normalized detection output on the
host over serial.

`artifacts/models/tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite`

The current app is intentionally independent from the VLM training code. Its job
is to validate the board, camera, Edge TPU runtime, tensor arena, serial
transport, and report generation before we deploy our own models.

## Files

| Path | Purpose |
| --- | --- |
| `artifacts/models/tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite` | Edge TPU object detection model. |
| `coral_micro/detect_objects_serial/` | Repo-owned Coral Micro app source. |
| `scripts/coral_micro_stage_detection_app.py` | Copies the app and model into the adjacent Coral SDK. |
| `scripts/capture_coral_detection_serial.py` | Captures serial output into raw and normalized artifacts. |
| `scripts/coral_detection_preview_server.py` | Browser preview for camera frames with detection boxes. |
| `conf/labels/coco_detection_labels.json` | COCO detection id-to-name mapping for this SSD model family. |
| `scripts/inspect_tflite.py` | Writes input/output tensor metadata and model hash. |
| `docs/decisions/0006-coral-tflite-detection-bringup.md` | Decision record for this bring-up path. |

## Prerequisites

- The Coral SDK checkout exists at `../coralmicro`.
- The board is connected over USB and can be flashed with the Coral SDK
  `scripts/flashtool.py`.
- Python dependencies are installed with the Coral extra:

```bash
uv sync --extra coral --extra dev
```

If the serial capture command cannot import `serial`, make sure `pyserial` is
installed through the `coral` extra.

## Step 1: Inspect The TFLite Artifact

```bash
uv run --extra coral python scripts/inspect_tflite.py \
  artifacts/models/tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite \
  --output artifacts/reports/coral_detection_tflite_inspection.json
```

Expected output:

- `artifacts/reports/coral_detection_tflite_inspection.json`
- model byte size and SHA256
- TFLite input tensor shape, dtype, and quantization
- TFLite output tensor shapes, dtypes, and quantization

Keep this report with board results so later profiling can be tied to the exact
model artifact.

## Step 2: Stage The App Into The Coral SDK

```bash
python scripts/coral_micro_stage_detection_app.py --force
```

This copies `coral_micro/detect_objects_serial/` into
`../coralmicro/examples/vlm_micro_detect_objects_serial/`, copies the model into
`../coralmicro/models/`, and appends the example to the Coral SDK examples
CMake file.

Use `--force` when intentionally refreshing the staged SDK copy from the
repo-owned source. The source of truth remains this repository.

## Step 3: Build And Flash

From `../coralmicro`:

```bash
bash build.sh
uv run --with-requirements scripts/requirements.txt \
  python scripts/flashtool.py -e vlm_micro_detect_objects_serial
```

`flashtool.py` imports packages such as `progress`, `hidapi`, `pyserial`, and
`pyusb` from `../coralmicro/scripts/requirements.txt`. Running it with plain
`python3 scripts/flashtool.py ...` will fail unless those packages are already
installed in that Python environment.

If `hidapi==0.10.1` fails to build under Python 3.12, create a Coral SDK venv
with Python 3.11 and install the old source package without build isolation:

```bash
cd ../coralmicro
uv venv --python /home/younes/.local/bin/python3.11 .venv
uv pip install --python .venv/bin/python "Cython<3" wheel setuptools
uv pip install --python .venv/bin/python --no-build-isolation \
  -r scripts/requirements.txt
.venv/bin/python scripts/flashtool.py -e vlm_micro_detect_objects_serial
```

On Arch Linux, if the `hidapi` build then fails with a missing HIDAPI header,
install the system library with `sudo pacman -S hidapi` and rerun the final
`uv pip install` command.

The flashed app:

1. Loads `/models/tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite` from littlefs.
2. Opens the Edge TPU.
3. Allocates an 8 MiB TFLM tensor arena in SDRAM.
4. Starts the onboard camera in trigger mode.
5. Discards 100 warm-up frames so auto exposure can calibrate.
6. Captures one RGB frame per second.
7. Runs inference.
8. Prints newline-delimited JSON records to the serial console.

## Step 4: Capture Serial Output

Replace the port with the board's serial device:

```bash
uv run --extra coral python scripts/capture_coral_detection_serial.py \
  --port /dev/ttyACM0 \
  --frames 100 \
  --raw-log artifacts/profiles/coral/serial.log \
  --jsonl artifacts/profiles/coral/detections.jsonl \
  --summary artifacts/profiles/coral/report.json
```

The app emits newline-delimited records prefixed with `VLM_MICRO_READY`,
`VLM_MICRO_DETECTION`, or `VLM_MICRO_ERROR`. The capture script preserves the
raw serial log and writes normalized JSONL plus a latency summary.

## Optional: Camera And Box Preview

The serial app logs detections but does not stream image bytes. For visual
preview, flash an RPC-capable detection app and run the browser preview server.
The upstream Coral SDK `detect_objects` example already exposes
`detect_from_camera`, returning one captured RGB frame plus the top detection:

```bash
cd ../coralmicro
.venv/bin/python scripts/flashtool.py -e detect_objects
```

Then from this repo:

```bash
uv run --extra coral python scripts/coral_detection_preview_server.py \
  --coral-host 10.10.10.1 \
  --listen-port 8765 \
  --labels conf/labels/coco_detection_labels.json \
  --no-show-label-names \
  --display-brightness-gain 1.5 \
  --jsonl artifacts/profiles/coral/preview_detections.jsonl
```

Open `http://127.0.0.1:8765` in a browser. Each refresh cycle triggers a board
camera capture plus inference, draws the returned detection box on the frame,
and appends normalized metadata to the JSONL log.

The preview logs raw model IDs by default. It can use
`conf/labels/coco_detection_labels.json` to map IDs to COCO item names, but keep
names disabled until the model's ID convention is verified against known
objects:

```bash
uv run --extra coral python scripts/coral_detection_preview_server.py \
  --show-label-names \
  --label-offset 0
```

If labels look systematically shifted, retry with `--label-offset 1` or
`--label-offset -1` and compare against obvious objects. COCO detection IDs are
not contiguous; for example, ID `1` is `person`, ID `3` is `car`, ID `18` is
`dog`, and ID `90` is `toothbrush`.

`--display-brightness-gain` is preview-only. It brightens the JPEG served to the
browser after inference has already happened on the board, so it should not be
used as evidence that model input exposure changed.

Current limitation: the upstream RPC endpoint returns only the top detection.
The serial app returns top-k detections, but no image. A later board app should
combine both: RPC image preview plus top-k JSON detections.

## Serial Record Format

Ready record:

```json
{
  "model_path": "/models/tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite",
  "width": 300,
  "height": 300,
  "tensor_arena_bytes": 8388608,
  "threshold": 0.5,
  "top_k": 10,
  "camera_warmup_discard_frames": 100
}
```

Detection record:

```json
{
  "frame_id": 0,
  "width": 300,
  "height": 300,
  "capture_us": 12345,
  "invoke_us": 67890,
  "threshold": 0.5,
  "top_k": 10,
  "detections": [
    {
      "id": 1,
      "score": 0.95,
      "bbox": {
        "ymin": 0.1,
        "xmin": 0.2,
        "ymax": 0.8,
        "xmax": 0.9
      }
    }
  ]
}
```

Host-side JSONL records add `captured_at_utc` and `event`.

## Output Artifacts

| Artifact | Meaning |
| --- | --- |
| `artifacts/profiles/coral/serial.log` | Raw serial console lines. Preserve this for debugging. |
| `artifacts/profiles/coral/detections.jsonl` | Parsed ready/error/detection events. |
| `artifacts/profiles/coral/report.json` | Capture summary with frame count and latency statistics. |

The report currently summarizes `capture_us` and `invoke_us`. It does not make
accuracy claims.

## Profiling Coverage

Current pipeline:

| Metric | Available now? | Source |
| --- | --- | --- |
| Model byte size | Yes | `scripts/inspect_tflite.py` report. |
| Model SHA256 | Yes | `scripts/inspect_tflite.py` report. |
| Input/output tensor metadata | Yes | `scripts/inspect_tflite.py` report. |
| Tensor arena allocation | Yes, configured size | `VLM_MICRO_READY` record from the serial app. |
| Camera capture latency | Yes | `capture_us` in `VLM_MICRO_DETECTION`. |
| Edge TPU/TFLM invoke latency | Yes | `invoke_us` in `VLM_MICRO_DETECTION`. |
| End-to-end preview RPC latency | Yes | `rpc_ms` from `scripts/coral_detection_preview_server.py`. |
| Detection output values | Yes | `detections.jsonl` or `preview_detections.jsonl`. |
| Edge TPU op mapping | Not from compiled model alone | Requires original uncompiled int8 `.tflite` plus `edgetpu_compiler` log. |
| MACs / arithmetic op count | Not reliable from compiled model alone | Requires original graph analysis or compiler/model-card metadata. |
| CPU fallback op count | Not from compiled model alone | Requires `edgetpu_compiler` log from compilation. |
| Board power / energy | Not yet | Requires external measurement hardware or a board power harness. |

Pure board latency:

```bash
uv run --extra coral python scripts/capture_coral_detection_serial.py \
  --port /dev/ttyACM0 \
  --frames 100 \
  --summary artifacts/profiles/coral/report.json
```

This measures camera capture and `interpreter.Invoke()` on the board. It does
not include browser rendering or host RPC overhead.

Preview latency:

```bash
uv run --extra coral python scripts/coral_detection_preview_server.py \
  --coral-host 10.10.10.1 \
  --jsonl artifacts/profiles/coral/preview_detections.jsonl
```

This records `rpc_ms`, which includes USB/Ethernet RPC, camera capture,
inference, response serialization, and host receipt. Use this for operator UX,
not for model-only latency.

For Edge TPU op mapping, keep the uncompiled full-int8 TFLite model and run the
compiler with logging:

```bash
edgetpu_compiler \
  -o artifacts/exports/coral \
  artifacts/exports/coral/model_int8.tflite \
  2>&1 | tee artifacts/profiles/coral/edgetpu_compiler.log
```

The already compiled artifact
`tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite` contains an Edge TPU custom op,
so the host-side TFLite inspection cannot recover the internal TPU subgraph,
MAC count, or CPU fallback breakdown.

For power, use an external measurement path and record the setup in the report:

- USB-C inline power meter for coarse board-level watts.
- Joulescope/Otii/Monsoon-style instrument for time-aligned current and energy.
- A board-specific shunt/INA-style harness if we add one later.

Always record voltage rail, sampling rate, whether the camera/Edge TPU are
active, number of frames, and the exact flashed app.

## Troubleshooting

If the app reports `model_load`, verify that staging copied the model into
`../coralmicro/models/` and that the app was rebuilt after staging.

If the app reports `edgetpu_open`, power-cycle the board and retry flashing. The
Edge TPU runtime must open before the interpreter is created.

If `allocate_tensors` fails, the tensor arena is too small or the model/runtime
combination is incompatible. Increase `kTensorArenaSize` in
`coral_micro/detect_objects_serial/detect_objects_serial.cc`, rebuild, and record
the new arena size in the profiling report.

If serial capture times out, check the serial port name and board permissions.
On Linux the device is often `/dev/ttyACM0`, but it can change after reconnects.

If detections are empty, point the camera at a COCO object class and confirm
lighting/focus. Empty detections are valid model output, not a transport error.

If the camera preview looks underexposed, distinguish sensor exposure from
display brightness. The repo-owned serial app discards 100 frames after camera
enable because the Coral camera API recommends discarding frames to calibrate
auto exposure. For the upstream RPC `detect_objects` preview app, underexposure
can still happen on the first requests after flashing or reset; leave the
preview running for several frames, improve scene lighting, or patch the
upstream app to call `CameraTask::GetSingleton()->DiscardFrames(100)` after
`Enable(CameraMode::kTrigger)`. For visual inspection only, increase
`--display-brightness-gain` in the preview server.

If flashing fails at `STATE_LOAD_FLASHLOADER` with a `blhost ... load-image`
non-zero exit status, manually reset the board into Serial Downloader mode and
retry flashing:

1. Hold the User button and press Reset, or hold the User button while plugging
   in USB.
2. Confirm the SDP USB device appears:

   ```bash
   lsusb -d 1fc9:013d
   ```

3. Rerun:

   ```bash
   cd ../coralmicro
   .venv/bin/python scripts/flashtool.py -e vlm_micro_detect_objects_serial
   ```

If running the same command with `sudo` works, the issue is host USB/HID
permissions. Install the Coral Micro udev rules instead of using sudo for normal
flashing:

```bash
cd ../coralmicro
sudo cp scripts/99-coral-micro.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
```

Then unplug/replug the board, reset it into Serial Downloader mode if needed,
and rerun the non-root flashtool command. The rules cover the ROM bootloader
`1fc9:013d`, NXP flashloader `15a2:0073`, and Coral application/bootloader USB
IDs.

If the same failure repeats, run `blhost` manually with `-V` against the
flashloader image:

```bash
cd edge_vlm
python scripts/coral_micro_debug_flashloader.py
```

The script generates a persistent
`artifacts/profiles/coral/flashloader_debug/ivt_flashloader.bin` from the Coral
SDK build output and then runs the same `blhost -u 0x1fc9,0x13d -- load-image`
step with verbose output enabled. This exposes the real NXP/HID error that
`flashtool.py` suppresses.

## Current Scope

This validates model loading, camera capture, Edge TPU invocation, serial
transport, and host-side report generation. It does not compute dataset mAP; for
that, use a fixed image/annotation replay path rather than live camera frames.

## Next Extensions

- Add a fixed-image replay app or RPC endpoint for dataset-level mAP.
- Save board/toolchain commit metadata into `artifacts/profiles/coral/report.json`.
- Add labels/class-name mapping for human-readable detection logs.
- Add power measurement fields once the measurement harness is selected.
