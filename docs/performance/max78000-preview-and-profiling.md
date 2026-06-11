# MAX78000 Preview And Profiling

This runbook defines the host-side preview and profiling path for
MAX78000/MAX78000FTHR deployments. The board app still needs to be integrated
into the generated MSDK project, but the host tooling and serial contract are
now fixed enough to support iterative firmware work.

## Pipeline Overview

The full path is:

1. Define an ADI-compatible model in `../MAX78000/ai8x-training`.
2. Train with the MAX78000 device flag and QAT policy, or train float and
   quantize after training.
3. Validate the quantized model in software with `--evaluate -8`.
4. Create or update the network YAML used by `ai8xize.py`.
5. Generate MSDK C with `../MAX78000/ai8x-synthesis/ai8xize.py`.
6. Preserve `cnn.c`, `cnn.h`, `weights.h`, and `log.txt` as generated model
   artifacts.
7. Add the camera, preprocessing, postprocessing, serial preview, and timing
   wrapper around the generated CNN API.
8. Build and flash the MSDK project.
9. Run the host preview server, retain raw serial logs, and summarize profiling.
10. Compare FP32, simulated 8-bit, generator estimates, and on-board results.

Keep all commands, toolchain commits, and raw logs with the experiment. Static
generator estimates and measured board results must stay separate in the final
report.

## Paths

Expected adjacent toolchains:

```text
../MAX78000/ai8x-training
../MAX78000/ai8x-synthesis
```

Repo-owned artifacts:

| Artifact | Purpose |
| --- | --- |
| `artifacts/exports/max78000/checkpoint.pth.tar` | Floating or QAT training checkpoint copied from `ai8x-training`. |
| `artifacts/exports/max78000/checkpoint_qat.pth.tar` | Quantized checkpoint passed to `ai8xize.py`. |
| `artifacts/exports/max78000/network.yaml` | Network YAML used for synthesis. |
| `artifacts/exports/max78000/sample.npy` | Representative sample input for known-answer generation. |
| `artifacts/exports/max78000/generated/` | Generated MSDK project or generated CNN files. |
| `artifacts/profiles/max78000/ai8xize.log` | Raw synthesis/generator log. |
| `artifacts/profiles/max78000/preview_serial.log` | Raw board serial log. |
| `artifacts/profiles/max78000/preview.jsonl` | Normalized preview/timing records. |
| `artifacts/profiles/max78000/report.json` | Normalized profiling summary. |

## Environment

Use Python 3.11 for this repo and the ADI toolchain unless a specific ADI
release requires otherwise.

```bash
uv sync --extra max78000 --extra dev
cd ../MAX78000/ai8x-training
# Install this environment according to ADI's README. Keep it outside edge_vlm.
```

Record toolchain revisions:

```bash
git -C ../MAX78000/ai8x-training rev-parse HEAD
git -C ../MAX78000/ai8x-synthesis rev-parse HEAD
```

## Model And Data Setup

The MAX78000 backend is not a generic PyTorch, ONNX, or TFLite target. Start
with an `ai8x.py` layer pattern and a network YAML that the ADI tools can map to
hardware. For the current VLM/counting work, the realistic first target should
be a compact camera model with:

- fixed input shape
- 8-bit activations/data
- 1/2/4/8-bit weights as supported by the ADI flow
- convolution/pooling/linear patterns already used by ADI examples
- postprocessing kept small enough for the Arm core

For object detection, use ADI's TinySSD/FPN detector examples as the closest
starting point. For count classification, use a simpler image classifier-style
head and emit `prediction` rather than `detections` in the serial contract.

Before training, materialize or export:

- the training dataset in the format expected by the `ai8x-training` dataset
  loader
- an `ai8x-training/models/*.py` model definition
- a QAT policy under `ai8x-training/policies/`
- object detection params under `ai8x-training/parameters/` if using a detector
- a matching `ai8x-synthesis/networks/*.yaml`

Keep copies or hashes of the model file, policy, params, and YAML in the W&B run
or under `artifacts/exports/max78000/`.

## People-Only MobileNetV3-Style Starter

The first TallyQA MAX78000 model lives in this repo at:

```text
max78000/ai8x_training/models/ai85net-tallyqa-mbv3-small.py
```

It is a MobileNetV3-small-style people-count classifier with no prompt input.
The head predicts five classes: `1`, `2`, `3`, `4`, `5+`. For first training,
filter the dataset to the `people` prompt and either drop zero-count examples or
handle zero separately before using this exact head.

Important implementation choices:

- folded 12x56x56 input, produced by resizing RGB to 112x112 and folding 2x2
- cut tensor: `14x14x112`
- average pool `14x14 -> 1x1`
- linear head `112 -> 5`
- only 1x1 and 3x3 convolutions
- no strided convolutions; downsampling is max pooling
- ReLU only
- no squeeze-excitation or hard-sigmoid path
- no depthwise convolutions, because this ADI `ai8x.py` tree restricts
  depthwise layers to MAX78002 rather than MAX78000

Stage the model into the adjacent ADI training repo:

```bash
cd /home/younes/Courses/ETH/ML_Micro/edge_vlm
uv run python scripts/stage_max78000_people_pipeline.py --force
```

After staging, ADI `train.py` should discover the model name:

```text
ai85tallyqambv3smallpeople
```

Materialize the MAX78000-friendly people-only view first:

```bash
cd /home/younes/Courses/ETH/ML_Micro/edge_vlm
uv run python scripts/materialize_max78000_people_dataset.py --force
```

This writes `data/max78000_tallyqa_people_count_fold2_56/manifest.jsonl` and
`metadata.json`. The current materialized view contains 28,180 examples:
19,620 train, 2,744 validation, and 5,816 test. Labels are positive people
counts only: `1`, `2`, `3`, `4`, and `5+`.

Stage the model, dataset adapter, QAT policy, and starter LR schedule:

```bash
cd /home/younes/Courses/ETH/ML_Micro/edge_vlm
uv run python scripts/stage_max78000_people_pipeline.py --force
```

Optional folded-frontend pretraining:

```bash
cd /home/younes/Courses/ETH/ML_Micro/edge_vlm

uv run python scripts/pretrain_max78000_folded_frontend.py \
  --dataset data/tallyqa_cauldron_target_mobilenet224_letterbox \
  --model-file max78000/ai8x_training/models/ai85net-tallyqa-mbv3-small.py \
  --teacher-backbone mobilenet_v3_large \
  --teacher-cutoff 13 \
  --prompt-class people \
  --batch-size 64 \
  --epochs 10 \
  --learning-rate 0.001 \
  --output artifacts/models/max78000/tallyqa_folded_frontend_mbv3_large_cut13.pt
```

This trains the folded MAX78000 frontend with an MSE loss against a frozen
pretrained MobileNetV3-large feature tensor at the `14x14x112` cutoff. The
resulting checkpoint is a frontend initialization artifact, not yet a full
classifier checkpoint.

Training command shape:

```bash
cd ../MAX78000/ai8x-training

uv venv --python 3.11 .venv
uv pip install -r requirements-base.txt -r requirements-datasets.txt pycocotools==2.0.8
uv pip install -e distiller --config-settings editable_mode=strict

uv run python train.py \
  --deterministic \
  --batch-size 16 \
  --epochs 50 \
  --optimizer Adam \
  --lr 0.0002 \
  --model ai85tallyqambv3smallpeople \
  --use-bias \
  --dataset tallyqa_people_count_fold2_56 \
  --data ../../edge_vlm/data/max78000_tallyqa_people_count_fold2_56 \
  --device MAX78000 \
  --qat-policy policies/qat_policy_tallyqa_people.yaml \
  --compress policies/schedule-tallyqa-people.yaml \
  --validation-split 0 \
  --print-freq 100 \
  --name tallyqa_people_mbv3small
```

Do not use `uv pip install -r requirements.txt` for this checkout. ADI's
`requirements-distiller.txt` includes `-e distiller --config-settings
editable_mode=strict`, which `uv` rejects when parsing the included requirements
file. Installing Distiller as its own command avoids that parser issue.

The next required file is the synthesis YAML matching this architecture.

## Training

Run training from the ADI training repo. A detector-style command follows the
shape of ADI's QR TinySSD example:

```bash
cd ../MAX78000/ai8x-training
python train.py \
  --deterministic \
  --batch-size 16 \
  --epochs 200 \
  --optimizer Adam \
  --lr 0.0002 \
  --wd 0 \
  --model <ai8x_model_name> \
  --use-bias \
  --dataset <dataset_name> \
  --device MAX78000 \
  --obj-detection \
  --obj-detection-params parameters/<obj_detection_params>.yaml \
  --qat-policy policies/<qat_policy>.yaml \
  --compress policies/<schedule>.yaml \
  --validation-split 0 \
  --print-freq 100
```

For a count classifier, omit `--obj-detection` and the detection params, and use
the appropriate loss/dataset/model options for the classifier.

Training outputs usually land under `../MAX78000/ai8x-training/logs/` or
`../MAX78000/ai8x-training/trained/`, depending on the command. Copy the
checkpoint selected for deployment into this repo:

```bash
mkdir -p artifacts/exports/max78000
cp ../MAX78000/ai8x-training/trained/<checkpoint>.pth.tar \
  artifacts/exports/max78000/checkpoint.pth.tar
```

Record:

- training command and config files
- checkpoint path, size, and SHA256
- FP32 or QAT validation accuracy
- W&B run URL or offline run directory
- ADI training commit

## Quantization

If the checkpoint is already a QAT checkpoint in the ADI format, use the ADI
flow's expected export directly. If post-training quantization is needed, run
`quantize.py` from `ai8x-synthesis`:

```bash
cd ../MAX78000/ai8x-synthesis
python quantize.py \
  ../../edge_vlm/artifacts/exports/max78000/checkpoint.pth.tar \
  ../../edge_vlm/artifacts/exports/max78000/checkpoint_qat.pth.tar \
  --device MAX78000 \
  -v
```

For QAT checkpoints, still make the final file name explicit:

```bash
cp ../MAX78000/ai8x-training/trained/<qat_checkpoint>.pth.tar \
  artifacts/exports/max78000/checkpoint_qat.pth.tar
```

Do not mix up these two cases in reports. Mark the quantization source as
`qat` or `ptq` and record the bit-width policy used by the network.

## Simulated 8-Bit Evaluation

Before generating C or flashing a board, evaluate the quantized checkpoint with
the ADI `-8` path. A detector-style command follows this shape:

```bash
cd ../MAX78000/ai8x-training
python train.py \
  --deterministic \
  --batch-size 16 \
  --epochs 200 \
  --model <ai8x_model_name> \
  --dataset <dataset_name> \
  --device MAX78000 \
  --obj-detection \
  --obj-detection-params parameters/<obj_detection_params>.yaml \
  --qat-policy policies/<qat_policy>.yaml \
  --compress policies/<schedule>.yaml \
  --exp-load-weights-from ../ai8x-synthesis/trained/<quantized_checkpoint>.pth.tar \
  --validation-split 0 \
  --print-freq 100 \
  --evaluate \
  -8
```

Use the equivalent classifier evaluation command for count classification.
Record accuracy/mAP/MAE as appropriate. For detectors, mAP is the primary model
quality metric; for count classifiers, keep MAE and within-one accuracy.

## Network YAML And Sample Input

`ai8xize.py` needs:

1. quantized checkpoint
2. network YAML
3. sample input `.npy`

Copy the exact YAML and representative sample into this repo:

```bash
cp ../MAX78000/ai8x-synthesis/networks/<network>.yaml \
  artifacts/exports/max78000/network.yaml
cp <sample_input>.npy artifacts/exports/max78000/sample.npy
```

For camera deployments, the sample input must have the same shape, channel
order, quantization, and normalization as the firmware preprocessing path.
Mismatch here can produce a passing generated known-answer test that does not
match live camera inference.

## C Generation

Generate an MSDK project from `ai8x-synthesis`. Use `--timer 0` so the generated
code includes timer support, and keep `--display-checkpoint --verbose` enabled
for richer logs:

```bash
cd ../MAX78000/ai8x-synthesis
mkdir -p ../../edge_vlm/artifacts/exports/max78000/generated
python ai8xize.py \
  --device MAX78000 \
  --board-name FTHR_RevA \
  --timer 0 \
  --display-checkpoint \
  --verbose \
  --test-dir ../../edge_vlm/artifacts/exports/max78000/generated \
  --prefix vlm_max78000 \
  --checkpoint-file ../../edge_vlm/artifacts/exports/max78000/checkpoint_qat.pth.tar \
  --config-file ../../edge_vlm/artifacts/exports/max78000/network.yaml \
  --sample-input ../../edge_vlm/artifacts/exports/max78000/sample.npy \
  --overwrite \
  2>&1 | tee ../../edge_vlm/artifacts/profiles/max78000/ai8xize.log
```

Detector and streaming models may need `--fifo`, `--fast-fifo`, or other
network-specific flags. Use the matching ADI example script as the first source
of truth. For example, ADI's TinySSD QR generation script uses `--fifo` and a
synthesized input.

After generation, preserve:

- `cnn.c`
- `cnn.h`
- `weights.h`
- `log.txt`
- `main.c` before local edits, if generated
- full `artifacts/profiles/max78000/ai8xize.log`

The `log.txt` and stdout log are where we expect static mapping information:
hardware ops/MACCs, software ops, per-layer ops, weight memory, bias memory,
latency estimates, and generated timer metadata.

## Firmware Integration

Keep generated `cnn.c`, `cnn.h`, `weights.h`, and `log.txt` under the
`ai8xize.py` output directory. Customize the MSDK wrapper around the generated
network, usually `main.c`, so model updates only require replacing generated
files.

The generated CNN API sequence is:

1. `cnn_enable(...)`
2. `cnn_init()`
3. `cnn_load_weights()`
4. `cnn_load_bias()`
5. `cnn_configure()`
6. `load_input()`
7. `cnn_start()`
8. wait for CNN completion
9. `cnn_unload(out_buf)`
10. `cnn_stop()` or restart sequence for the next frame

For the live preview app, wrap that sequence in a loop:

1. Capture a frame from the FTHR camera.
2. Center-crop the sensor frame to square, downsample to 112x112, fold 2x2 into
   12x56x56, then normalize
   to the exact training and synthesis input layout.
3. Load the input through generated `load_input()` or a customized equivalent.
4. Start the accelerator with `cnn_start()`.
5. Wait for completion, then call `cnn_unload()`.
6. Run postprocessing for counts/detections.
7. Emit the preview JSON line and retain raw serial logs.

Use hardware timers or cycle counters around at least:

- camera capture
- preprocessing
- input load
- CNN inference
- unload
- postprocess
- serial encode/write
- end-to-end loop

Record whether the emitted image is pre- or post-normalization. For operator
preview, it should usually be the display-space frame before normalization.

The demo camera input is not assumed to arrive as 12x56x56. The firmware wrapper
must explicitly crop and resize:

1. Read camera RGB frame with dimensions `camera_width x camera_height`.
2. Let `side = min(camera_width, camera_height)`.
3. Let `crop_x = (camera_width - side) / 2` and
   `crop_y = (camera_height - side) / 2`.
4. Sample only the centered square
   `[crop_x, crop_x + side) x [crop_y, crop_y + side)`.
5. Downsample that square to `112x112`.
6. Fold 2x2 spatial neighborhoods into channels: `3x112x112 -> 12x56x56`.
7. Convert/reorder to the generated model input layout.

This keeps each input channel below the MAX78000 8192-byte per-channel limit:
`56 * 56 = 3136` bytes. Do not feed a larger camera tensor into the generated
CNN app and rely on later layers to shrink it; the input itself must satisfy the
memory layout constraint.

## Build And Flash

Generated projects use ADI/MSDK makefiles. The exact command depends on how
`ai8xize.py` laid out the target and how the local MSDK/OpenOCD environment is
installed. The common pattern is:

```bash
cd artifacts/exports/max78000/generated/MAX78000/CNN/vlm_max78000
make clean
make -j
make flash.openocd
```

If the generated directory layout differs, build from the generated project
directory containing the MSDK `Makefile`.

Before flashing, confirm:

- board is `FTHR_RevA` unless using EVKit
- OpenOCD can see the debug probe
- serial port permissions are set
- the board can still be recovered if the app hangs early during CNN power/load

For first hardware tests, keep a generated known-answer test path available
before enabling live camera looping. That separates accelerator loading errors
from camera/preprocessing errors.

## Host Preview

Run against a live serial port:

```bash
uv run --extra max78000 python scripts/max78000_preview_server.py \
  --port /dev/ttyACM0 \
  --baud 115200 \
  --jsonl artifacts/profiles/max78000/preview.jsonl \
  --raw-log artifacts/profiles/max78000/preview_serial.log
```

Open `http://127.0.0.1:8775`.

Replay an existing preview JSONL without a board:

```bash
uv run --extra max78000 python scripts/max78000_preview_server.py \
  --replay-jsonl artifacts/profiles/max78000/preview.jsonl
```

The browser view draws detection boxes when the serial records contain boxes,
shows the primary prediction, and writes normalized records to
`artifacts/profiles/max78000/preview.jsonl`.

## Serial Contract

The board firmware should emit newline-delimited JSON. The host accepts raw JSON
lines or lines prefixed with one of:

- `VLM_MAX78000_PREVIEW`
- `VLM_MAX78000_FRAME`
- `VLM_MICRO_MAX78000_PREVIEW`
- `VLM_MICRO_MAX78000_FRAME`

Minimum useful record:

```json
{
  "event": "preview",
  "frame_id": 17,
  "image": {
    "width": 96,
    "height": 96,
    "encoding": "rgb565le",
    "data_b64": "..."
  },
  "prediction": {
    "id": 3,
    "score": 0.87
  },
  "timing": {
    "capture_us": 12000,
    "load_input_us": 900,
    "inference_us": 5400,
    "unload_us": 200,
    "postprocess_us": 300,
    "end_to_end_us": 18800
  }
}
```

For object detection/counting models, prefer:

```json
{
  "detections": [
    {
      "id": 1,
      "score": 0.91,
      "xmin": 0.14,
      "ymin": 0.18,
      "xmax": 0.54,
      "ymax": 0.78
    }
  ]
}
```

Supported image encodings are `rgb565le`, `rgb565be`, `rgb888`, `gray8`, and
`jpeg`. Use a small preview frame first, for example `96x96` RGB565, because
base64 image transport over UART is intentionally a debug path rather than a
high-frame-rate video stream.

## Profiling Summary

Normalize static synthesis logs and live serial timing into one report:

```bash
uv run python scripts/summarize_max78000_profile.py \
  --synthesis-log artifacts/profiles/max78000/ai8xize.log \
  --serial-jsonl artifacts/profiles/max78000/preview.jsonl \
  --generated-project artifacts/exports/max78000/generated \
  --output artifacts/profiles/max78000/report.json
```

The summarizer extracts, when present:

- hardware ops and MACCs from `SUMMARY OF OPS`
- software ops for CPU-side layers
- per-layer ops and MACCs
- CNN weight memory bytes and utilization
- CNN bias memory bytes and utilization
- generated project file sizes and hashes
- board timing summaries from serial JSONL
- prediction counts from preview records

Additional fields to add once firmware exposes them:

- processor map per layer from the network YAML/generated log
- data SRAM/TRAM allocation by layer
- measured `cnn_time` or cycle-counter latency
- CPU postprocess latency
- EVKit `--energy` results, or external FTHR rail measurements

## Board Testing

Run board testing in stages. Do not start with a live camera loop if the
generated known-answer test has not passed.

1. **Generated KAT boot test**: flash the unmodified generated project, or a
   wrapper that still runs `check_output()`, and confirm the serial console
   reports pass/fail clearly.
2. **Static sample through wrapper**: replace `main.c` with the preview/timing
   wrapper, but feed the same `sample.npy` payload. Confirm the output matches
   the generated expected output or the Python reference postprocess.
3. **Single camera frame**: capture one frame, run one inference, emit one JSON
   preview record, then stop or wait. Confirm the host preview can decode it.
4. **Continuous preview**: enable the frame loop and run
   `scripts/max78000_preview_server.py`.
5. **Profiling capture**: collect at least 100 preview records for timing
   summaries, or more if serial image transfer dominates the loop.
6. **Accuracy replay**: for real accuracy/mAP/MAE, use a fixed image replay
   harness or a board-fed validation subset. Live camera preview is a bring-up
   and UX check, not a benchmark dataset.

Minimum board JSON timing fields:

- `capture_us`
- `preprocess_us`
- `load_input_us`
- `inference_us`
- `unload_us`
- `postprocess_us`
- `serial_write_us`
- `end_to_end_us`

Prefer cycle counts as well as microseconds when possible:

```json
{
  "timing": {
    "inference_cycles": 270000,
    "inference_us": 5400
  }
}
```

For power/energy:

- EVKit: use the ADI `ai8xize.py --energy` path where supported, and preserve
  the raw generated/serial logs.
- FTHR: use an external rail measurement setup unless a board-specific power
  harness is added. Record voltage rail, sampling rate, shunt/resistor value,
  whether camera and serial are active, and whether the timing window covers
  only CNN inference or the whole loop.

## Experiment Summary

Every MAX78000 deployment experiment should end with:

```bash
uv run python scripts/summarize_max78000_profile.py \
  --synthesis-log artifacts/profiles/max78000/ai8xize.log \
  --serial-jsonl artifacts/profiles/max78000/preview.jsonl \
  --generated-project artifacts/exports/max78000/generated \
  --output artifacts/profiles/max78000/report.json

uv run vlm-micro artifact-report \
  artifacts/exports/max78000/checkpoint.pth.tar \
  artifacts/exports/max78000/checkpoint_qat.pth.tar \
  artifacts/exports/max78000/network.yaml \
  artifacts/exports/max78000/sample.npy \
  artifacts/profiles/max78000/report.json
```

Record a comparison row for each available stage:

| Stage | Quality Metric | Size | Latency | Energy/Power | Source |
| --- | ---: | ---: | ---: | ---: | --- |
| PyTorch FP32 | | checkpoint bytes | validation runtime | | W&B/training log |
| ADI simulated 8-bit | | quantized checkpoint bytes | software eval runtime | | `--evaluate -8` log |
| ai8xize estimate | | generated project bytes | estimated cycles | optional `--energy` | `ai8xize.log` |
| MAX78000 board KAT | pass/fail | flashed image bytes | `inference_us` | measured if available | serial JSONL |
| MAX78000 board camera | preview quality or replay metric | flashed image bytes | end-to-end loop | measured if available | preview JSONL |

For detectors, use mAP as the primary quality metric and log score thresholds
used for preview. For count classifiers, use MAE, within-one accuracy, and the
same class-weighted metrics used in training.

## Troubleshooting

If `ai8xize.py` fails before C generation, check the YAML layer sequence,
processor maps, input dimensions, and whether every layer has a supported
quantization setting.

If generated C builds but the KAT fails on board, verify sample input channel
order, input offset/FIFO mode, output unload shape, and whether local changes to
`main.c` altered the generated load/start/unload sequence.

If live camera outputs look wrong but KAT passes, focus on preprocessing:
resize/crop, color order, signedness, quantization zero point, and normalization.

If preview JSON reaches the host but the image does not render, reduce to
`gray8` or a tiny `rgb565le` frame and validate the `width`, `height`,
`encoding`, and `data_b64` payload before adding boxes or full-resolution
frames.

## Notes

Compared with Coral, the MAX78000 path should give substantially better
hardware mapping visibility because the ADI generator knows processor maps,
kernel/bias memory use, hardware MACCs, software fallback ops, and latency
estimates. Do not report these as board measurements unless they came from the
board serial log or an EVKit/external power measurement; keep generator estimates
separate in `artifacts/profiles/max78000/report.json`.
