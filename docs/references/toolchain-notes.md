# Toolchain Notes

## Coral Dev Board Micro

Local SDK: `../coralmicro`

Observed from local README/docs and official Coral docs:

- The board runs FreeRTOS on an NXP RT1176 and has an on-board camera,
  microphone, 64 MB SDRAM, 128 MiB flash, and Edge TPU.
- ML apps use TensorFlow Lite for Microcontrollers through
  `tflite::MicroInterpreter`.
- Edge TPU models must run from the M7 side, open the Edge TPU device, and
  register the Coral custom op before interpreter creation.
- Dev Board Micro can load `.tflite` models from littlefs, so compiled model
  files do not need to be C arrays during development.
- Edge TPU acceleration requires fully 8-bit quantized TFLite models compiled
  with `edgetpu_compiler`.
- Unsupported Edge TPU ops can force CPU fallback; for Dev Board Micro, fallback
  ops must also be supported by TFLM and fit the tensor arena.

Primary references:

- https://www.coral.ai/docs/dev-board-micro/get-started/
- https://www.coral.ai/docs/reference/micro/tensorflow/
- https://www.coral.ai/docs/edgetpu/models-intro/
- https://www.coral.ai/docs/edgetpu/compiler/
- https://www.tensorflow.org/model_optimization/guide/quantization/post_training

## MAX78000

Local SDK/tooling: `../MAX78000/ai8x-training` and
`../MAX78000/ai8x-synthesis`

Observed from local ADI README/docs and official ADI pages:

- Current ADI tooling is PyTorch 2 based. TensorFlow/Keras support is documented
  as deprecated in the local README.
- The supported training environment is Python 3.11.x and PyTorch 2.3 in the
  ADI docs; keep this repo on Python 3.11 unless there is a specific reason to
  diverge.
- The synthesis repo converts checkpoints plus network YAML into generated C via
  `ai8xize.py`.
- MAX78000 weights are 1, 2, 4, or 8 bits by layer; activations/data are
  generally 8-bit except specific output modes.
- Hardware simulation and `-8` evaluation are important before board flashing.
- `ai8xize.py --energy` exists for EVKit energy measurements, but FTHR power
  measurement may need an external meter or board-specific harness.

Primary references:

- https://github.com/analogdevicesinc/ai8x-training
- https://github.com/analogdevicesinc/ai8x-synthesis
- https://github.com/analogdevicesinc/MaximAI_Documentation
- https://www.analog.com/en/products/max78000
