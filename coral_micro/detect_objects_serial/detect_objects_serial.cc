// Serial-first object detection bring-up app for Coral Dev Board Micro.
//
// The app captures RGB frames from the onboard camera, invokes an Edge TPU SSD
// detector, and prints one newline-delimited JSON record per inference. Host
// tooling should filter lines with the VLM_MICRO_DETECTION prefix.

#include <cstring>
#include <vector>

#include "libs/base/filesystem.h"
#include "libs/base/led.h"
#include "libs/base/timer.h"
#include "libs/camera/camera.h"
#include "libs/tensorflow/detection.h"
#include "libs/tensorflow/utils.h"
#include "libs/tpu/edgetpu_manager.h"
#include "libs/tpu/edgetpu_op.h"
#include "third_party/freertos_kernel/include/FreeRTOS.h"
#include "third_party/freertos_kernel/include/task.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_error_reporter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_interpreter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_mutable_op_resolver.h"

namespace coralmicro {
namespace {

constexpr char kModelPath[] =
    "/models/tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite";
constexpr int kTensorArenaSize = 8 * 1024 * 1024;
constexpr float kScoreThreshold = 0.5f;
constexpr size_t kTopK = 10;
constexpr uint32_t kFrameDelayMs = 1000;
constexpr int kCameraWarmupDiscardFrames = 100;

STATIC_TENSOR_ARENA_IN_SDRAM(tensor_arena, kTensorArenaSize);

void PrintDetectionJson(uint32_t frame_id, int width, int height,
                        uint64_t capture_us, uint64_t invoke_us,
                        const std::vector<tensorflow::Object>& results) {
  printf(
      "VLM_MICRO_DETECTION {\"frame_id\":%lu,\"width\":%d,\"height\":%d,"
      "\"capture_us\":%llu,\"invoke_us\":%llu,\"threshold\":%.3f,"
      "\"top_k\":%u,\"detections\":[",
      static_cast<unsigned long>(frame_id), width, height,
      static_cast<unsigned long long>(capture_us),
      static_cast<unsigned long long>(invoke_us), kScoreThreshold,
      static_cast<unsigned int>(kTopK));

  for (size_t i = 0; i < results.size(); ++i) {
    const auto& result = results[i];
    printf(
        "%s{\"id\":%d,\"score\":%.6f,\"bbox\":{\"ymin\":%.6f,"
        "\"xmin\":%.6f,\"ymax\":%.6f,\"xmax\":%.6f}}",
        i == 0 ? "" : ",", result.id, result.score, result.bbox.ymin,
        result.bbox.xmin, result.bbox.ymax, result.bbox.xmax);
  }
  printf("]}\r\n");
}

bool CaptureAndDetect(tflite::MicroInterpreter* interpreter, uint32_t frame_id,
                      std::vector<uint8_t>* image) {
  auto* input_tensor = interpreter->input_tensor(0);
  const int model_height = input_tensor->dims->data[1];
  const int model_width = input_tensor->dims->data[2];

  CameraFrameFormat fmt{CameraFormat::kRgb,   CameraFilterMethod::kBilinear,
                        CameraRotation::k270, model_width,
                        model_height,         false,
                        image->data()};

  const uint64_t capture_start_us = TimerMicros();
  CameraTask::GetSingleton()->Trigger();
  if (!CameraTask::GetSingleton()->GetFrame({fmt})) {
    printf("VLM_MICRO_ERROR {\"frame_id\":%lu,\"error\":\"camera_frame\"}\r\n",
           static_cast<unsigned long>(frame_id));
    return false;
  }
  const uint64_t capture_us = TimerMicros() - capture_start_us;

  std::memcpy(tflite::GetTensorData<uint8_t>(input_tensor), image->data(),
              image->size());

  const uint64_t invoke_start_us = TimerMicros();
  if (interpreter->Invoke() != kTfLiteOk) {
    printf("VLM_MICRO_ERROR {\"frame_id\":%lu,\"error\":\"invoke\"}\r\n",
           static_cast<unsigned long>(frame_id));
    return false;
  }
  const uint64_t invoke_us = TimerMicros() - invoke_start_us;

  auto results =
      tensorflow::GetDetectionResults(interpreter, kScoreThreshold, kTopK);
  PrintDetectionJson(frame_id, model_width, model_height, capture_us, invoke_us,
                     results);
  return true;
}

[[noreturn]] void Main() {
  printf("VLM Micro serial detection app starting.\r\n");
  LedSet(Led::kStatus, true);

  std::vector<uint8_t> model;
  if (!LfsReadFile(kModelPath, &model)) {
    printf("VLM_MICRO_ERROR {\"error\":\"model_load\",\"path\":\"%s\"}\r\n",
           kModelPath);
    vTaskSuspend(nullptr);
  }

  auto tpu_context = EdgeTpuManager::GetSingleton()->OpenDevice();
  if (!tpu_context) {
    printf("VLM_MICRO_ERROR {\"error\":\"edgetpu_open\"}\r\n");
    vTaskSuspend(nullptr);
  }

  tflite::MicroErrorReporter error_reporter;
  tflite::MicroMutableOpResolver<3> resolver;
  resolver.AddDequantize();
  resolver.AddDetectionPostprocess();
  resolver.AddCustom(kCustomOp, RegisterCustomOp());

  tflite::MicroInterpreter interpreter(tflite::GetModel(model.data()), resolver,
                                       tensor_arena, kTensorArenaSize,
                                       &error_reporter);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    printf("VLM_MICRO_ERROR {\"error\":\"allocate_tensors\"}\r\n");
    vTaskSuspend(nullptr);
  }
  if (interpreter.inputs().size() != 1) {
    printf("VLM_MICRO_ERROR {\"error\":\"input_count\",\"count\":%u}\r\n",
           static_cast<unsigned int>(interpreter.inputs().size()));
    vTaskSuspend(nullptr);
  }

  auto* input_tensor = interpreter.input_tensor(0);
  const int model_height = input_tensor->dims->data[1];
  const int model_width = input_tensor->dims->data[2];
  std::vector<uint8_t> image(model_height * model_width *
                             CameraFormatBpp(CameraFormat::kRgb));

  CameraTask::GetSingleton()->SetPower(true);
  CameraTask::GetSingleton()->Enable(CameraMode::kTrigger);
  CameraTask::GetSingleton()->DiscardFrames(kCameraWarmupDiscardFrames);

  printf(
      "VLM_MICRO_READY {\"model_path\":\"%s\",\"width\":%d,\"height\":%d,"
      "\"tensor_arena_bytes\":%d,\"threshold\":%.3f,\"top_k\":%u,"
      "\"camera_warmup_discard_frames\":%d}\r\n",
      kModelPath, model_width, model_height, kTensorArenaSize, kScoreThreshold,
      static_cast<unsigned int>(kTopK), kCameraWarmupDiscardFrames);

  uint32_t frame_id = 0;
  while (true) {
    CaptureAndDetect(&interpreter, frame_id++, &image);
    vTaskDelay(pdMS_TO_TICKS(kFrameDelayMs));
  }
}

}  // namespace
}  // namespace coralmicro

extern "C" void app_main(void* param) {
  (void)param;
  coralmicro::Main();
}
