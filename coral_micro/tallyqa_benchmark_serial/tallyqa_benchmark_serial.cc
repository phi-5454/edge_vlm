// Self-contained Coral Micro seeded inference benchmark.
//
// The board loads the staged Edge TPU model, fills inputs with deterministic
// pseudo-random data on-device, invokes repeatedly, and emits newline-delimited
// JSON timing records over USB serial. This intentionally avoids host payloads
// and per-inference output dumps so serial I/O does not dominate the benchmark.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "libs/base/filesystem.h"
#include "libs/base/led.h"
#include "libs/base/timer.h"
#include "libs/tensorflow/utils.h"
#include "libs/tpu/edgetpu_manager.h"
#include "libs/tpu/edgetpu_op.h"
#include "vlm_micro_selftest_config.h"
#if VLM_MICRO_ENABLE_PROMPT_LOOKUP
#include "tallyqa_prompt_embedding_lookup.h"
#else
#define TALLYQA_PROMPT_EMBEDDING_COUNT 0
#define TALLYQA_PROMPT_EMBEDDING_DIM 0
#endif
#include "third_party/freertos_kernel/include/FreeRTOS.h"
#include "third_party/freertos_kernel/include/task.h"
#include "third_party/tflite-micro/tensorflow/lite/c/common.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_error_reporter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_interpreter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "third_party/tflite-micro/tensorflow/lite/schema/schema_generated.h"

namespace coralmicro {
namespace {

constexpr char kModelPath[] = "/models/tallyqa_prompt_patch_mlp_edgetpu.tflite";
constexpr int kTensorArenaSize = 8 * 1024 * 1024;
constexpr uint32_t kSelfTestSeed = 0x5EED1234;
constexpr int kSelfTestWarmupIterations = 3;
constexpr int kSelfTestMeasuredIterations = 100;
constexpr int kSelfTestRepeatDelayMs = 5000;

STATIC_TENSOR_ARENA_IN_SDRAM(tensor_arena, kTensorArenaSize);

void PrintError(const char* error) {
  printf("VLM_MICRO_ERROR {\"error\":\"%s\"}\r\n", error);
  fflush(stdout);
}

void PrintStage(const char* stage) {
  printf("VLM_MICRO_EVENT {\"stage\":\"%s\"}\r\n", stage);
  fflush(stdout);
}

void PrintStageWithValue(const char* stage, const char* key, uint32_t value) {
  printf("VLM_MICRO_EVENT {\"stage\":\"%s\",\"%s\":%lu}\r\n", stage, key,
         static_cast<unsigned long>(value));
  fflush(stdout);
}

uint32_t NextRandom(uint32_t* state) {
  *state = (*state * 1664525u) + 1013904223u;
  return *state;
}

int TensorElementCount(const TfLiteTensor* tensor) {
  int count = 1;
  for (int i = 0; i < tensor->dims->size; ++i) {
    count *= tensor->dims->data[i];
  }
  return count;
}

void FillTensorDeterministic(TfLiteTensor* tensor, uint32_t* state) {
  if (tensor == nullptr) return;
  if (tensor->type == kTfLiteUInt8) {
    for (size_t i = 0; i < tensor->bytes; ++i) {
      tensor->data.uint8[i] =
          static_cast<uint8_t>((NextRandom(state) >> 24) & 0xff);
    }
    return;
  }
  if (tensor->type == kTfLiteInt8) {
    for (size_t i = 0; i < tensor->bytes; ++i) {
      tensor->data.int8[i] =
          static_cast<int8_t>((NextRandom(state) >> 24) - 128);
    }
    return;
  }
  if (tensor->type == kTfLiteFloat32) {
    const int count = TensorElementCount(tensor);
    for (int i = 0; i < count; ++i) {
      const int value =
          static_cast<int>((NextRandom(state) >> 24) & 0xff) - 128;
      tensor->data.f[i] = static_cast<float>(value) / 128.0f;
    }
    return;
  }
  std::memset(tensor->data.raw, 0, tensor->bytes);
}

const char* TensorTypeName(TfLiteType type) {
  switch (type) {
    case kTfLiteFloat32:
      return "float32";
    case kTfLiteInt32:
      return "int32";
    case kTfLiteUInt8:
      return "uint8";
    case kTfLiteInt8:
      return "int8";
    default:
      return "other";
  }
}

void PrintDims(const TfLiteTensor* tensor) {
  printf("[");
  for (int i = 0; i < tensor->dims->size; ++i) {
    printf("%s%d", i == 0 ? "" : ",", tensor->dims->data[i]);
  }
  printf("]");
}

bool IsImageTensor(const TfLiteTensor* tensor) {
  return tensor != nullptr && tensor->dims != nullptr &&
         tensor->dims->size == 4 && tensor->dims->data[0] == 1 &&
         tensor->dims->data[3] == 3;
}

bool IsPromptTensor(const TfLiteTensor* tensor) {
#if VLM_MICRO_ENABLE_PROMPT_LOOKUP
  if (tensor == nullptr || tensor->dims == nullptr) return false;
  if (tensor->bytes != TALLYQA_PROMPT_EMBEDDING_DIM) return false;
  if (tensor->type != kTfLiteUInt8) return false;
  if (tensor->dims->size == 2) {
    return tensor->dims->data[0] == 1 &&
           tensor->dims->data[1] == TALLYQA_PROMPT_EMBEDDING_DIM;
  }
  if (tensor->dims->size == 1) {
    return tensor->dims->data[0] == TALLYQA_PROMPT_EMBEDDING_DIM;
  }
  return false;
#else
  (void)tensor;
  return false;
#endif
}

int FindInputIndex(tflite::MicroInterpreter* interpreter,
                   bool (*predicate)(const TfLiteTensor*)) {
  for (size_t i = 0; i < interpreter->inputs().size(); ++i) {
    if (predicate(interpreter->input_tensor(i))) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

void PrintTensorSummaryJson(const TfLiteTensor* tensor) {
  printf("{\"bytes\":%u,\"type\":\"%s\",\"scale\":%.9g,\"zero_point\":%ld,"
         "\"shape\":",
         static_cast<unsigned int>(tensor->bytes), TensorTypeName(tensor->type),
         tensor->params.scale, static_cast<long>(tensor->params.zero_point));
  PrintDims(tensor);
  printf("}");
}

void PrintReadyJson(tflite::MicroInterpreter* interpreter) {
  const int image_input_index = FindInputIndex(interpreter, IsImageTensor);
  const int prompt_input_index = FindInputIndex(interpreter, IsPromptTensor);
  const TfLiteTensor* image_input =
      interpreter->input_tensor(image_input_index >= 0 ? image_input_index : 0);
  const TfLiteTensor* output = interpreter->output_tensor(0);
  printf(
      "VLM_MICRO_READY {\"model_path\":\"%s\",\"model_kind\":\"%s\","
      "\"tensor_arena_bytes\":%d,\"arena_used_bytes\":%u,"
      "\"input_count\":%u,\"output_count\":%u,\"image_input_index\":%d,"
      "\"prompt_input_index\":%d,\"prompt_lookup_count\":%d,"
      "\"prompt_lookup_dim\":%d,\"image_input\":",
      kModelPath, VLM_MICRO_MODEL_KIND, kTensorArenaSize,
      static_cast<unsigned int>(interpreter->arena_used_bytes()),
      static_cast<unsigned int>(interpreter->inputs().size()),
      static_cast<unsigned int>(interpreter->outputs().size()),
      image_input_index, prompt_input_index, TALLYQA_PROMPT_EMBEDDING_COUNT,
      TALLYQA_PROMPT_EMBEDDING_DIM);
  PrintTensorSummaryJson(image_input);
  printf(",\"output\":");
  PrintTensorSummaryJson(output);
  printf("}\r\n");
  fflush(stdout);
}

void PrintSelfTestResultJson(uint32_t run_id, int iteration, bool warmup,
                             uint32_t seed, uint32_t prompt_id,
                             uint64_t fill_us, uint64_t prompt_copy_us,
                             uint64_t invoke_us) {
  printf(
      "VLM_MICRO_SELFTEST_RESULT {\"run_id\":%lu,\"iteration\":%d,"
      "\"warmup\":%s,\"seed\":%lu,\"prompt_id\":%lu,\"fill_us\":%lu,"
      "\"prompt_copy_us\":%lu,\"copy_us\":%lu,\"invoke_us\":%lu}\r\n",
      static_cast<unsigned long>(run_id), iteration, warmup ? "true" : "false",
      static_cast<unsigned long>(seed), static_cast<unsigned long>(prompt_id),
      static_cast<unsigned long>(fill_us),
      static_cast<unsigned long>(prompt_copy_us),
      static_cast<unsigned long>(fill_us + prompt_copy_us),
      static_cast<unsigned long>(invoke_us));
  fflush(stdout);
}

void PrintSelfTestSummaryJson(uint32_t run_id, int measured_count,
                              uint64_t min_invoke_us,
                              uint64_t max_invoke_us,
                              uint64_t total_invoke_us,
                              uint64_t total_copy_us) {
  const uint64_t avg_invoke_us =
      measured_count > 0 ? total_invoke_us / measured_count : 0;
  const uint64_t avg_copy_us =
      measured_count > 0 ? total_copy_us / measured_count : 0;
  printf(
      "VLM_MICRO_SELFTEST_SUMMARY {\"run_id\":%lu,\"seed\":%lu,"
      "\"warmup_iterations\":%d,\"measured_iterations\":%d,"
      "\"invoke_min_us\":%lu,\"invoke_avg_us\":%lu,\"invoke_max_us\":%lu,"
      "\"copy_avg_us\":%lu}\r\n",
      static_cast<unsigned long>(run_id),
      static_cast<unsigned long>(kSelfTestSeed), kSelfTestWarmupIterations,
      measured_count, static_cast<unsigned long>(min_invoke_us),
      static_cast<unsigned long>(avg_invoke_us),
      static_cast<unsigned long>(max_invoke_us),
      static_cast<unsigned long>(avg_copy_us));
  fflush(stdout);
}

void CopyPromptEmbedding(TfLiteTensor* prompt_input, uint32_t prompt_id) {
#if VLM_MICRO_ENABLE_PROMPT_LOOKUP
  if (prompt_input == nullptr) return;
  std::memcpy(prompt_input->data.raw,
              kTallyQAPromptEmbeddingTable[prompt_id],
              TALLYQA_PROMPT_EMBEDDING_DIM);
#else
  (void)prompt_input;
  (void)prompt_id;
#endif
}

void RunSelfTestLoop(tflite::MicroInterpreter* interpreter,
                     TfLiteTensor* image_input, TfLiteTensor* prompt_input) {
  uint32_t run_id = 0;
  while (true) {
    const uint32_t run_seed = kSelfTestSeed + run_id * 7919u;
    uint64_t min_invoke_us = UINT64_MAX;
    uint64_t max_invoke_us = 0;
    uint64_t total_invoke_us = 0;
    uint64_t total_copy_us = 0;
    int measured_count = 0;
    printf(
        "VLM_MICRO_SELFTEST_BEGIN {\"run_id\":%lu,\"seed\":%lu,"
        "\"warmup_iterations\":%d,\"measured_iterations\":%d,"
        "\"prompt_lookup_count\":%d}\r\n",
        static_cast<unsigned long>(run_id), static_cast<unsigned long>(run_seed),
        kSelfTestWarmupIterations, kSelfTestMeasuredIterations,
        TALLYQA_PROMPT_EMBEDDING_COUNT);
    fflush(stdout);

    const int total_iterations =
        kSelfTestWarmupIterations + kSelfTestMeasuredIterations;
    for (int iteration = 0; iteration < total_iterations; ++iteration) {
      const bool warmup = iteration < kSelfTestWarmupIterations;
      uint32_t state = run_seed + static_cast<uint32_t>(iteration) * 2654435761u;
      const uint32_t prompt_id =
          TALLYQA_PROMPT_EMBEDDING_COUNT > 0
              ? (NextRandom(&state) % TALLYQA_PROMPT_EMBEDDING_COUNT)
              : 0;
      if (iteration == 0) {
        PrintStageWithValue("first_iteration_started", "run_id", run_id);
      }

      const uint64_t fill_start_us = TimerMicros();
      if (iteration == 0) PrintStage("first_input_fill_started");
      FillTensorDeterministic(image_input, &state);
      const uint64_t fill_us = TimerMicros() - fill_start_us;
      if (iteration == 0) PrintStage("first_input_fill_done");

      uint64_t prompt_copy_us = 0;
      if (prompt_input != nullptr) {
        if (iteration == 0) PrintStage("first_prompt_copy_started");
        const uint64_t prompt_copy_start_us = TimerMicros();
        CopyPromptEmbedding(prompt_input, prompt_id);
        prompt_copy_us = TimerMicros() - prompt_copy_start_us;
        if (iteration == 0) PrintStage("first_prompt_copy_done");
      }

      vTaskDelay(pdMS_TO_TICKS(10));
      if (iteration == 0) PrintStage("first_invoke_started");
      const uint64_t invoke_start_us = TimerMicros();
      if (interpreter->Invoke() != kTfLiteOk) {
        printf(
            "VLM_MICRO_ERROR {\"run_id\":%lu,\"iteration\":%d,"
            "\"error\":\"invoke\"}\r\n",
            static_cast<unsigned long>(run_id), iteration);
        fflush(stdout);
        continue;
      }
      const uint64_t invoke_us = TimerMicros() - invoke_start_us;
      if (iteration == 0) {
        PrintStageWithValue("executed_once", "invoke_us",
                            static_cast<uint32_t>(invoke_us));
      }
      PrintSelfTestResultJson(run_id, iteration, warmup, run_seed, prompt_id,
                              fill_us, prompt_copy_us, invoke_us);

      if (!warmup) {
        min_invoke_us = invoke_us < min_invoke_us ? invoke_us : min_invoke_us;
        max_invoke_us = invoke_us > max_invoke_us ? invoke_us : max_invoke_us;
        total_invoke_us += invoke_us;
        total_copy_us += fill_us + prompt_copy_us;
        ++measured_count;
      }
    }
    PrintSelfTestSummaryJson(run_id, measured_count,
                             measured_count > 0 ? min_invoke_us : 0,
                             max_invoke_us, total_invoke_us, total_copy_us);
    ++run_id;
    vTaskDelay(pdMS_TO_TICKS(kSelfTestRepeatDelayMs));
  }
}

void Main() {
  printf("VLM Micro self-test benchmark app starting.\r\n");
  fflush(stdout);
  LedSet(Led::kStatus, true);

  std::vector<uint8_t> model;
  PrintStage("model_load_started");
  if (!LfsReadFile(kModelPath, &model)) {
    printf("VLM_MICRO_ERROR {\"error\":\"model_load\",\"path\":\"%s\"}\r\n",
           kModelPath);
    fflush(stdout);
    vTaskSuspend(nullptr);
  }
  PrintStageWithValue("model_load_done", "model_bytes",
                      static_cast<uint32_t>(model.size()));

  PrintStage("edgetpu_open_started");
  auto tpu_context = EdgeTpuManager::GetSingleton()->OpenDevice();
  if (!tpu_context) {
    PrintError("edgetpu_open");
    vTaskSuspend(nullptr);
  }
  PrintStage("edgetpu_open_done");

  tflite::MicroErrorReporter error_reporter;
#if VLM_MICRO_ENABLE_DETECTION_POSTPROCESS
  PrintStage("resolver_detection_configured");
  tflite::MicroMutableOpResolver<3> resolver;
  resolver.AddCustom(kCustomOp, RegisterCustomOp());
  resolver.AddDequantize();
  resolver.AddDetectionPostprocess();
#else
  PrintStage("resolver_tallyqa_configured");
  tflite::MicroMutableOpResolver<11> resolver;
  resolver.AddCustom(kCustomOp, RegisterCustomOp());
  resolver.AddAdd();
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddFullyConnected();
  resolver.AddHardSwish();
  resolver.AddMean();
  resolver.AddMul();
  resolver.AddQuantize();
  resolver.AddDequantize();
  resolver.AddSoftmax();
#endif

  tflite::MicroInterpreter interpreter(tflite::GetModel(model.data()), resolver,
                                       tensor_arena, kTensorArenaSize,
                                       &error_reporter);
  PrintStage("allocate_tensors_started");
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    PrintError("allocate_tensors");
    vTaskSuspend(nullptr);
  }
  PrintStageWithValue("allocate_tensors_done", "arena_used_bytes",
                      static_cast<uint32_t>(interpreter.arena_used_bytes()));
  if (interpreter.inputs().size() != 1 && interpreter.inputs().size() != 2) {
    printf("VLM_MICRO_ERROR {\"error\":\"input_count\",\"count\":%u}\r\n",
           static_cast<unsigned int>(interpreter.inputs().size()));
    fflush(stdout);
    vTaskSuspend(nullptr);
  }

  const int image_input_index = FindInputIndex(&interpreter, IsImageTensor);
  const int prompt_input_index = FindInputIndex(&interpreter, IsPromptTensor);
  if (image_input_index < 0) {
    PrintError("image_input_not_found");
    vTaskSuspend(nullptr);
  }
  if (interpreter.inputs().size() == 2 && prompt_input_index < 0) {
    PrintError("prompt_input_not_found");
    vTaskSuspend(nullptr);
  }

  TfLiteTensor* image_input = interpreter.input_tensor(image_input_index);
  TfLiteTensor* prompt_input =
      prompt_input_index >= 0 ? interpreter.input_tensor(prompt_input_index)
                              : nullptr;
  PrintStage("ready_print_started");
  PrintReadyJson(&interpreter);
  PrintStage("selftest_loop_entered");
  RunSelfTestLoop(&interpreter, image_input, prompt_input);
}

}  // namespace
}  // namespace coralmicro

extern "C" void app_main(void* param) {
  (void)param;
  coralmicro::Main();
}
