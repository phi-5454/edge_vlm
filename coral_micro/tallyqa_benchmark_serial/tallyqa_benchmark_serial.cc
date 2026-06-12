// Self-contained TallyQA benchmark app for Coral Dev Board Micro.
//
// The board loads the staged EdgeTPU model, fills model inputs with deterministic
// seeded data on-device, invokes repeatedly, and emits newline-delimited JSON
// timing/output records over the USB serial console.

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "libs/base/console_m7.h"
#include "libs/base/filesystem.h"
#include "libs/base/led.h"
#include "libs/base/timer.h"
#include "libs/tensorflow/utils.h"
#include "libs/tpu/edgetpu_manager.h"
#include "libs/tpu/edgetpu_op.h"
#include "tallyqa_prompt_embedding_lookup.h"
#include "third_party/freertos_kernel/include/FreeRTOS.h"
#include "third_party/freertos_kernel/include/task.h"
#include "third_party/tflite-micro/tensorflow/lite/c/common.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_error_reporter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_interpreter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/recording_micro_allocator.h"
#include "third_party/tflite-micro/tensorflow/lite/schema/schema_generated.h"

namespace coralmicro {
namespace {

constexpr char kModelPath[] = "/models/tallyqa_prompt_patch_mlp_edgetpu.tflite";
constexpr int kTensorArenaSize = 8 * 1024 * 1024;
constexpr int kMaxLineBytes = 512;
constexpr int kMaxOutputValuesPerTensor = 256;
constexpr uint32_t kSelfTestSeed = 0x5EED1234;
constexpr int kSelfTestWarmupIterations = 3;
constexpr int kSelfTestMeasuredIterations = 32;
constexpr int kSelfTestRepeatDelayMs = 5000;

STATIC_TENSOR_ARENA_IN_SDRAM(tensor_arena, kTensorArenaSize);

struct InputHeader {
  uint32_t dataset_index = 0;
  uint32_t image_index = 0;
  uint32_t prompt_id = 0;
  uint32_t bytes = 0;
  bool has_prompt_id = false;
};

void PrintReadyJson(tflite::MicroInterpreter* interpreter,
                    const tflite::RecordingMicroAllocator* allocator);
int TensorElementCount(const TfLiteTensor* tensor);

void PrintError(const char* error) {
  printf("VLM_MICRO_ERROR {\"error\":\"%s\"}\r\n", error);
}

void PrintEvent(const char* event, const InputHeader& header) {
  printf("VLM_MICRO_EVENT {\"event\":\"%s\",\"dataset_index\":%lu,"
         "\"image_index\":%lu,\"prompt_id\":%lu,\"bytes\":%lu}\r\n",
         event, static_cast<unsigned long>(header.dataset_index),
         static_cast<unsigned long>(header.image_index),
         static_cast<unsigned long>(header.prompt_id),
         static_cast<unsigned long>(header.bytes));
}

void PrintTimedEvent(const char* event, const InputHeader& header,
                     uint64_t elapsed_us) {
  printf("VLM_MICRO_EVENT {\"event\":\"%s\",\"dataset_index\":%lu,"
         "\"image_index\":%lu,\"prompt_id\":%lu,\"bytes\":%lu,"
         "\"elapsed_us\":%lu}\r\n",
         event, static_cast<unsigned long>(header.dataset_index),
         static_cast<unsigned long>(header.image_index),
         static_cast<unsigned long>(header.prompt_id),
         static_cast<unsigned long>(header.bytes),
         static_cast<unsigned long>(elapsed_us));
}

uint32_t NextRandom(uint32_t* state) {
  *state = (*state * 1664525u) + 1013904223u;
  return *state;
}

void FillTensorDeterministic(TfLiteTensor* tensor, uint32_t* state) {
  if (tensor == nullptr) return;
  if (tensor->type == kTfLiteUInt8) {
    for (size_t i = 0; i < tensor->bytes; ++i) {
      tensor->data.uint8[i] = static_cast<uint8_t>((NextRandom(state) >> 24) & 0xff);
    }
    return;
  }
  if (tensor->type == kTfLiteInt8) {
    for (size_t i = 0; i < tensor->bytes; ++i) {
      tensor->data.int8[i] = static_cast<int8_t>((NextRandom(state) >> 24) - 128);
    }
    return;
  }
  if (tensor->type == kTfLiteFloat32) {
    const int count = TensorElementCount(tensor);
    for (int i = 0; i < count; ++i) {
      const int value = static_cast<int>((NextRandom(state) >> 24) & 0xff) - 128;
      tensor->data.f[i] = static_cast<float>(value) / 128.0f;
    }
    return;
  }
  std::memset(tensor->data.raw, 0, tensor->bytes);
}

bool ReadExact(uint8_t* data, size_t bytes) {
  size_t offset = 0;
  while (offset < bytes) {
    const int read = ConsoleM7::GetSingleton()->Read(
        reinterpret_cast<char*>(data + offset), bytes - offset);
    if (read > 0) {
      offset += static_cast<size_t>(read);
    } else {
      taskYIELD();
    }
  }
  return true;
}

bool ReadLine(char* line, size_t capacity, tflite::MicroInterpreter* interpreter,
              const tflite::RecordingMicroAllocator* allocator) {
  size_t offset = 0;
  uint64_t last_ready_us = TimerMicros();
  while (offset + 1 < capacity) {
    char ch;
    const int read = ConsoleM7::GetSingleton()->Read(&ch, 1);
    if (read != 1) {
      const uint64_t now_us = TimerMicros();
      if (offset == 0 && now_us - last_ready_us > 1000000) {
        PrintReadyJson(interpreter, allocator);
        last_ready_us = now_us;
      }
      taskYIELD();
      continue;
    }
    if (ch == '\r') {
      continue;
    }
    if (ch == '\n') {
      line[offset] = '\0';
      return true;
    }
    line[offset++] = ch;
  }
  line[capacity - 1] = '\0';
  return false;
}

bool ParseUnsignedField(const char* line, const char* key, uint32_t* value) {
  const char* found = std::strstr(line, key);
  if (found == nullptr) return false;
  found += std::strlen(key);
  while (*found != '\0' && (*found == ' ' || *found == ':' || *found == '"')) {
    ++found;
  }
  if (!std::isdigit(static_cast<unsigned char>(*found))) return false;
  uint32_t result = 0;
  while (std::isdigit(static_cast<unsigned char>(*found))) {
    result = result * 10 + static_cast<uint32_t>(*found - '0');
    ++found;
  }
  *value = result;
  return true;
}

bool ParseInputHeader(const char* line, InputHeader* header) {
  if (std::strncmp(line, "VLM_MICRO_INPUT ", 16) != 0) return false;
  header->has_prompt_id = ParseUnsignedField(line, "\"prompt_id\"", &header->prompt_id);
  return ParseUnsignedField(line, "\"dataset_index\"", &header->dataset_index) &&
         ParseUnsignedField(line, "\"image_index\"", &header->image_index) &&
         ParseUnsignedField(line, "\"bytes\"", &header->bytes);
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

int TensorElementCount(const TfLiteTensor* tensor) {
  int count = 1;
  for (int i = 0; i < tensor->dims->size; ++i) {
    count *= tensor->dims->data[i];
  }
  return count;
}

bool IsImageTensor(const TfLiteTensor* tensor) {
  return tensor != nullptr && tensor->dims != nullptr && tensor->dims->size == 4 &&
         tensor->dims->data[0] == 1 && tensor->dims->data[3] == 3;
}

bool IsPromptTensor(const TfLiteTensor* tensor) {
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

void PrintTensorValues(const TfLiteTensor* tensor, int limit) {
  const int count = TensorElementCount(tensor);
  const int printed = count < limit ? count : limit;
  printf("[");
  for (int i = 0; i < printed; ++i) {
    if (i > 0) printf(",");
    switch (tensor->type) {
      case kTfLiteFloat32:
        printf("%.8g", tensor->data.f[i]);
        break;
      case kTfLiteInt32:
        printf("%ld", static_cast<long>(tensor->data.i32[i]));
        break;
      case kTfLiteUInt8:
        printf("%u", static_cast<unsigned int>(tensor->data.uint8[i]));
        break;
      case kTfLiteInt8:
        printf("%d", static_cast<int>(tensor->data.int8[i]));
        break;
      default:
        printf("0");
        break;
    }
  }
  printf("]");
}

void PrintOutputsJson(tflite::MicroInterpreter* interpreter) {
  printf("\"outputs\":[");
  for (size_t i = 0; i < interpreter->outputs().size(); ++i) {
    const int tensor_index = interpreter->outputs().Get(i);
    const TfLiteTensor* tensor = interpreter->output_tensor(i);
    if (i > 0) printf(",");
    printf(
        "{\"index\":%d,\"type\":\"%s\",\"bytes\":%u,\"scale\":%.9g,"
        "\"zero_point\":%ld,\"shape\":",
        tensor_index, TensorTypeName(tensor->type),
        static_cast<unsigned int>(tensor->bytes), tensor->params.scale,
        static_cast<long>(tensor->params.zero_point));
    PrintDims(tensor);
    printf(",\"values\":");
    PrintTensorValues(tensor, kMaxOutputValuesPerTensor);
    printf("}");
  }
  printf("]");
}

void PrintRecordedAllocationJson(
    const tflite::RecordingMicroAllocator* allocator,
    tflite::RecordedAllocationType type) {
  const tflite::RecordedAllocation allocation =
      allocator->GetRecordedAllocation(type);
  printf("{\"requested_bytes\":%u,\"used_bytes\":%u,\"count\":%u}",
         static_cast<unsigned int>(allocation.requested_bytes),
         static_cast<unsigned int>(allocation.used_bytes),
         static_cast<unsigned int>(allocation.count));
}

void PrintTensorSummaryJson(tflite::MicroInterpreter* interpreter,
                            bool inputs) {
  const size_t count = inputs ? interpreter->inputs().size()
                              : interpreter->outputs().size();
  printf("[");
  for (size_t i = 0; i < count; ++i) {
    const int tensor_index =
        inputs ? interpreter->inputs().Get(i) : interpreter->outputs().Get(i);
    const TfLiteTensor* tensor =
        inputs ? interpreter->input_tensor(i) : interpreter->output_tensor(i);
    if (i > 0) printf(",");
    printf(
        "{\"index\":%d,\"type\":\"%s\",\"bytes\":%u,\"scale\":%.9g,"
        "\"zero_point\":%ld,\"shape\":",
        tensor_index, TensorTypeName(tensor->type),
        static_cast<unsigned int>(tensor->bytes), tensor->params.scale,
        static_cast<long>(tensor->params.zero_point));
    PrintDims(tensor);
    printf("}");
  }
  printf("]");
}

void PrintReadyJson(tflite::MicroInterpreter* interpreter,
                    const tflite::RecordingMicroAllocator* allocator) {
  const int image_input_index = FindInputIndex(interpreter, IsImageTensor);
  const int prompt_input_index = FindInputIndex(interpreter, IsPromptTensor);
  const TfLiteTensor* input =
      interpreter->input_tensor(image_input_index >= 0 ? image_input_index : 0);
  const TfLiteTensor* output = interpreter->output_tensor(0);
  const auto* simple_allocator = allocator->GetSimpleMemoryAllocator();
  printf(
      "VLM_MICRO_READY {\"model_path\":\"%s\",\"tensor_arena_bytes\":%d,"
      "\"arena_used_bytes\":%u,\"arena_recorded_used_bytes\":%u,"
      "\"arena_recorded_requested_bytes\":%u,\"arena_recorded_alloc_count\":%u,"
      "\"input_count\":%u,\"output_count\":%u,\"image_input_index\":%d,"
      "\"prompt_input_index\":%d,\"prompt_lookup_count\":%d,"
      "\"prompt_lookup_dim\":%d,\"input\":{\"bytes\":%u,"
      "\"type\":\"%s\",\"scale\":%.9g,\"zero_point\":%ld,\"shape\":",
      kModelPath, kTensorArenaSize,
      static_cast<unsigned int>(interpreter->arena_used_bytes()),
      static_cast<unsigned int>(simple_allocator->GetUsedBytes()),
      static_cast<unsigned int>(simple_allocator->GetRequestedBytes()),
      static_cast<unsigned int>(simple_allocator->GetAllocatedCount()),
      static_cast<unsigned int>(interpreter->inputs().size()),
      static_cast<unsigned int>(interpreter->outputs().size()),
      image_input_index, prompt_input_index, TALLYQA_PROMPT_EMBEDDING_COUNT,
      TALLYQA_PROMPT_EMBEDDING_DIM,
      static_cast<unsigned int>(input->bytes), TensorTypeName(input->type),
      input->params.scale, static_cast<long>(input->params.zero_point));
  PrintDims(input);
  printf("},\"output\":{\"bytes\":%u,\"type\":\"%s\",\"scale\":%.9g,"
         "\"zero_point\":%ld,\"shape\":",
         static_cast<unsigned int>(output->bytes), TensorTypeName(output->type),
         output->params.scale, static_cast<long>(output->params.zero_point));
  PrintDims(output);
  printf("},\"inputs\":");
  PrintTensorSummaryJson(interpreter, true);
  printf(",\"outputs\":");
  PrintTensorSummaryJson(interpreter, false);
  printf("}\r\n");
}

void PrintResultJson(const InputHeader& header, uint64_t receive_us,
                     uint64_t image_copy_us, uint64_t prompt_copy_us,
                     uint64_t invoke_us,
                     tflite::MicroInterpreter* interpreter) {
  printf(
      "VLM_MICRO_RESULT {\"dataset_index\":%lu,\"image_index\":%lu,"
      "\"prompt_id\":%lu,\"input_bytes\":%lu,\"receive_us\":%lu,"
      "\"copy_us\":%lu,\"image_copy_us\":%lu,\"prompt_copy_us\":%lu,"
      "\"invoke_us\":%lu,",
      static_cast<unsigned long>(header.dataset_index),
      static_cast<unsigned long>(header.image_index),
      static_cast<unsigned long>(header.prompt_id),
      static_cast<unsigned long>(header.bytes),
      static_cast<unsigned long>(receive_us),
      static_cast<unsigned long>(image_copy_us + prompt_copy_us),
      static_cast<unsigned long>(image_copy_us),
      static_cast<unsigned long>(prompt_copy_us),
      static_cast<unsigned long>(invoke_us));
  PrintOutputsJson(interpreter);
  printf("}\r\n");
}

void PrintSelfTestResultJson(uint32_t run_id, int iteration, bool warmup,
                             uint32_t seed, uint32_t prompt_id,
                             uint64_t fill_us, uint64_t prompt_copy_us,
                             uint64_t invoke_us,
                             tflite::MicroInterpreter* interpreter) {
  printf(
      "VLM_MICRO_SELFTEST_RESULT {\"run_id\":%lu,\"iteration\":%d,"
      "\"warmup\":%s,\"seed\":%lu,\"prompt_id\":%lu,\"fill_us\":%lu,"
      "\"prompt_copy_us\":%lu,\"copy_us\":%lu,\"invoke_us\":%lu,",
      static_cast<unsigned long>(run_id), iteration, warmup ? "true" : "false",
      static_cast<unsigned long>(seed), static_cast<unsigned long>(prompt_id),
      static_cast<unsigned long>(fill_us),
      static_cast<unsigned long>(prompt_copy_us),
      static_cast<unsigned long>(fill_us + prompt_copy_us),
      static_cast<unsigned long>(invoke_us));
  PrintOutputsJson(interpreter);
  printf("}\r\n");
}

void PrintSelfTestSummaryJson(uint32_t run_id, int measured_count,
                              uint64_t min_invoke_us,
                              uint64_t max_invoke_us,
                              uint64_t total_invoke_us,
                              uint64_t total_copy_us) {
  const uint64_t avg_invoke_us =
      measured_count > 0 ? total_invoke_us / measured_count : 0;
  const uint64_t avg_copy_us = measured_count > 0 ? total_copy_us / measured_count : 0;
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

    const int total_iterations =
        kSelfTestWarmupIterations + kSelfTestMeasuredIterations;
    for (int iteration = 0; iteration < total_iterations; ++iteration) {
      const bool warmup = iteration < kSelfTestWarmupIterations;
      uint32_t state = run_seed + static_cast<uint32_t>(iteration) * 2654435761u;
      const uint32_t prompt_id =
          TALLYQA_PROMPT_EMBEDDING_COUNT > 0
              ? (NextRandom(&state) % TALLYQA_PROMPT_EMBEDDING_COUNT)
              : 0;

      const uint64_t fill_start_us = TimerMicros();
      FillTensorDeterministic(image_input, &state);
      const uint64_t fill_us = TimerMicros() - fill_start_us;

      uint64_t prompt_copy_us = 0;
      if (prompt_input != nullptr) {
        const uint64_t prompt_copy_start_us = TimerMicros();
        std::memcpy(prompt_input->data.raw,
                    kTallyQAPromptEmbeddingTable[prompt_id],
                    TALLYQA_PROMPT_EMBEDDING_DIM);
        prompt_copy_us = TimerMicros() - prompt_copy_start_us;
      }

      vTaskDelay(pdMS_TO_TICKS(10));
      const uint64_t invoke_start_us = TimerMicros();
      if (interpreter->Invoke() != kTfLiteOk) {
        printf(
            "VLM_MICRO_ERROR {\"run_id\":%lu,\"iteration\":%d,"
            "\"error\":\"invoke\"}\r\n",
            static_cast<unsigned long>(run_id), iteration);
        continue;
      }
      const uint64_t invoke_us = TimerMicros() - invoke_start_us;
      PrintSelfTestResultJson(run_id, iteration, warmup, run_seed, prompt_id,
                              fill_us, prompt_copy_us, invoke_us, interpreter);

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

[[noreturn]] void Main() {
  printf("VLM Micro TallyQA benchmark serial app starting.\r\n");
  LedSet(Led::kStatus, true);

  std::vector<uint8_t> model;
  if (!LfsReadFile(kModelPath, &model)) {
    printf("VLM_MICRO_ERROR {\"error\":\"model_load\",\"path\":\"%s\"}\r\n",
           kModelPath);
    vTaskSuspend(nullptr);
  }

  auto tpu_context = EdgeTpuManager::GetSingleton()->OpenDevice();
  if (!tpu_context) {
    PrintError("edgetpu_open");
    vTaskSuspend(nullptr);
  }

  tflite::MicroErrorReporter error_reporter;
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

  tflite::RecordingMicroAllocator* allocator =
      tflite::RecordingMicroAllocator::Create(tensor_arena, kTensorArenaSize,
                                              &error_reporter);
  if (allocator == nullptr) {
    PrintError("recording_allocator");
    vTaskSuspend(nullptr);
  }

  tflite::MicroInterpreter interpreter(tflite::GetModel(model.data()), resolver,
                                       allocator, &error_reporter);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    PrintError("allocate_tensors");
    vTaskSuspend(nullptr);
  }
  if (interpreter.inputs().size() != 1 && interpreter.inputs().size() != 2) {
    printf("VLM_MICRO_ERROR {\"error\":\"input_count\",\"count\":%u}\r\n",
           static_cast<unsigned int>(interpreter.inputs().size()));
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
      prompt_input_index >= 0 ? interpreter.input_tensor(prompt_input_index) : nullptr;
  PrintReadyJson(&interpreter, allocator);
  RunSelfTestLoop(&interpreter, image_input, prompt_input);
}

}  // namespace
}  // namespace coralmicro

extern "C" void app_main(void* param) {
  (void)param;
  coralmicro::Main();
}
