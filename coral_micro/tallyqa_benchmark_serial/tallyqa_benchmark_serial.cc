// Host-driven TallyQA benchmark app for Coral Dev Board Micro.
//
// The host sends one framed image at a time over the USB serial console:
//   VLM_MICRO_INPUT {"dataset_index":123,"image_index":45,"bytes":150528}\n
//   <exactly bytes raw NHWC uint8 image bytes>
//
// The board copies the bytes into the model input tensor, invokes the model,
// and prints one newline-delimited JSON result with timing and output tensors.

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

STATIC_TENSOR_ARENA_IN_SDRAM(tensor_arena, kTensorArenaSize);

struct InputHeader {
  uint32_t dataset_index = 0;
  uint32_t image_index = 0;
  uint32_t bytes = 0;
};

void PrintReadyJson(tflite::MicroInterpreter* interpreter,
                    const tflite::RecordingMicroAllocator* allocator);

void PrintError(const char* error) {
  printf("VLM_MICRO_ERROR {\"error\":\"%s\"}\r\n", error);
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
  const TfLiteTensor* input = interpreter->input_tensor(0);
  const TfLiteTensor* output = interpreter->output_tensor(0);
  const auto* simple_allocator = allocator->GetSimpleMemoryAllocator();
  printf(
      "VLM_MICRO_READY {\"model_path\":\"%s\",\"tensor_arena_bytes\":%d,"
      "\"arena_used_bytes\":%u,\"arena_recorded_used_bytes\":%u,"
      "\"arena_recorded_requested_bytes\":%u,\"arena_recorded_alloc_count\":%u,"
      "\"input_count\":%u,\"output_count\":%u,\"input\":{\"bytes\":%u,"
      "\"type\":\"%s\",\"scale\":%.9g,\"zero_point\":%ld,\"shape\":",
      kModelPath, kTensorArenaSize,
      static_cast<unsigned int>(interpreter->arena_used_bytes()),
      static_cast<unsigned int>(simple_allocator->GetUsedBytes()),
      static_cast<unsigned int>(simple_allocator->GetRequestedBytes()),
      static_cast<unsigned int>(simple_allocator->GetAllocatedCount()),
      static_cast<unsigned int>(interpreter->inputs().size()),
      static_cast<unsigned int>(interpreter->outputs().size()),
      static_cast<unsigned int>(input->bytes), TensorTypeName(input->type),
      input->params.scale, static_cast<long>(input->params.zero_point));
  PrintDims(input);
  printf("},\"output\":{\"bytes\":%u,\"type\":\"%s\",\"scale\":%.9g,"
         "\"zero_point\":%ld,\"shape\":",
         static_cast<unsigned int>(output->bytes), TensorTypeName(output->type),
         output->params.scale, static_cast<long>(output->params.zero_point));
  PrintDims(output);
  printf("}}\r\n");
}

void PrintResultJson(const InputHeader& header, uint64_t receive_us,
                     uint64_t copy_us, uint64_t invoke_us,
                     tflite::MicroInterpreter* interpreter) {
  printf(
      "VLM_MICRO_RESULT {\"dataset_index\":%lu,\"image_index\":%lu,"
      "\"input_bytes\":%lu,\"receive_us\":%lu,\"copy_us\":%lu,"
      "\"invoke_us\":%lu,",
      static_cast<unsigned long>(header.dataset_index),
      static_cast<unsigned long>(header.image_index),
      static_cast<unsigned long>(header.bytes),
      static_cast<unsigned long>(receive_us),
      static_cast<unsigned long>(copy_us),
      static_cast<unsigned long>(invoke_us));
  PrintOutputsJson(interpreter);
  printf("}\r\n");
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
  if (interpreter.inputs().size() != 1) {
    printf("VLM_MICRO_ERROR {\"error\":\"input_count\",\"count\":%u}\r\n",
           static_cast<unsigned int>(interpreter.inputs().size()));
    vTaskSuspend(nullptr);
  }

  TfLiteTensor* input = interpreter.input_tensor(0);
  std::vector<uint8_t> image(input->bytes);
  PrintReadyJson(&interpreter, allocator);

  char line[kMaxLineBytes];
  while (true) {
    if (!ReadLine(line, sizeof(line), &interpreter, allocator)) {
      PrintError("line_too_long");
      continue;
    }
    InputHeader header;
    if (!ParseInputHeader(line, &header)) {
      printf("VLM_MICRO_ERROR {\"error\":\"bad_header\",\"line\":\"%s\"}\r\n",
             line);
      continue;
    }
    if (header.bytes != input->bytes) {
      printf(
          "VLM_MICRO_ERROR {\"dataset_index\":%lu,\"error\":\"input_bytes\","
          "\"expected\":%u,\"actual\":%lu}\r\n",
          static_cast<unsigned long>(header.dataset_index),
          static_cast<unsigned int>(input->bytes),
          static_cast<unsigned long>(header.bytes));
      continue;
    }

    printf("VLM_MICRO_RX_READY {\"dataset_index\":%lu,\"bytes\":%lu}\r\n",
           static_cast<unsigned long>(header.dataset_index),
           static_cast<unsigned long>(header.bytes));
    const uint64_t receive_start_us = TimerMicros();
    ReadExact(image.data(), image.size());
    const uint64_t receive_us = TimerMicros() - receive_start_us;

    const uint64_t copy_start_us = TimerMicros();
    std::memcpy(input->data.raw, image.data(), image.size());
    const uint64_t copy_us = TimerMicros() - copy_start_us;

    const uint64_t invoke_start_us = TimerMicros();
    if (interpreter.Invoke() != kTfLiteOk) {
      printf("VLM_MICRO_ERROR {\"dataset_index\":%lu,\"error\":\"invoke\"}\r\n",
             static_cast<unsigned long>(header.dataset_index));
      continue;
    }
    const uint64_t invoke_us = TimerMicros() - invoke_start_us;
    PrintResultJson(header, receive_us, copy_us, invoke_us, &interpreter);
  }
}

}  // namespace
}  // namespace coralmicro

extern "C" void app_main(void* param) {
  (void)param;
  coralmicro::Main();
}
