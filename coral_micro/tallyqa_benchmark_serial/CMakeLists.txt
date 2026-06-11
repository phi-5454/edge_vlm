# Copyright 2026
#
# Repo-owned Coral Micro TallyQA benchmark app scaffold. Stage this directory
# into the Coral SDK with scripts/coral_micro_stage_tallyqa_benchmark_app.py.

add_executable_m7(vlm_micro_tallyqa_benchmark_serial
    tallyqa_benchmark_serial.cc
    DATA
    ${PROJECT_SOURCE_DIR}/models/tallyqa_prompt_patch_mlp_edgetpu.tflite
)

target_link_libraries(vlm_micro_tallyqa_benchmark_serial
    libs_base-m7_freertos
)
