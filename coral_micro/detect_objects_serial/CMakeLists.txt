# Copyright 2026
#
# Repo-owned Coral Micro app scaffold. Stage this directory into the Coral SDK
# with scripts/coral_micro_stage_detection_app.py before building.

add_executable_m7(vlm_micro_detect_objects_serial
    detect_objects_serial.cc
    DATA
    ${PROJECT_SOURCE_DIR}/models/tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite
)

target_link_libraries(vlm_micro_detect_objects_serial
    libs_base-m7_freertos
)
