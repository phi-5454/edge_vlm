"""Coral Micro deployment notes.

Keep this module free of TensorFlow imports until the export implementation is
added, so the common training environment does not require the Coral stack.
"""

REQUIRED_STAGES = (
    "torch_checkpoint",
    "keras_saved_model",
    "tflite_int8",
    "edgetpu_compiled_tflite",
    "coralmicro_flash",
    "board_profile",
)
