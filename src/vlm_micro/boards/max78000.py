"""MAX78000 deployment notes.

The concrete implementation should shell out to the ADI tools rather than
copying them into this repository.
"""

REQUIRED_STAGES = (
    "torch_checkpoint",
    "ai8x_qat_or_quantized_checkpoint",
    "network_yaml",
    "ai8xize_generated_c",
    "msdk_flash",
    "board_profile",
)
