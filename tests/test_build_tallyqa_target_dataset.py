from __future__ import annotations

import numpy as np
from PIL import Image

from scripts.build_tallyqa_target_dataset import IMAGENET_MEAN_UINT8_RGB, image_tensor


IMAGENET_MEAN_CHW = np.asarray(IMAGENET_MEAN_UINT8_RGB, dtype=np.uint8).reshape(3, 1, 1)


def test_tallyqa_image_tensor_letterboxes_wide_image() -> None:
    image = Image.new("RGB", (40, 20), (10, 20, 30))

    tensor, metadata = image_tensor(image, image_size=16)

    assert tensor.shape == (3, 16, 16)
    assert metadata["original_size"] == [40, 20]
    assert metadata["resized_size"] == [16, 8]
    assert metadata["padding_ltrb"] == [0, 4, 0, 4]
    assert metadata["padding_value_rgb"] == IMAGENET_MEAN_UINT8_RGB
    assert np.all(tensor[:, :4, :] == IMAGENET_MEAN_CHW)
    assert np.all(tensor[:, 12:, :] == IMAGENET_MEAN_CHW)
    assert np.any(tensor[:, 4:12, :] != 0)


def test_tallyqa_image_tensor_letterboxes_tall_image() -> None:
    image = Image.new("RGB", (20, 40), (10, 20, 30))

    tensor, metadata = image_tensor(image, image_size=16)

    assert tensor.shape == (3, 16, 16)
    assert metadata["original_size"] == [20, 40]
    assert metadata["resized_size"] == [8, 16]
    assert metadata["padding_ltrb"] == [4, 0, 4, 0]
    assert metadata["padding_value_rgb"] == IMAGENET_MEAN_UINT8_RGB
    assert np.all(tensor[:, :, :4] == IMAGENET_MEAN_CHW)
    assert np.all(tensor[:, :, 12:] == IMAGENET_MEAN_CHW)
    assert np.any(tensor[:, :, 4:12] != 0)
