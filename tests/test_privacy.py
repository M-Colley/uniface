# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

import cv2
import numpy as np
import pytest

from uniface.privacy import BlurFace, EllipticalBlur

METHODS = ['gaussian', 'median', 'pixelate', 'blackout', 'elliptical']
BBOX = [40, 40, 120, 140]


@pytest.fixture
def image():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(200, 200, 3), dtype=np.uint8)


@pytest.mark.parametrize('method', METHODS)
def test_inplace_modifies_the_input_image(image, method):
    """With inplace=True the caller's array is blurred and returned."""
    original = image.copy()

    result = BlurFace(method=method).blur_regions(image, [BBOX], inplace=True)

    assert result is image
    assert not np.array_equal(image, original)


@pytest.mark.parametrize('method', METHODS)
def test_default_leaves_the_input_image_untouched(image, method):
    """Without inplace the caller's array is kept and a blurred copy is returned."""
    original = image.copy()

    result = BlurFace(method=method).blur_regions(image, [BBOX])

    assert result is not image
    assert np.array_equal(image, original)
    assert not np.array_equal(result, original)


def _full_frame_elliptical_blur(image, bboxes, blur_strength=3.0, margin=20):
    """The original whole-image EllipticalBlur, kept as the reference for the faster version."""
    image = image.copy()
    h, w = image.shape[:2]
    for bbox in bboxes:
        x1, y1, x2, y2 = map(int, bbox)
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        axes_x = (x2 - x1) // 2 + margin
        axes_y = (y2 - y1) // 2 + margin
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(mask, (center_x, center_y), (axes_x, axes_y), 0, 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (51, 51), 0) / 255.0
        mask = mask[:, :, np.newaxis]
        kernel_size = max(3, int((min(axes_y, axes_x) * 2 / 7) * blur_strength)) | 1
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        image[:] = (blurred * mask + image * (1 - mask)).astype(np.uint8)
    return image


@pytest.mark.parametrize(
    ('bboxes', 'margin'),
    [
        ([[60, 40, 150, 160]], 20),  # face inside the image
        ([[-30, 10, 60, 120]], 20),  # cut by the left edge
        ([[250, 180, 360, 300]], 20),  # cut by the bottom-right corner
        ([[400, 400, 500, 500]], 20),  # fully outside the image
        ([[60, 40, 150, 160], [120, 60, 220, 190]], 20),  # two overlapping faces
        ([[100, 80, 200, 80]], 20),  # zero-height box
        ([[150, 20, 151, 180]], 0),  # 1 px wide box
    ],
)
def test_elliptical_blur_matches_full_frame_reference(bboxes, margin):
    """Blurring only the region around each face gives the same pixels as blurring the whole image."""
    image = np.random.default_rng(1).integers(0, 256, size=(240, 320, 3), dtype=np.uint8)

    result = EllipticalBlur(margin=margin)(image, bboxes)

    assert np.array_equal(result, _full_frame_elliptical_blur(image, bboxes, margin=margin))
