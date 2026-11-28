# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from uniface.detection import YOLOv8Face

TEST_IMAGE = Path(__file__).resolve().parent.parent / 'assets' / 'source' / 'detect_group.jpg'


@pytest.fixture(scope='module')
def yolov8_model():
    return YOLOv8Face(providers=['CPUExecutionProvider'])


def test_yolov8_returns_float32_arrays(yolov8_model):
    faces = yolov8_model.detect(cv2.imread(str(TEST_IMAGE)))

    assert len(faces) > 0
    for face in faces:
        assert face.bbox.dtype == np.float32
        assert face.landmarks.dtype == np.float32


def test_yolov8_very_negative_logits_do_not_overflow(yolov8_model):
    """Scores for very negative logits are 0 without a NumPy overflow warning."""
    outputs = [np.full((1, 80, size, size), -1000.0, dtype=np.float32) for size in (80, 40, 20)]

    with np.errstate(over='raise'):
        detections, _ = yolov8_model.postprocess(outputs, 1.0, (0, 0), (640, 640))

    assert len(detections) == 0
