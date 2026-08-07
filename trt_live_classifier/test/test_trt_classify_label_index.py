# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.

import numpy as np
import pytest
import torch

from trt_live_classifier.trt_classifier_helper import TRTWebcamClassifier


def test_trt_classify_image_returns_string_label(monkeypatch):
    # Regression test: classify_image must convert the tensor index to int.
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self: self)

    fake_self = type('Fake', (), {})()
    fake_self.labels = ['class_a', 'class_b', 'class_c']
    fake_self.classification_model_trt = lambda x: torch.tensor(
        [[0.1, 0.8, 0.1]], dtype=torch.float32
    )

    img = np.zeros((4, 4, 3), dtype=np.uint8)
    label, confidence = TRTWebcamClassifier.classify_image(fake_self, img)

    assert label == 'class_b'
