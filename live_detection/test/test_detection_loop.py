# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.

import os

import numpy as np
import pytest
import rclpy
import torch
from sensor_msgs.msg import Image
from unittest.mock import MagicMock

from live_detection import live_detection_helper as helper


@pytest.fixture(scope='function', autouse=True)
def ros_context():
    rclpy.init(args=[])
    yield
    rclpy.shutdown()


def test_listener_callback_converts_detection_tensors(monkeypatch, tmp_path):
    monkeypatch.setenv('HOME', str(tmp_path))
    models_dir = tmp_path / 'ros2_models'
    models_dir.mkdir()
    (models_dir / 'voc-model-labels.txt').write_text('''background
person
car
''')
    (models_dir / 'mobilenet-v1-ssd-mp-0_675.pth').write_text('')

    mock_net = MagicMock()
    mock_predictor = MagicMock()
    boxes = torch.tensor([[10.0, 20.0, 30.0, 40.0], [5.0, 5.0, 15.0, 15.0]])
    labels = torch.tensor([2, 1])
    probs = torch.tensor([0.9, 0.75])
    mock_predictor.predict.return_value = (boxes, labels, probs)

    monkeypatch.setattr(helper, 'create_mobilenetv1_ssd', lambda *a, **k: mock_net)
    monkeypatch.setattr(helper, 'create_mobilenetv1_ssd_predictor', lambda *a, **k: mock_predictor)

    # Avoid GUI windows in headless test environments; keep cv2 drawing real
    # so the original code fails when it passes tensors instead of ints.
    monkeypatch.setattr(helper.cv2, 'imshow', lambda *a, **k: None)
    monkeypatch.setattr(helper.cv2, 'waitKey', lambda *a, **k: -1)

    node = helper.DetectionNode()
    node.bridge = MagicMock(imgmsg_to_cv2=lambda *a, **k: np.zeros((10, 10, 3), dtype=np.uint8))
    node.result_publisher.publish = lambda msg: None

    published = []
    node.detection_publisher.publish = lambda msg: published.append(msg)

    msg = Image()
    msg.header.frame_id = 'camera_frame'
    msg.height = 10
    msg.width = 10
    msg.encoding = 'bgr8'
    msg.step = 30
    msg.data = np.zeros(300, dtype=np.uint8).tolist()

    node.listener_callback(msg)

    assert len(published) == 1
    arr = published[0]
    assert len(arr.detections) == 2
