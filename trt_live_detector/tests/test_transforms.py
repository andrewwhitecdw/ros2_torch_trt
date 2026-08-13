import numpy as np
from unittest import mock

from trt_live_detector.transforms import RandomSampleCrop


def test_random_sample_crop_offset_uniform_has_zero_lower_bound():
    """RandomSampleCrop must sample left/top offsets with an explicit lower bound of 0."""
    image = np.zeros((100, 100, 3), dtype=np.float32)
    boxes = np.array([[10, 10, 90, 90]], dtype=np.float32)
    labels = np.array([1])

    with mock.patch('trt_live_detector.transforms.random') as m_random:
        m_random.choice.return_value = (0.1, None)
        m_random.randint.return_value = 0
        # w, h, left, top
        m_random.uniform.side_effect = [50.0, 50.0, 10.0, 10.0]

        RandomSampleCrop()(image, boxes, labels)

        assert m_random.uniform.call_args_list[2] == mock.call(0, 50.0)
