import os
import tempfile
import unittest
from unittest import mock

import torch

from trt_live_detector.misc import load_checkpoint


class TestLoadCheckpoint(unittest.TestCase):
    def test_load_checkpoint_uses_weights_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pth")
            torch.save({"epoch": 1, "best_score": 0.5}, path)

            with mock.patch("trt_live_detector.misc.torch.load") as mock_load:
                mock_load.return_value = {"epoch": 1, "best_score": 0.5}
                result = load_checkpoint(path)

            mock_load.assert_called_once_with(path, weights_only=True)
            self.assertEqual(result, {"epoch": 1, "best_score": 0.5})
