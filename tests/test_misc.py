import unittest
from unittest.mock import patch
from live_detection.misc import load_checkpoint


class TestMisc(unittest.TestCase):
    def test_load_checkpoint_defaults_to_cpu(self):
        with patch('live_detection.misc.torch.load') as mock_load:
            load_checkpoint('path.pth')
            mock_load.assert_called_once_with('path.pth', map_location='cpu')

    def test_load_checkpoint_allows_override(self):
        with patch('live_detection.misc.torch.load') as mock_load:
            load_checkpoint('path.pth', map_location='cuda:0')
            mock_load.assert_called_once_with('path.pth', map_location='cuda:0')


if __name__ == '__main__':
