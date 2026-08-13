import unittest
from unittest.mock import patch, MagicMock

from trt_live_classifier import trt_classifier


class TestTRTClassifierMain(unittest.TestCase):
    @patch.object(trt_classifier, 'rclpy')
    @patch.object(trt_classifier, 'TRTWebcamClassifier')
    def test_main_cleans_up_when_spin_raises(self, mock_cls, mock_rclpy):
        node = MagicMock()
        mock_cls.return_value = node
        mock_rclpy.ok.return_value = True
        mock_rclpy.spin.side_effect = RuntimeError('spin failed')

        with self.assertRaises(RuntimeError):
            trt_classifier.main()

        mock_rclpy.init.assert_called_once()
        mock_cls.assert_called_once()
        mock_rclpy.spin.assert_called_once_with(node)
        node.destroy_node.assert_called_once()
        mock_rclpy.shutdown.assert_called_once()

    @patch.object(trt_classifier, 'rclpy')
    @patch.object(trt_classifier, 'TRTWebcamClassifier')
    def test_main_cleans_up_when_spin_succeeds(self, mock_cls, mock_rclpy):
        node = MagicMock()
        mock_cls.return_value = node
        mock_rclpy.ok.return_value = True

        trt_classifier.main()

        node.destroy_node.assert_called_once()
        mock_rclpy.shutdown.assert_called_once()


