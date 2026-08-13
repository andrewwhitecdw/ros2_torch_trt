import unittest
from unittest.mock import MagicMock, patch

import rclpy

from cv_bridge import CvBridgeError
from trt_live_detector.trt_detection_helper import TRTDetectionNode


class TestTRTDetectionHelper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init(args=[])

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def test_listener_callback_returns_on_cvbridge_error(self):
        """If imgmsg_to_cv2 raises CvBridgeError the callback must return early."""
        with patch('trt_live_detector.trt_detection_helper.create_mobilenetv1_ssd'), \
             patch('trt_live_detector.trt_detection_helper.create_mobilenetv1_ssd_predictor') as mock_predictor_factory, \
             patch('trt_live_detector.trt_detection_helper.TRTModule') as mock_trt_module, \
             patch('trt_live_detector.trt_detection_helper.torch.load'), \
             patch('trt_live_detector.trt_detection_helper.os.path.isfile', return_value=True), \
             patch('trt_live_detector.trt_detection_helper.open', MagicMock()), \
             patch('trt_live_detector.trt_detection_helper.cv2'):
            mock_trt_module.return_value.load_state_dict = MagicMock()
            mock_predictor_factory.return_value.predict = MagicMock(
                return_value=(MagicMock(), MagicMock(), MagicMock())
            )

            node = TRTDetectionNode()
            node.bridge = MagicMock()
            node.bridge.imgmsg_to_cv2.side_effect = CvBridgeError("conversion failed")
            logger = MagicMock()
            node.get_logger = MagicMock(return_value=logger)

            fake_msg = MagicMock()
            fake_msg.header.frame_id = "camera"

            node.listener_callback(fake_msg)

            node.bridge.imgmsg_to_cv2.assert_called_once_with(fake_msg, "bgr8")
            logger.error.assert_called_once()
