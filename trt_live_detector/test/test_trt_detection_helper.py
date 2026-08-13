import unittest
from unittest.mock import Mock

from sensor_msgs.msg import Image
from cv_bridge import CvBridgeError

from trt_live_detector.trt_detection_helper import TRTDetectionNode


class TestTRTDetectionHelper(unittest.TestCase):

    def test_listener_callback_returns_on_cv_bridge_error(self):
        node = TRTDetectionNode.__new__(TRTDetectionNode)
        node.bridge = Mock()
        node.bridge.imgmsg_to_cv2 = Mock(side_effect=CvBridgeError('conversion failed'))
        node.get_logger = Mock(return_value=Mock())
        node.predictor = Mock()
        node.timer = Mock()
        node.detection_publisher = Mock()
        node.result_publisher = Mock()

        node.listener_callback(Image())

        node.bridge.imgmsg_to_cv2.assert_called_once()
        node.predictor.predict.assert_not_called()


if __name__ == '__main__':
