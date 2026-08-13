import unittest
from unittest.mock import MagicMock, patch

from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridgeError

from live_classifier.live_classifier_helper import WebcamClassifier


class FakeHelper:
    def __init__(self):
        self.classify_image = MagicMock(return_value=('cat', 99.0))
        self.classification_publisher = MagicMock()
        self.get_logger = MagicMock()
        self.bridge = MagicMock()
        self.bridge.imgmsg_to_cv2.side_effect = CvBridgeError('conversion failed')


class TestListenerCallback(unittest.TestCase):
    @patch('live_classifier.live_classifier_helper.cv2')
    def test_returns_on_cvbridge_error(self, mock_cv2):
        fake = FakeHelper()
        msg = Image()
        msg.header = Header()
        msg.height = 2
        msg.width = 2
        msg.data = [0] * 12

        # The callback should not raise, and should skip the imshow/waitKey
        # calls when CvBridgeError is raised.
        WebcamClassifier.listener_callback(fake, msg)

        fake.get_logger().info.assert_called_once()
        fake.classification_publisher.publish.assert_called_once()
        fake.bridge.imgmsg_to_cv2.assert_called_once_with(msg, 'bgr8')
        mock_cv2.imshow.assert_not_called()
        mock_cv2.waitKey.assert_not_called()

