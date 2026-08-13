import sys
from unittest.mock import MagicMock, patch, mock_open

import numpy as np
import pytest

# Stub heavy optional dependencies before importing the node module.
sys.modules.setdefault('torch', MagicMock())
sys.modules.setdefault('torch2trt', MagicMock())
sys.modules.setdefault('cv_bridge', MagicMock())
sys.modules['cv_bridge'].CvBridgeError = Exception
sys.modules.setdefault('live_detection', MagicMock())
sys.modules.setdefault('live_detection.mobilenetv1_ssd', MagicMock())
sys.modules.setdefault('live_detection.misc', MagicMock())

import rclpy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray


@pytest.fixture(scope='module')
def ros_context():
    rclpy.init()
    yield
    rclpy.shutdown()


def test_empty_detection_array_carries_input_header(ros_context):
    """When no objects are detected the published Detection2DArray must still
    carry the input image header.
    """
    mock_boxes = MagicMock()
    mock_boxes.size = MagicMock(return_value=0)
    mock_labels = MagicMock()
    mock_labels.size = MagicMock(return_value=0)

    mock_predictor = MagicMock()
    mock_predictor.predict.return_value = (mock_boxes, mock_labels, [])

    with patch('trt_live_detector.trt_detection_helper.create_mobilenetv1_ssd_predictor',
               return_value=mock_predictor), \
         patch('trt_live_detector.trt_detection_helper.torch.load', return_value={}), \
         patch('trt_live_detector.trt_detection_helper.os.path.isfile', return_value=True), \
         patch('trt_live_detector.trt_detection_helper.Timer',
               return_value=MagicMock(start=MagicMock(), end=MagicMock(return_value=0.0))), \
         patch('trt_live_detector.trt_detection_helper.cv2.imshow'), \
         patch('trt_live_detector.trt_detection_helper.cv2.waitKey'), \
         patch('trt_live_detector.trt_detection_helper.cv2.cvtColor',
               side_effect=lambda img, code: img), \
         patch('builtins.open', mock_open(read_data='background\nperson\n')):

        from trt_live_detector.trt_detection_helper import TRTDetectionNode

        with patch('rclpy.node.Node.create_publisher') as mock_create_publisher:
            mock_pub = MagicMock()
            mock_create_publisher.return_value = mock_pub

            node = TRTDetectionNode()
            node.bridge.imgmsg_to_cv2.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

            msg = Image()
            msg.header.frame_id = 'camera'
            msg.header.stamp.sec = 42

            node.listener_callback(msg)

            # Locate the Detection2DArray publish call.
            detection_array = None
            for call in mock_pub.publish.call_args_list:
                published = call[0][0]
                if isinstance(published, Detection2DArray):
                    detection_array = published
                    break

            assert detection_array is not None
            assert detection_array.header.frame_id == 'camera'
            assert detection_array.header.stamp.sec == 42

            node.destroy_node()
