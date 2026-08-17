import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


def _find_helper_path():
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        candidate = parent / "live_classifier" / "live_classifier_helper.py"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find live_classifier_helper.py")


def _load_helper_module():
    helper_path = _find_helper_path()
    spec = importlib.util.spec_from_file_location(
        "live_classifier_helper_under_test", helper_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["live_classifier_helper_under_test"] = module

    with mock.patch.dict(sys.modules, {
        "rclpy": mock.MagicMock(),
        "rclpy.node": mock.MagicMock(Node=object),
        "sensor_msgs": mock.MagicMock(),
        "sensor_msgs.msg": mock.MagicMock(Image=object),
        "std_msgs": mock.MagicMock(),
        "std_msgs.msg": mock.MagicMock(Header=object),
        "vision_msgs": mock.MagicMock(),
        "vision_msgs.msg": mock.MagicMock(Classification2D=object, ObjectHypothesis=object),
        "torch": mock.MagicMock(),
        "torchvision": mock.MagicMock(models=mock.MagicMock(), transforms=mock.MagicMock()),
        "numpy": mock.MagicMock(),
        "cv2": mock.MagicMock(),
        "cv_bridge": mock.MagicMock(CvBridge=object, CvBridgeError=Exception),
    }):
        spec.loader.exec_module(module)

    return module


class CreateClassificationModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helper = _load_helper_module()

    def test_invalid_model_selection_raises_value_error(self):
        webcam_classifier = object.__new__(self.helper.WebcamClassifier)
        model_name = SimpleNamespace(value="invalid-model")
        with self.assertRaises(ValueError):
            webcam_classifier.create_classification_model(model_name)


if __name__ == "__main__":
    unittest.main()
