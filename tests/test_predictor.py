import torch
from unittest.mock import Mock, patch

from trt_live_detector.predictor import Predictor


def _make_predictor(filter_threshold):
    net = Mock()
    net.to.return_value = net
    net.eval.return_value = net
    net.forward.return_value = (
        torch.tensor([[[0.0, 0.005, 0.1]]]),  # 1 image, 1 box, 3 classes
        torch.tensor([[[0.1, 0.1, 0.2, 0.2]]]),  # 1 image, 1 box, 4 coords
    )

    predictor = Predictor(
        net,
        size=300,
        filter_threshold=filter_threshold,
        device=torch.device("cpu"),
    )
    predictor.transform = Mock(return_value=torch.zeros(3, 300, 300))
    return predictor


def test_zero_prob_threshold_is_not_replaced_by_filter_threshold():
    predictor = _make_predictor(filter_threshold=0.5)

    with patch("trt_live_detector.predictor.box_utils.nms") as mock_nms:
        mock_nms.return_value = torch.tensor([[0.1, 0.1, 0.2, 0.2, 0.005]])
        predictor.predict(torch.zeros(100, 100, 3), prob_threshold=0.0)

        assert mock_nms.called
        for call in mock_nms.call_args_list:
            args, kwargs = call
            assert kwargs["score_threshold"] == 0.0


def test_none_prob_threshold_uses_filter_threshold():
    predictor = _make_predictor(filter_threshold=0.5)

    with patch("trt_live_detector.predictor.box_utils.nms") as mock_nms:
        mock_nms.return_value = torch.tensor([[0.1, 0.1, 0.2, 0.2, 0.1]])
        predictor.predict(torch.zeros(100, 100, 3), prob_threshold=None)

        assert mock_nms.called
        for call in mock_nms.call_args_list:
            args, kwargs = call
            assert kwargs["score_threshold"] == 0.5
