import torch

from trt_live_detector.box_utils import soft_nms


def test_soft_nms_preserves_device_dtype_and_shape():
    box_scores = torch.tensor([
        [0.0, 0.0, 1.0, 1.0, 0.9],
        [0.0, 0.0, 0.9, 0.9, 0.8],
    ])
    result = soft_nms(box_scores, score_threshold=0.0)
    assert result.device == box_scores.device
    assert result.dtype == box_scores.dtype
    assert result.dim() == 2

    if torch.cuda.is_available():
        box_scores_gpu = box_scores.cuda()
        result_gpu = soft_nms(box_scores_gpu, score_threshold=0.0)
        assert result_gpu.device == box_scores_gpu.device
        assert result_gpu.dtype == box_scores_gpu.dtype
        assert result_gpu.dim() == 2


def test_soft_nms_empty_result_has_correct_shape():
    box_scores = torch.empty(0, 5)
    result = soft_nms(box_scores, score_threshold=0.0)
    assert result.shape == torch.Size([0, 5])
    assert result.device == box_scores.device
