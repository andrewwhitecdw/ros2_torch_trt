from trt_live_detector.misc import str2bool


def test_str2bool_string_inputs():
    assert str2bool("True") is True
    assert str2bool("false") is False
    assert str2bool("1") is True
    assert str2bool("0") is False


def test_str2bool_non_string_inputs():
    assert str2bool(True) is True
