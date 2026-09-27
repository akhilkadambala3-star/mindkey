from ml.evidence import (
    classify_deviation,
    interpret_feature_deviation,
)


def test_classify_deviation_boundaries():
    assert classify_deviation(0.5) == "normal"
    assert classify_deviation(1.0) == "mild_deviation"
    assert classify_deviation(2.0) == "moderate_deviation"
    assert classify_deviation(3.0) == "strong_deviation"


def test_positive_deviation():
    result = interpret_feature_deviation(2.5, 2.5)

    assert result["deviation_band"] == "moderate_deviation"
    assert result["direction"] == "above_baseline"


def test_negative_deviation():
    result = interpret_feature_deviation(-5.0, 5.0)

    assert result["deviation_band"] == "strong_deviation"
    assert result["direction"] == "below_baseline"


def test_zero_deviation():
    result = interpret_feature_deviation(0.0, 0.0)

    assert result["deviation_band"] == "normal"
    assert result["direction"] == "at_baseline"