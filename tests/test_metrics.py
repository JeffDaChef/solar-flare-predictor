import pytest

from metrics import (
    accuracy,
    all_scores,
    confusion_counts,
    hss,
    sensitivity,
    specificity,
    tss,
)


def make_arrays(tp, fp, fn, tn):
    y_true = [1] * tp + [1] * fn + [0] * fp + [0] * tn
    y_pred = [1] * tp + [0] * fn + [1] * fp + [0] * tn
    return y_true, y_pred


def test_confusion_counts_match_inputs():
    y_true, y_pred = make_arrays(tp=8, fp=10, fn=2, tn=80)
    assert confusion_counts(y_true, y_pred) == (8, 10, 2, 80)


def test_perfect_prediction():
    y_true, y_pred = make_arrays(tp=10, fp=0, fn=0, tn=90)
    assert tss(y_true, y_pred) == pytest.approx(1.0)
    assert hss(y_true, y_pred) == pytest.approx(1.0)
    assert accuracy(y_true, y_pred) == pytest.approx(1.0)


def test_inverted_prediction_is_worst_tss():
    y_true, y_pred = make_arrays(tp=0, fp=90, fn=10, tn=0)
    assert tss(y_true, y_pred) == pytest.approx(-1.0)


def test_always_no_is_useless_despite_high_accuracy():
    y_true, y_pred = make_arrays(tp=0, fp=0, fn=2, tn=98)
    assert accuracy(y_true, y_pred) == pytest.approx(0.98)
    assert tss(y_true, y_pred) == pytest.approx(0.0)
    assert hss(y_true, y_pred) == pytest.approx(0.0)


def test_known_mixed_matrix():
    y_true, y_pred = make_arrays(tp=8, fp=10, fn=2, tn=80)
    assert sensitivity(y_true, y_pred) == pytest.approx(0.8)
    assert specificity(y_true, y_pred) == pytest.approx(80 / 90)
    assert tss(y_true, y_pred) == pytest.approx(0.8 - 10 / 90)
    assert hss(y_true, y_pred) == pytest.approx(1240 / 2440)


def test_balanced_accuracy_identity():
    y_true, y_pred = make_arrays(tp=7, fp=2, fn=3, tn=8)
    score = tss(y_true, y_pred)
    assert score == pytest.approx(0.5)
    assert accuracy(y_true, y_pred) == pytest.approx((score + 1) / 2)


def test_all_scores_returns_every_metric():
    y_true, y_pred = make_arrays(tp=8, fp=10, fn=2, tn=80)
    scores = all_scores(y_true, y_pred)
    assert set(scores) == {"tss", "hss", "sensitivity", "specificity", "accuracy"}
