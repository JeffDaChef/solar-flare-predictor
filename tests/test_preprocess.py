import os

import numpy as np
import pytest

from load import HISTORY, LIVE_PARAMETERS, PARAMETERS
from preprocess import (
    N_FEATURES,
    N_LIVE_FEATURES,
    SEQ_LEN,
    Standardizer,
    build_features,
    instance_to_sequence,
    live_columns,
    summarize_history,
    summarize_instance,
)

PARTITION3 = os.path.join("data", "partition3_instances.tar.gz")


def test_summarize_shape():
    features = np.ones((10, len(PARAMETERS)))
    assert summarize_instance(features).shape == (N_FEATURES,)


def test_summarize_known_values():
    p = len(PARAMETERS)
    features = np.zeros((3, p))
    features[:, 0] = [1.0, 2.0, 3.0]
    vec = summarize_instance(features)
    assert vec[0] == pytest.approx(2.0)
    assert vec[4 * p + 0] == pytest.approx(3.0)
    assert vec[5 * p + 0] == pytest.approx(1.0)


def test_summarize_handles_all_nan_and_empty():
    features = np.full((5, len(PARAMETERS)), np.nan)
    assert np.isnan(summarize_instance(features)).all()
    assert summarize_instance(np.zeros((0,))).shape == (N_FEATURES,)


def test_standardizer_imputes_nan_to_zero():
    X = np.array([[1.0, 10.0], [3.0, 30.0], [np.nan, 50.0]])
    scaler = Standardizer().fit(X)
    transformed = scaler.transform(X)
    assert transformed[2, 0] == pytest.approx(0.0)
    assert not np.isnan(transformed).any()


def test_instance_to_sequence_pads_and_truncates():
    p = len(PARAMETERS)
    short = np.ones((10, p))
    assert instance_to_sequence(short).shape == (SEQ_LEN, p)
    long = np.ones((90, p))
    assert instance_to_sequence(long).shape == (SEQ_LEN, p)
    assert instance_to_sequence(np.zeros((0,))).shape == (SEQ_LEN, p)


def test_instance_to_sequence_fills_missing():
    p = len(PARAMETERS)
    arr = np.ones((SEQ_LEN, p))
    arr[5, 0] = np.nan
    seq = instance_to_sequence(arr)
    assert not np.isnan(seq).any()


def test_summarize_history_peak_and_empty():
    hist = np.zeros((5, len(HISTORY)))
    hist[2, 1] = 3.0
    out = summarize_history(hist)
    assert out.shape == (len(HISTORY),)
    assert out[1] == 3.0
    assert summarize_history(np.empty((0, len(HISTORY)))).shape == (len(HISTORY),)


@pytest.mark.skipif(not os.path.exists(PARTITION3), reason="partition3 not downloaded")
def test_build_features_smoke():
    X, y, classes = build_features(PARTITION3, limit=100)
    assert X.shape == (100, N_FEATURES)
    assert set(np.unique(y)).issubset({0, 1})
    assert len(classes) == 100


@pytest.mark.skipif(not os.path.exists(PARTITION3), reason="partition3 not downloaded")
def test_build_features_with_history():
    X, _, _ = build_features(PARTITION3, limit=50, with_history=True)
    assert X.shape == (50, N_FEATURES + len(HISTORY))


def test_live_parameters_exclude_keywords_jsoc_does_not_serve():
    from load import NOT_SERVED_BY_JSOC
    assert len(LIVE_PARAMETERS) == 17
    assert not set(LIVE_PARAMETERS) & set(NOT_SERVED_BY_JSOC)
    assert LIVE_PARAMETERS == [p for p in PARAMETERS if p in LIVE_PARAMETERS]


def test_training_and_live_feature_order_match():
    rng = np.random.default_rng(7)
    raw = rng.normal(size=(30, len(PARAMETERS)))
    keep = [PARAMETERS.index(p) for p in LIVE_PARAMETERS]
    from_training = live_columns(summarize_instance(raw).reshape(1, -1))[0]
    from_live = summarize_instance(raw[:, keep])
    assert from_training.shape == (N_LIVE_FEATURES,)
    assert np.allclose(from_training, from_live, equal_nan=True)


def test_live_columns_selects_expected_width():
    rng = np.random.default_rng(3)
    assert live_columns(rng.normal(size=(5, N_FEATURES))).shape == (5, N_LIVE_FEATURES)
