import numpy as np
from sklearn.linear_model import LogisticRegression

from load import PARAMETERS
from live.forecast import forecast_from_windows
from preprocess import N_FEATURES, Standardizer, summarize_instance


def tiny_model():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, N_FEATURES))
    y = (X[:, 0] > 0).astype(int)
    scaler = Standardizer().fit(X)
    model = LogisticRegression(max_iter=500).fit(scaler.transform(X), y)
    return scaler, model


def test_forecast_from_windows_basic():
    scaler, model = tiny_model()
    windows = [
        {"harpnum": 1, "noaa_ars": "100", "features": np.ones((30, len(PARAMETERS)))},
        {"harpnum": 2, "noaa_ars": "200", "features": np.full((30, len(PARAMETERS)), 2.0)},
    ]
    full_disk, rows = forecast_from_windows(windows, scaler, model)
    assert 0.005 <= full_disk <= 0.995
    assert len(rows) == 2
    assert rows[0]["prob"] >= rows[1]["prob"]


def test_forecast_skips_too_new_regions():
    scaler, model = tiny_model()
    windows = [{"harpnum": 9, "noaa_ars": "x", "features": np.ones((3, len(PARAMETERS)))}]
    full_disk, rows = forecast_from_windows(windows, scaler, model)
    assert rows == []
    assert full_disk <= 0.01


def test_forecast_empty():
    scaler, model = tiny_model()
    full_disk, rows = forecast_from_windows([], scaler, model)
    assert rows == []
    assert full_disk <= 0.01

