import numpy as np
from sklearn.linear_model import LogisticRegression

from load import PARAMETERS
from live import forecast
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
    assert 0.001 <= full_disk <= 0.995
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


def stub_bundle(monkeypatch):
    scaler, model = tiny_model()
    monkeypatch.setattr(forecast.joblib, "load",
                        lambda path: {"scaler": scaler, "model": model, "trained_utc": "2026-01-01"})


def test_make_forecast_returns_none_when_jsoc_has_no_data(tmp_path, monkeypatch):
    stub_bundle(monkeypatch)
    monkeypatch.setattr(forecast, "fetch_current_windows", lambda *args, **kwargs: [])
    log_path = tmp_path / "forecast_log.jsonl"
    assert forecast.make_forecast(log_path=str(log_path)) is None
    assert not log_path.exists()


def test_make_forecast_returns_none_when_every_region_is_too_short(tmp_path, monkeypatch):
    stub_bundle(monkeypatch)
    short = [{"harpnum": 1, "noaa_ars": "100", "features": np.ones((4, len(PARAMETERS)))}]
    monkeypatch.setattr(forecast, "fetch_current_windows", lambda *args, **kwargs: short)
    log_path = tmp_path / "forecast_log.jsonl"
    assert forecast.make_forecast(log_path=str(log_path)) is None
    assert not log_path.exists()


def test_make_forecast_logs_when_data_is_usable(tmp_path, monkeypatch):
    stub_bundle(monkeypatch)
    usable = [{"harpnum": 1, "noaa_ars": "100", "features": np.ones((30, len(PARAMETERS)))}]
    monkeypatch.setattr(forecast, "fetch_current_windows", lambda *args, **kwargs: usable)
    monkeypatch.setattr(forecast, "fetch_forecast", lambda *args, **kwargs: "")
    log_path = tmp_path / "forecast_log.jsonl"
    record = forecast.make_forecast(log_path=str(log_path))
    assert record["n_regions"] == 1
    assert len(log_path.read_text().strip().splitlines()) == 1
