from datetime import datetime, timedelta, timezone

import numpy as np
from sklearn.linear_model import LogisticRegression

from load import LIVE_PARAMETERS
from live import forecast
from live.forecast import forecast_from_windows
from preprocess import N_LIVE_FEATURES, Standardizer, summarize_instance

ISSUED = datetime(2026, 6, 19, 12, 0, tzinfo=timezone.utc)


def make_window(harpnum, value, steps=30, end=ISSUED):
    times = np.array([np.datetime64((end - timedelta(minutes=12 * (steps - i))).replace(tzinfo=None))
                      for i in range(steps)])
    return {
        "harpnum": harpnum,
        "noaa_ars": str(harpnum),
        "times": times,
        "features": np.full((steps, len(LIVE_PARAMETERS)), value),
    }


def tiny_model():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, N_LIVE_FEATURES))
    y = (X[:, 0] > 0).astype(int)
    scaler = Standardizer().fit(X)
    model = LogisticRegression(max_iter=500).fit(scaler.transform(X), y)
    return scaler, model


def test_forecast_from_windows_basic():
    scaler, model = tiny_model()
    windows = [make_window(1, 1.0), make_window(2, 2.0)]
    full_disk, rows = forecast_from_windows(windows, scaler, model, ISSUED)
    assert 0.001 <= full_disk <= 0.995
    assert len(rows) == 2
    assert rows[0]["prob"] >= rows[1]["prob"]
    assert set(rows[0]["features"]) == set(LIVE_PARAMETERS)


def test_forecast_takes_the_strongest_window_not_the_latest():
    scaler, model = tiny_model()
    steps = 120
    features = np.zeros((steps, len(LIVE_PARAMETERS)))
    features[:40] = 6.0
    times = np.array([np.datetime64((ISSUED - timedelta(minutes=12 * (steps - i))).replace(tzinfo=None))
                      for i in range(steps)])
    window = {"harpnum": 7, "noaa_ars": "7", "times": times, "features": features}
    _, rows = forecast_from_windows([window], scaler, model, ISSUED)
    latest = summarize_instance(features[-60:]).reshape(1, -1)
    latest_prob = float(model.predict_proba(scaler.transform(latest))[0, 1])
    assert rows[0]["prob"] >= latest_prob


def test_forecast_skips_too_new_regions():
    scaler, model = tiny_model()
    windows = [make_window(9, 1.0, steps=3)]
    full_disk, rows = forecast_from_windows(windows, scaler, model, ISSUED)
    assert rows == []
    assert full_disk <= 0.01


def test_forecast_empty():
    scaler, model = tiny_model()
    full_disk, rows = forecast_from_windows([], scaler, model, ISSUED)
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
    short = [make_window(1, 1.0, steps=4, end=datetime.now(timezone.utc))]
    monkeypatch.setattr(forecast, "fetch_current_windows", lambda *args, **kwargs: short)
    log_path = tmp_path / "forecast_log.jsonl"
    assert forecast.make_forecast(log_path=str(log_path)) is None
    assert not log_path.exists()


def test_make_forecast_logs_when_data_is_usable(tmp_path, monkeypatch):
    stub_bundle(monkeypatch)
    usable = [make_window(1, 1.0, end=datetime.now(timezone.utc))]
    monkeypatch.setattr(forecast, "fetch_current_windows", lambda *args, **kwargs: usable)
    monkeypatch.setattr(forecast, "fetch_forecast", lambda *args, **kwargs: "")
    log_path = tmp_path / "forecast_log.jsonl"
    record = forecast.make_forecast(log_path=str(log_path))
    assert record["n_regions"] == 1
    assert record["scoring"] == forecast.SCORING
    assert record["full_disk_prob"] == min(max(record["regions"][0]["prob"], 0.001), 0.995)
    assert len(log_path.read_text().strip().splitlines()) == 1
