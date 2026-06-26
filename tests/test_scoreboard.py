from datetime import datetime

from live.scoreboard import grade_forecasts, summarize


def dt(text):
    return datetime.fromisoformat(text)


def test_grade_only_elapsed_windows():
    forecasts = [
        {"issued_utc": "2026-06-01T00:00:00+00:00", "full_disk_prob": 0.8},
        {"issued_utc": "2026-06-20T06:00:00+00:00", "full_disk_prob": 0.1},
    ]
    records = [(dt("2026-06-01T05:00:00+00:00"), 3e-5)]
    now = dt("2026-06-20T12:00:00+00:00")
    graded = grade_forecasts(forecasts, records, now=now)
    assert len(graded) == 1
    assert graded[0]["actual"] is True


def test_no_flare_in_window():
    forecasts = [{"issued_utc": "2026-06-01T00:00:00+00:00", "full_disk_prob": 0.4}]
    records = [(dt("2026-06-01T05:00:00+00:00"), 3e-6)]
    now = dt("2026-06-03T00:00:00+00:00")
    graded = grade_forecasts(forecasts, records, now=now)
    assert graded[0]["actual"] is False


def test_summarize_metrics():
    graded = [
        {"issued_utc": "a", "prob": 0.8, "actual": True},
        {"issued_utc": "b", "prob": 0.1, "actual": False},
        {"issued_utc": "c", "prob": 0.7, "actual": False},
    ]
    summary = summarize(graded, threshold=0.5)
    assert summary["n"] == 3
    assert summary["flares"] == 1
    assert 0.0 <= summary["model"]["brier"] <= 1.0
    assert "noaa" not in summary


def test_summarize_head_to_head_with_noaa():
    graded = [
        {"issued_utc": "a", "prob": 0.8, "noaa_prob": 0.4, "actual": True},
        {"issued_utc": "b", "prob": 0.1, "noaa_prob": 0.3, "actual": False},
    ]
    summary = summarize(graded, threshold=0.5)
    assert summary["noaa"]["n"] == 2
    assert summary["model_head_to_head"]["n"] == 2


def test_summarize_empty():
    assert summarize([])["n"] == 0
