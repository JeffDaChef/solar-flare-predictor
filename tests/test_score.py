from datetime import date

import pytest

from live.score import (
    daily_peaks,
    fetch_goes_xray,
    flare_class,
    major_flare_occurred,
    parse_records,
    peak_flux_on,
)

SAMPLE = [
    {"time_tag": "2024-06-20T01:00:00Z", "energy": "0.1-0.8nm", "flux": 2e-6},
    {"time_tag": "2024-06-20T02:00:00Z", "energy": "0.1-0.8nm", "flux": 3e-5},
    {"time_tag": "2024-06-20T02:00:00Z", "energy": "0.05-0.4nm", "flux": 9e-9},
    {"time_tag": "2024-06-21T05:00:00Z", "energy": "0.1-0.8nm", "flux": 4e-6},
]


def test_flare_class_thresholds():
    assert flare_class(2e-4) == "X"
    assert flare_class(3e-5) == "M"
    assert flare_class(5e-6) == "C"
    assert flare_class(5e-7) == "B"
    assert flare_class(1e-9) == "A"


def test_parse_keeps_only_long_band():
    records = parse_records(SAMPLE)
    assert len(records) == 3


def test_peak_and_major_detection():
    records = parse_records(SAMPLE)
    assert peak_flux_on(records, date(2024, 6, 20)) == pytest.approx(3e-5)
    assert major_flare_occurred(records, date(2024, 6, 20)) is True
    assert major_flare_occurred(records, date(2024, 6, 21)) is False


def test_daily_peaks_sorted():
    peaks = daily_peaks(parse_records(SAMPLE))
    assert [day for day, _ in peaks] == [date(2024, 6, 20), date(2024, 6, 21)]


def test_live_fetch_returns_records():
    try:
        records = fetch_goes_xray()
    except Exception:
        pytest.skip("no network for live NOAA fetch")
    assert len(records) > 0
    assert all(flux > 0 for _, flux in records)
