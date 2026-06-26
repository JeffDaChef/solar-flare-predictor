import pytest

from live.noaa import fetch_forecast, major_probability, parse_forecast

SAMPLE = """C. NOAA Radio Blackout Activity and Forecast

Radio Blackout Forecast for Jun 22-Jun 24 2026

              Jun 22        Jun 23        Jun 24
R1-R2           40%           40%           40%
R3 or greater    5%            5%            5%
"""


def test_parse_extracts_dates_and_probs():
    forecast = parse_forecast(SAMPLE)
    assert forecast["2026-06-22"]["m_class"] == pytest.approx(0.40)
    assert forecast["2026-06-22"]["x_class"] == pytest.approx(0.05)
    assert set(forecast) == {"2026-06-22", "2026-06-23", "2026-06-24"}


def test_major_probability_lookup():
    forecast = parse_forecast(SAMPLE)
    assert major_probability(forecast, "2026-06-23") == pytest.approx(0.40)
    assert major_probability(forecast, "2030-01-01") is None


def test_live_forecast_parses():
    try:
        forecast = parse_forecast(fetch_forecast())
    except Exception:
        pytest.skip("no network or NOAA unavailable")
    assert len(forecast) == 3
    for entry in forecast.values():
        assert 0.0 <= entry["m_class"] <= 1.0
