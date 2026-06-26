import pandas as pd
import pytest

from load import PARAMETERS
from live.fetch import fetch_current_windows, group_windows


def make_df():
    rows = []
    for harp in [100, 200]:
        for step in range(3):
            row = {
                "HARPNUM": harp,
                "T_REC": "2026.06.19_%02d:00:00_TAI" % step,
                "NOAA_ARS": "14465",
                "QUALITY": 0,
            }
            for j, name in enumerate(PARAMETERS):
                row[name] = float(harp + step + j)
            rows.append(row)
    return pd.DataFrame(rows)


def test_group_windows_shapes():
    windows = group_windows(make_df())
    assert len(windows) == 2
    for window in windows:
        assert window["features"].shape == (3, len(PARAMETERS))
        assert "harpnum" in window
        assert "noaa_ars" in window


def test_live_fetch_smoke():
    try:
        windows = fetch_current_windows("2026.06.19_TAI", hours=12, cadence_min=60)
    except Exception:
        pytest.skip("no network or JSOC unavailable")
    assert len(windows) > 0
    assert windows[0]["features"].shape[1] == len(PARAMETERS)
