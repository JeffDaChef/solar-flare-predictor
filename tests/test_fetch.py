import socket
from urllib.error import URLError

import pandas as pd
import pytest

from load import PARAMETERS
from live.fetch import TIMEOUT, drop_flagged, fetch_current_windows, group_windows, query_window


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


def test_group_windows_empty_frame():
    assert group_windows(pd.DataFrame()) == []


def test_group_windows_frame_with_no_rows():
    assert group_windows(make_df().iloc[0:0]) == []


class StubClient:
    def __init__(self, df):
        self.df = df

    def query(self, *args, **kwargs):
        return self.df


def test_drop_flagged_keeps_routine_quality_bits():
    df = make_df()
    df["QUALITY"] = df["QUALITY"].astype(object)
    df.loc[0, "QUALITY"] = 0x400
    df.loc[1, "QUALITY"] = "0x00000400"
    df.loc[2, "QUALITY"] = 1024.0
    assert len(drop_flagged(df)) == len(df)


def test_drop_flagged_drops_bad_records():
    df = make_df()
    df["QUALITY"] = df["QUALITY"].astype(object)
    df.loc[0, "QUALITY"] = 0x10000
    df.loc[1, "QUALITY"] = "0x00011C00"
    df.loc[2, "QUALITY"] = float("nan")
    assert len(drop_flagged(df)) == len(df) - 3


def test_drop_flagged_handles_empty_and_missing_column():
    assert drop_flagged(pd.DataFrame()).empty
    trimmed = make_df().drop(columns=["QUALITY"])
    assert len(drop_flagged(trimmed)) == len(trimmed)


def test_fetch_current_windows_filters_flagged_rows():
    df = make_df()
    df["QUALITY"] = [0, 0x10000, 0, 0, 0x11C00, 0]
    windows = fetch_current_windows("2026.06.19_TAI", client=StubClient(df))
    assert len(windows) == 2
    for window in windows:
        assert window["features"].shape == (2, len(PARAMETERS))


class FlakyClient:
    def __init__(self, failures):
        self.failures = failures
        self.calls = 0
        self.timeouts = []

    def query(self, *args, **kwargs):
        self.calls += 1
        self.timeouts.append(socket.getdefaulttimeout())
        if self.calls <= self.failures:
            raise URLError("connection timed out")
        return make_df()


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr("live.fetch.time.sleep", lambda seconds: None)


def test_query_window_recovers_after_failures(no_sleep):
    client = FlakyClient(failures=2)
    df = query_window("[2026.06.19_TAI/12h@60m]", client=client)
    assert client.calls == 3
    assert len(df) == 6


def test_query_window_raises_when_every_attempt_fails(no_sleep):
    client = FlakyClient(failures=99)
    with pytest.raises(URLError):
        query_window("[2026.06.19_TAI/12h@60m]", client=client)
    assert client.calls == 4


def test_query_window_applies_and_restores_socket_timeout(no_sleep):
    previous = socket.getdefaulttimeout()
    client = FlakyClient(failures=0)
    query_window("[2026.06.19_TAI/12h@60m]", client=client)
    assert client.timeouts == [TIMEOUT]
    assert socket.getdefaulttimeout() == previous


def test_query_window_restores_socket_timeout_on_failure(no_sleep):
    previous = socket.getdefaulttimeout()
    with pytest.raises(URLError):
        query_window("[2026.06.19_TAI/12h@60m]", client=FlakyClient(failures=99))
    assert socket.getdefaulttimeout() == previous


def test_live_fetch_smoke():
    try:
        windows = fetch_current_windows("2026.06.19_TAI", hours=12, cadence_min=60)
    except Exception:
        pytest.skip("no network or JSOC unavailable")
    assert len(windows) > 0
    assert windows[0]["features"].shape[1] == len(PARAMETERS)
