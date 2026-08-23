import pytest

import daily
from live.forecast import NO_DATA_MESSAGE


def stub_scoreboard(monkeypatch, calls):
    def build():
        calls.append(True)
        return {"n": 3}, []
    monkeypatch.setattr(daily, "build_scoreboard", build)


def test_main_fails_loudly_when_there_is_no_data(monkeypatch):
    calls = []
    stub_scoreboard(monkeypatch, calls)
    monkeypatch.setattr(daily, "make_forecast", lambda: None)
    with pytest.raises(SystemExit) as error:
        daily.main()
    assert error.value.code == NO_DATA_MESSAGE
    assert calls == [True]


def test_main_succeeds_when_a_forecast_was_issued(monkeypatch):
    calls = []
    stub_scoreboard(monkeypatch, calls)
    record = {"full_disk_prob": 0.1, "n_regions": 5, "noaa_major_prob": None}
    monkeypatch.setattr(daily, "make_forecast", lambda: record)
    daily.main()
    assert calls == [True]
