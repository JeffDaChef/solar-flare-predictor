import json
import os
from datetime import datetime, timedelta, timezone

import joblib
import numpy as np

from load import LIVE_PARAMETERS
from live.fetch import SPAN_HOURS, fetch_current_windows
from live.noaa import fetch_forecast, major_probability, parse_forecast
from preprocess import summarize_instance

MODEL_PATH = "models/production.joblib"
LOG_PATH = "results/forecast_log.jsonl"
MIN_STEPS = 20
WINDOW_HOURS = 12
STEP_HOURS = 1.5
N_WINDOWS = 9
SCORING = "strongest-region/9-window"
NO_DATA_MESSAGE = ("No usable SHARP data in this window, so no forecast was issued. "
                   "This usually means JSOC has a gap in the near real time series.")


def recent_window_start(hours=SPAN_HOURS):
    start = datetime.now(timezone.utc) - timedelta(hours=hours)
    return start.strftime("%Y.%m.%d_%H:%M:%S_TAI")


def best_window(window, scaler, model, issued, min_steps=MIN_STEPS):
    times = window["times"]
    best = None
    for step in range(N_WINDOWS):
        end = issued - timedelta(hours=STEP_HOURS * step)
        start = end - timedelta(hours=WINDOW_HOURS)
        mask = ((times > np.datetime64(start.replace(tzinfo=None)))
                & (times <= np.datetime64(end.replace(tzinfo=None))))
        if mask.sum() < min_steps:
            continue
        summary = summarize_instance(window["features"][mask])
        prob = float(model.predict_proba(scaler.transform(summary.reshape(1, -1)))[0, 1])
        if best is None or prob > best[0]:
            best = (prob, summary)
    return best


def forecast_from_windows(windows, scaler, model, issued, min_steps=MIN_STEPS):
    rows = []
    for window in windows:
        best = best_window(window, scaler, model, issued, min_steps)
        if best is None:
            continue
        prob, summary = best
        rows.append({
            "harpnum": window["harpnum"],
            "noaa_ars": window["noaa_ars"],
            "prob": prob,
            "features": {name: float("%.6g" % summary[i]) for i, name in enumerate(LIVE_PARAMETERS)},
        })
    probs = np.array([r["prob"] for r in rows])
    full_disk = float(probs.max()) if probs.size else 0.0
    full_disk = min(max(full_disk, 0.001), 0.995)
    rows.sort(key=lambda r: r["prob"], reverse=True)
    return full_disk, rows


def make_forecast(start_tai=None, hours=SPAN_HOURS, model_path=MODEL_PATH, log_path=LOG_PATH):
    if start_tai is None:
        start_tai = recent_window_start(hours)
    bundle = joblib.load(model_path)
    issued = datetime.now(timezone.utc)
    windows = fetch_current_windows(start_tai, hours=hours)
    full_disk, rows = forecast_from_windows(windows, bundle["scaler"], bundle["model"], issued)
    if not rows:
        return None
    try:
        noaa = major_probability(parse_forecast(fetch_forecast()), issued.date().isoformat())
    except Exception:
        noaa = None
    record = {
        "issued_utc": issued.isoformat(),
        "data_window_start": start_tai,
        "n_regions": len(windows),
        "full_disk_prob": full_disk,
        "noaa_major_prob": noaa,
        "model_trained_utc": bundle.get("trained_utc"),
        "scoring": SCORING,
        "regions": rows,
    }
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as handle:
        handle.write(json.dumps(record) + "\n")
    return record


if __name__ == "__main__":
    result = make_forecast()
    if result is None:
        raise SystemExit(NO_DATA_MESSAGE)
    print("Forecast issued %s" % result["issued_utc"])
    print("Chance of an M or X flare in the next 24h: %.1f%%"
          % (100 * result["full_disk_prob"]))
    if result["noaa_major_prob"] is not None:
        print("NOAA's forecast for the same day:          %.1f%%"
              % (100 * result["noaa_major_prob"]))
    print("Based on %d active regions. Most active:" % result["n_regions"])
    for row in result["regions"][:3]:
        print("  HARP %d (NOAA %s): %.1f%%"
              % (row["harpnum"], row["noaa_ars"], 100 * row["prob"]))
