import json
import os
from datetime import datetime, timedelta, timezone

import joblib
import numpy as np

from live.fetch import fetch_current_windows
from live.noaa import fetch_forecast, major_probability, parse_forecast
from preprocess import summarize_instance

MODEL_PATH = "models/production.joblib"
LOG_PATH = "results/forecast_log.jsonl"
MIN_STEPS = 20


def recent_window_start(hours=12):
    start = datetime.now(timezone.utc) - timedelta(hours=hours)
    return start.strftime("%Y.%m.%d_%H:%M:%S_TAI")


def forecast_from_windows(windows, scaler, model, min_steps=MIN_STEPS):
    rows = []
    for window in windows:
        if window["features"].shape[0] < min_steps:
            continue
        features = summarize_instance(window["features"]).reshape(1, -1)
        prob = float(model.predict_proba(scaler.transform(features))[0, 1])
        rows.append({
            "harpnum": window["harpnum"],
            "noaa_ars": window["noaa_ars"],
            "prob": prob,
        })
    probs = np.array([r["prob"] for r in rows])
    full_disk = float(1.0 - np.prod(1.0 - probs)) if probs.size else 0.0
    full_disk = min(max(full_disk, 0.005), 0.995)
    rows.sort(key=lambda r: r["prob"], reverse=True)
    return full_disk, rows


def make_forecast(start_tai=None, hours=12, model_path=MODEL_PATH, log_path=LOG_PATH):
    if start_tai is None:
        start_tai = recent_window_start(hours)
    bundle = joblib.load(model_path)
    windows = fetch_current_windows(start_tai, hours=hours)
    full_disk, rows = forecast_from_windows(windows, bundle["scaler"], bundle["model"])
    issued = datetime.now(timezone.utc)
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
        "top_regions": rows[:3],
    }
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as handle:
        handle.write(json.dumps(record) + "\n")
    return record


if __name__ == "__main__":
    result = make_forecast()
    print("Forecast issued %s" % result["issued_utc"])
    print("Chance of an M or X flare in the next 24h: %.1f%%"
          % (100 * result["full_disk_prob"]))
    if result["noaa_major_prob"] is not None:
        print("NOAA's forecast for the same day:          %.1f%%"
              % (100 * result["noaa_major_prob"]))
    print("Based on %d active regions. Most active:" % result["n_regions"])
    for row in result["top_regions"]:
        print("  HARP %d (NOAA %s): %.1f%%"
              % (row["harpnum"], row["noaa_ars"], 100 * row["prob"]))
