import json
import os
from datetime import datetime, timedelta, timezone

from live.score import fetch_goes_xray, major_flare_in_window
from metrics import hss, tss

LOG_PATH = "results/forecast_log.jsonl"
BOARD_PATH = "results/scoreboard.json"


def read_forecasts(log_path=LOG_PATH):
    forecasts = []
    if not os.path.exists(log_path):
        return forecasts
    with open(log_path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                forecasts.append(json.loads(line))
    return forecasts


def grade_forecasts(forecasts, records, horizon_hours=24, now=None):
    now = now or datetime.now(timezone.utc)
    graded = []
    for forecast in forecasts:
        issued = datetime.fromisoformat(forecast["issued_utc"])
        end = issued + timedelta(hours=horizon_hours)
        if now < end:
            continue
        graded.append({
            "issued_utc": forecast["issued_utc"],
            "prob": forecast["full_disk_prob"],
            "noaa_prob": forecast.get("noaa_major_prob"),
            "actual": bool(major_flare_in_window(records, issued, end)),
        })
    return graded


def _score(probs, actual, threshold):
    predicted = [1 if p >= threshold else 0 for p in probs]
    brier = sum((p - a) ** 2 for p, a in zip(probs, actual)) / len(probs)
    return {"n": len(probs), "tss": tss(actual, predicted), "hss": hss(actual, predicted), "brier": brier}


def summarize(graded, threshold=0.5):
    if not graded:
        return {"n": 0, "threshold": threshold}
    actual = [1 if g["actual"] else 0 for g in graded]
    summary = {
        "n": len(graded),
        "flares": sum(actual),
        "threshold": threshold,
        "model": _score([g["prob"] for g in graded], actual, threshold),
    }
    paired = [(g["prob"], g["noaa_prob"], a)
              for g, a in zip(graded, actual) if g.get("noaa_prob") is not None]
    if paired:
        summary["noaa"] = _score([p[1] for p in paired], [p[2] for p in paired], threshold)
        summary["model_head_to_head"] = _score([p[0] for p in paired], [p[2] for p in paired], threshold)
    return summary


def build_scoreboard(log_path=LOG_PATH, board_path=BOARD_PATH, threshold=0.5):
    forecasts = read_forecasts(log_path)
    records = fetch_goes_xray()
    graded = grade_forecasts(forecasts, records)
    summary = summarize(graded, threshold)
    os.makedirs(os.path.dirname(board_path), exist_ok=True)
    with open(board_path, "w") as handle:
        json.dump({"summary": summary, "graded": graded}, handle, indent=2)
    return summary, graded


def main():
    summary, _ = build_scoreboard()
    if summary["n"] == 0:
        print("No forecasts have completed their 24h window yet.")
        print("Pending forecasts: %d. The scoreboard fills in as they close."
              % len(read_forecasts()))
        return
    model = summary["model"]
    print("Scoreboard over %d graded forecasts (%d had a real flare):"
          % (summary["n"], summary["flares"]))
    print("  our model:  TSS %.3f  HSS %.3f  Brier %.4f"
          % (model["tss"], model["hss"], model["brier"]))
    if "noaa" in summary:
        noaa = summary["noaa"]
        ours = summary["model_head_to_head"]
        print("  head to head on the %d days NOAA also forecast:" % noaa["n"])
        print("    us:   TSS %.3f   Brier %.4f" % (ours["tss"], ours["brier"]))
        print("    NOAA: TSS %.3f   Brier %.4f" % (noaa["tss"], noaa["brier"]))


if __name__ == "__main__":
    main()
