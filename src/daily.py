from live.forecast import make_forecast
from live.scoreboard import build_scoreboard


def main():
    record = make_forecast()
    print("Forecast: %.1f%% chance of a major flare in 24h (%d active regions)"
          % (100 * record["full_disk_prob"], record["n_regions"]))
    if record.get("noaa_major_prob") is not None:
        print("NOAA forecast for the day: %.1f%%" % (100 * record["noaa_major_prob"]))
    summary, _ = build_scoreboard()
    print("Scoreboard: %d forecasts graded so far" % summary["n"])


if __name__ == "__main__":
    main()

