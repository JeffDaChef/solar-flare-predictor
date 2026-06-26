import re
import urllib.request
from datetime import datetime

FORECAST_URL = "https://services.swpc.noaa.gov/text/3-day-forecast.txt"


def fetch_forecast(url=FORECAST_URL):
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode()


def parse_forecast(text):
    lines = text.splitlines()
    year = int(re.search(r"Radio Blackout Forecast for.*?(\d{4})", text).group(1))
    r1_index = next(i for i, line in enumerate(lines) if line.strip().startswith("R1-R2"))
    months = re.findall(r"([A-Za-z]{3})\s+(\d{1,2})", lines[r1_index - 1])
    r1_values = [int(v) for v in re.findall(r"(\d+)%", lines[r1_index])]
    r3_line = next(line for line in lines if line.strip().startswith("R3"))
    r3_values = [int(v) for v in re.findall(r"(\d+)%", r3_line)]
    forecast = {}
    for (month, day), r1, r3 in zip(months, r1_values, r3_values):
        date = datetime.strptime("%s %s %d" % (month, day, year), "%b %d %Y").date()
        forecast[date.isoformat()] = {"m_class": r1 / 100.0, "x_class": r3 / 100.0}
    return forecast


def major_probability(forecast, day_iso):
    entry = forecast.get(day_iso)
    return entry["m_class"] if entry else None


def fetch_major_probability(day_iso, url=FORECAST_URL):
    return major_probability(parse_forecast(fetch_forecast(url)), day_iso)
