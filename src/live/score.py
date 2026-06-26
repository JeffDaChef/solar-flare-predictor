import json
import urllib.request
from datetime import datetime, timezone

LONG_BAND = "0.1-0.8nm"
GOES_XRAY_URL = "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json"

CLASS_THRESHOLDS = [("X", 1e-4), ("M", 1e-5), ("C", 1e-6), ("B", 1e-7)]
MAJOR_THRESHOLD = 1e-5


def flare_class(flux):
    for name, threshold in CLASS_THRESHOLDS:
        if flux >= threshold:
            return name
    return "A"


def parse_records(raw):
    records = []
    for row in raw:
        if row.get("energy") != LONG_BAND:
            continue
        flux = row.get("flux")
        if flux is None or float(flux) <= 0:
            continue
        stamp = datetime.strptime(row["time_tag"], "%Y-%m-%dT%H:%M:%SZ")
        records.append((stamp.replace(tzinfo=timezone.utc), float(flux)))
    return records


def peak_flux_on(records, day):
    same_day = [flux for stamp, flux in records if stamp.date() == day]
    return max(same_day) if same_day else None


def major_flare_occurred(records, day):
    peak = peak_flux_on(records, day)
    return peak is not None and peak >= MAJOR_THRESHOLD


def major_flare_in_window(records, start, end):
    return any(start <= stamp < end and flux >= MAJOR_THRESHOLD for stamp, flux in records)


def daily_peaks(records):
    by_day = {}
    for stamp, flux in records:
        day = stamp.date()
        if day not in by_day or flux > by_day[day]:
            by_day[day] = flux
    return sorted(by_day.items())


def fetch_goes_xray(url=GOES_XRAY_URL):
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return parse_records(json.load(response))
