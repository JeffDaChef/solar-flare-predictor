import socket
import time

import drms
import pandas as pd

from load import LIVE_PARAMETERS

NRT_SERIES = "hmi.sharp_cea_720s_nrt"
KEY_LIST = "HARPNUM,T_REC,NOAA_ARS,QUALITY," + ",".join(LIVE_PARAMETERS)
RETRIES = 4
RETRY_WAITS = (30, 120, 300)
TIMEOUT = 45
BAD_QUALITY = 0x10000


def query_window(time_spec, client=None, retries=RETRIES, waits=RETRY_WAITS,
                 timeout=TIMEOUT):
    client = client or drms.Client()
    previous = socket.getdefaulttimeout()
    socket.setdefaulttimeout(timeout)
    try:
        for attempt in range(retries):
            try:
                return client.query(NRT_SERIES + "[]" + time_spec, key=KEY_LIST)
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(waits[min(attempt, len(waits) - 1)])
    finally:
        socket.setdefaulttimeout(previous)


def _quality_flags(value):
    text = str(value).strip()
    try:
        if text.lower().startswith("0x"):
            return int(text, 16)
        return int(float(text))
    except (TypeError, ValueError):
        return BAD_QUALITY


def drop_flagged(df):
    if df is None or df.empty or "QUALITY" not in df.columns:
        return df
    return df[df["QUALITY"].map(_quality_flags) < BAD_QUALITY]


def group_windows(df):
    windows = []
    if df is None or df.empty:
        return windows
    for harpnum, group in df.groupby("HARPNUM"):
        group = group.sort_values("T_REC")
        numeric = group[LIVE_PARAMETERS].apply(pd.to_numeric, errors="coerce")
        windows.append({
            "harpnum": int(harpnum),
            "noaa_ars": str(group["NOAA_ARS"].iloc[-1]),
            "features": numeric.to_numpy(dtype=float),
        })
    return windows


def fetch_current_windows(start_tai, hours=12, cadence_min=12, client=None):
    time_spec = "[%s/%dh@%dm]" % (start_tai, hours, cadence_min)
    return group_windows(drop_flagged(query_window(time_spec, client=client)))
