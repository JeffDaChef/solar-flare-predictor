import time

import drms
import pandas as pd

from load import PARAMETERS

NRT_SERIES = "hmi.sharp_cea_720s_nrt"
KEY_LIST = "HARPNUM,T_REC,NOAA_ARS,QUALITY," + ",".join(PARAMETERS)
RETRIES = 3
RETRY_WAIT = 15


def query_window(time_spec, client=None, retries=RETRIES, wait=RETRY_WAIT):
    client = client or drms.Client()
    for attempt in range(retries):
        try:
            return client.query(NRT_SERIES + "[]" + time_spec, key=KEY_LIST)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(wait)


def group_windows(df):
    windows = []
    for harpnum, group in df.groupby("HARPNUM"):
        group = group.sort_values("T_REC")
        numeric = group[PARAMETERS].apply(pd.to_numeric, errors="coerce")
        windows.append({
            "harpnum": int(harpnum),
            "noaa_ars": str(group["NOAA_ARS"].iloc[-1]),
            "features": numeric.to_numpy(dtype=float),
        })
    return windows


def fetch_current_windows(start_tai, hours=12, cadence_min=12, client=None):
    time_spec = "[%s/%dh@%dm]" % (start_tai, hours, cadence_min)
    return group_windows(query_window(time_spec, client=client))
