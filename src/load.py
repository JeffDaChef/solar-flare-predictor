import tarfile
from collections import namedtuple

import numpy as np

PARAMETERS = [
    "TOTUSJH", "TOTBSQ", "TOTPOT", "TOTUSJZ", "ABSNJZH", "SAVNCPP",
    "USFLUX", "TOTFZ", "MEANPOT", "EPSZ", "MEANSHR", "SHRGT45",
    "MEANGAM", "MEANGBT", "MEANGBZ", "MEANGBH", "MEANJZH", "TOTFY",
    "MEANJZD", "MEANALP", "TOTFX", "EPSY", "EPSX", "R_VALUE",
]

HISTORY = ["BFLARE", "CFLARE", "MFLARE", "XFLARE"]

Instance = namedtuple("Instance", ["features", "history", "label", "flare_class", "name"])


def _to_float(value):
    try:
        return float(value)
    except ValueError:
        return np.nan


def _class_token(name):
    for i, char in enumerate(name):
        if char in "@_":
            return name[:i]
    return name


def _parse(raw_bytes, names):
    lines = raw_bytes.decode("utf-8").splitlines()
    header = lines[0].split("\t")
    index = [header.index(name) for name in names]
    rows = []
    for line in lines[1:]:
        if not line:
            continue
        fields = line.split("\t")
        rows.append([_to_float(fields[i]) for i in index])
    if not rows:
        return np.empty((0, len(names)))
    return np.asarray(rows, dtype=float)


def parse_instance(raw_bytes):
    return _parse(raw_bytes, PARAMETERS)


def parse_with_history(raw_bytes):
    combined = _parse(raw_bytes, PARAMETERS + HISTORY)
    cut = len(PARAMETERS)
    return combined[:, :cut], combined[:, cut:]


def iter_partition(tarball_path, limit=None):
    count = 0
    with tarfile.open(tarball_path, "r:gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".csv"):
                continue
            parts = member.name.split("/")
            folder = parts[-2]
            name = parts[-1]
            label = 1 if folder == "FL" else 0
            features, history = parse_with_history(tar.extractfile(member).read())
            yield Instance(features, history, label, _class_token(name), name)
            count += 1
            if limit is not None and count >= limit:
                break


def load_partition(tarball_path, limit=None):
    features, labels, classes, names = [], [], [], []
    for inst in iter_partition(tarball_path, limit=limit):
        features.append(inst.features)
        labels.append(inst.label)
        classes.append(inst.flare_class)
        names.append(inst.name)
    return features, np.asarray(labels), classes, names
