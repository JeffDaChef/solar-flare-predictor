import os
import warnings

import numpy as np

from load import HISTORY, PARAMETERS, iter_partition

STATS = ["mean", "std", "min", "max", "last", "slope"]
N_FEATURES = len(PARAMETERS) * len(STATS)


def _last(column):
    mask = ~np.isnan(column)
    if not mask.any():
        return np.nan
    return column[mask][-1]


def _slope(column):
    mask = ~np.isnan(column)
    if mask.sum() < 2:
        return np.nan
    t = np.arange(column.shape[0])
    return np.polyfit(t[mask], column[mask], 1)[0]


def summarize_instance(features):
    if features.ndim != 2 or features.shape[0] == 0:
        return np.full(N_FEATURES, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(features, axis=0)
        std = np.nanstd(features, axis=0)
        cmin = np.nanmin(features, axis=0)
        cmax = np.nanmax(features, axis=0)
    last = np.array([_last(features[:, j]) for j in range(features.shape[1])])
    slope = np.array([_slope(features[:, j]) for j in range(features.shape[1])])
    return np.concatenate([mean, std, cmin, cmax, last, slope])


def summarize_history(history):
    if history.ndim != 2 or history.shape[0] == 0:
        return np.zeros(len(HISTORY))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        peak = np.nanmax(history, axis=0)
    return np.where(np.isnan(peak), 0.0, peak)


class Standardizer:
    def fit(self, X):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self.mean_ = np.nanmean(X, axis=0)
            self.std_ = np.nanstd(X, axis=0)
        self.mean_ = np.where(np.isnan(self.mean_), 0.0, self.mean_)
        self.std_ = np.where(np.isnan(self.std_) | (self.std_ == 0), 1.0, self.std_)
        return self

    def transform(self, X):
        X = np.where(np.isnan(X), self.mean_, X)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def build_features(tarball_path, limit=None, with_history=False):
    rows, labels, classes = [], [], []
    for inst in iter_partition(tarball_path, limit=limit):
        vector = summarize_instance(inst.features)
        if with_history:
            vector = np.concatenate([vector, summarize_history(inst.history)])
        rows.append(vector)
        labels.append(inst.label)
        classes.append(inst.flare_class)
    return np.asarray(rows), np.asarray(labels), classes


def features_for_partition(n, data_dir="data", limit=None, with_history=False):
    name = "histfeat_partition%d.npz" if with_history else "partition%d.npz"
    cache = os.path.join(data_dir, "processed", name % n)
    if limit is None and os.path.exists(cache):
        stored = np.load(cache, allow_pickle=True)
        return stored["X"], stored["y"], list(stored["classes"])
    tarball = os.path.join(data_dir, "partition%d_instances.tar.gz" % n)
    X, y, classes = build_features(tarball, limit=limit, with_history=with_history)
    if limit is None:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez(cache, X=X, y=y, classes=np.asarray(classes))
    return X, y, classes


SEQ_LEN = 60


def instance_to_sequence(features, length=SEQ_LEN):
    arr = np.asarray(features, dtype=float)
    width = len(PARAMETERS)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.zeros((length, width))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        col_mean = np.nanmean(arr, axis=0)
    col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
    bad = np.isnan(arr)
    if bad.any():
        arr = arr.copy()
        arr[bad] = np.take(col_mean, np.where(bad)[1])
    steps = arr.shape[0]
    if steps >= length:
        return arr[-length:]
    pad = np.repeat(arr[:1], length - steps, axis=0)
    return np.vstack([pad, arr])


def sequences_for_partition(n, max_neg=8000, data_dir="data"):
    cache = os.path.join(data_dir, "processed", "seq_partition%d.npz" % n)
    if os.path.exists(cache):
        stored = np.load(cache)
        return stored["X"], stored["y"]
    tarball = os.path.join(data_dir, "partition%d_instances.tar.gz" % n)
    seqs, labels, neg = [], [], 0
    for inst in iter_partition(tarball):
        if inst.label == 0:
            if neg >= max_neg:
                continue
            neg += 1
        seqs.append(instance_to_sequence(inst.features).astype(np.float32))
        labels.append(inst.label)
    X = np.asarray(seqs, dtype=np.float32)
    y = np.asarray(labels)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, X=X, y=y)
    return X, y


def load_split(train_partitions, test_partitions, data_dir="data"):
    def stack(parts):
        xs, ys = [], []
        for n in parts:
            X, y, _ = features_for_partition(n, data_dir=data_dir)
            xs.append(X)
            ys.append(y)
        return np.vstack(xs), np.concatenate(ys)

    x_train, y_train = stack(train_partitions)
    x_test, y_test = stack(test_partitions)
    scaler = Standardizer().fit(x_train)
    return scaler.transform(x_train), y_train, scaler.transform(x_test), y_test, scaler
