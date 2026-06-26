import os
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression

from preprocess import Standardizer, features_for_partition

MODEL_PATH = "models/production.joblib"


def _stack(parts):
    xs, ys = [], []
    for n in parts:
        X, y, _ = features_for_partition(n)
        xs.append(X)
        ys.append(y)
    return np.vstack(xs), np.concatenate(ys)


def train_and_save(train_parts=(1, 2, 3, 4), calib_parts=(5,), path=MODEL_PATH):
    x_train, y_train = _stack(train_parts)
    x_calib, y_calib = _stack(calib_parts)
    scaler = Standardizer().fit(x_train)
    base = LogisticRegression(class_weight="balanced", max_iter=2000)
    base.fit(scaler.transform(x_train), y_train)
    model = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic")
    model.fit(scaler.transform(x_calib), y_calib)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(
        {
            "scaler": scaler,
            "model": model,
            "trained_utc": datetime.now(timezone.utc).isoformat(),
            "n_train": int(len(y_train) + len(y_calib)),
            "n_flares": int(y_train.sum() + y_calib.sum()),
        },
        path,
    )
    return path, int(len(y_train) + len(y_calib)), int(y_train.sum() + y_calib.sum())


if __name__ == "__main__":
    path, n, flares = train_and_save()
    print("saved %s, trained on %d instances (%d flares)" % (path, n, flares))
