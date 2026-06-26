import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from metrics import accuracy, hss, sensitivity, specificity, tss
from preprocess import Standardizer, features_for_partition


def load_partitions(parts):
    xs, ys = [], []
    for n in parts:
        X, y, _ = features_for_partition(n)
        xs.append(X)
        ys.append(y)
    return np.vstack(xs), np.concatenate(ys)


def best_threshold(y_true, proba):
    thresholds = np.linspace(0.02, 0.98, 49)
    scores = [tss(y_true, (proba >= t).astype(int)) for t in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def evaluate(model, x_tr, y_tr, x_val, y_val, x_te, y_te):
    scaler = Standardizer().fit(x_tr)
    model.fit(scaler.transform(x_tr), y_tr)
    p_val = model.predict_proba(scaler.transform(x_val))[:, 1]
    p_te = model.predict_proba(scaler.transform(x_te))[:, 1]
    threshold = best_threshold(y_val, p_val)
    pred = (p_te >= threshold).astype(int)
    return {
        "tss": tss(y_te, pred),
        "hss": hss(y_te, pred),
        "sens": sensitivity(y_te, pred),
        "spec": specificity(y_te, pred),
        "accuracy": accuracy(y_te, pred),
        "threshold": threshold,
        "predicted_flares": int(pred.sum()),
        "actual_flares": int(y_te.sum()),
    }


def show(name, r):
    print("  %-22s TSS %.3f  HSS %.3f  sens %.3f  spec %.3f  acc %.3f"
          % (name, r["tss"], r["hss"], r["sens"], r["spec"], r["accuracy"]))
    print("  %-22s flagged %d as flares, of which %d were real (threshold %.2f)"
          % ("", r["predicted_flares"], r["actual_flares"], r["threshold"]))


def logistic():
    return LogisticRegression(class_weight="balanced", max_iter=2000)


def forest():
    return RandomForestClassifier(
        n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=0
    )


def main():
    print("HONEST split: train on 1,2  pick threshold on 3  test on 4,5")
    x_tr, y_tr = load_partitions([1, 2])
    x_val, y_val = load_partitions([3])
    x_te, y_te = load_partitions([4, 5])
    print("  train %d  val %d  test %d  (test is %.1f%% flares)"
          % (len(y_tr), len(y_val), len(y_te), 100 * y_te.mean()))
    print()
    lr_honest = evaluate(logistic(), x_tr, y_tr, x_val, y_val, x_te, y_te)
    show("logistic regression", lr_honest)
    rf_honest = evaluate(forest(), x_tr, y_tr, x_val, y_val, x_te, y_te)
    show("random forest", rf_honest)

    print()
    print("LEAKY split: same data pooled, split at random (overlapping windows leak)")
    x_all, y_all = load_partitions([1, 2, 3, 4, 5])
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y_all))
    a = int(0.6 * len(idx))
    b = int(0.75 * len(idx))
    tr, val, te = idx[:a], idx[a:b], idx[b:]
    print()
    lr_leak = evaluate(logistic(), x_all[tr], y_all[tr], x_all[val], y_all[val],
                       x_all[te], y_all[te])
    show("logistic regression", lr_leak)
    rf_leak = evaluate(forest(), x_all[tr], y_all[tr], x_all[val], y_all[val],
                       x_all[te], y_all[te])
    show("random forest", rf_leak)

    print()
    print("Leakage inflation (leaky TSS minus honest TSS):")
    print("  logistic regression: %+.3f" % (lr_leak["tss"] - lr_honest["tss"]))
    print("  random forest:       %+.3f" % (rf_leak["tss"] - rf_honest["tss"]))


if __name__ == "__main__":
    main()
