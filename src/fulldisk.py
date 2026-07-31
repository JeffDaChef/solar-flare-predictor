import json
import os
import re
from collections import defaultdict

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

from load import iter_partition
from metrics import best_threshold, tss
from preprocess import summarize_instance

MODEL_PATH = "models/production.joblib"
RESULT_PATH = "results/fulldisk.json"


def daily_scores(partition, scaler, model, data_dir="data"):
    feats, regions, days, labels = [], [], [], []
    tarball = os.path.join(data_dir, "partition%d_instances.tar.gz" % partition)
    for inst in iter_partition(tarball):
        region = re.search(r"_ar(\d+)_", inst.name)
        day = re.search(r"_e(\d{4}-\d{2}-\d{2})T", inst.name)
        if not region or not day:
            continue
        feats.append(summarize_instance(inst.features))
        regions.append(region.group(1))
        days.append(day.group(1))
        labels.append(inst.label)
    probs = model.predict_proba(scaler.transform(np.asarray(feats)))[:, 1]
    per_day = defaultdict(dict)
    flare_day = defaultdict(int)
    for region, day, prob, label in zip(regions, days, probs, labels):
        if region not in per_day[day] or prob > per_day[day][region]:
            per_day[day][region] = prob
        if label == 1:
            flare_day[day] = 1
    order = sorted(per_day)
    noisy_or = np.array([1.0 - np.prod([1.0 - p for p in per_day[d].values()]) for d in order])
    y = np.array([flare_day[d] for d in order])
    return noisy_or, y


def evaluate(partition=5, model_path=MODEL_PATH, result_path=RESULT_PATH):
    bundle = joblib.load(model_path)
    scores, y = daily_scores(partition, bundle["scaler"], bundle["model"])
    deployed = bundle.get("threshold")
    hindsight = best_threshold(y, scores)
    if deployed is None:
        deployed = hindsight
    result = {
        "partition": partition,
        "days": int(len(y)),
        "flare_days": int(y.sum()),
        "flare_rate": float(y.mean()),
        "mean_forecast": float(scores.mean()),
        "auc": float(roc_auc_score(y, scores)),
        "tss": tss(y, (scores >= deployed).astype(int)),
        "threshold": float(deployed),
        "threshold_source": bundle.get("threshold_parts"),
        "tss_in_hindsight": tss(y, (scores >= hindsight).astype(int)),
        "best_threshold_in_hindsight": hindsight,
    }
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w") as handle:
        json.dump(result, handle, indent=2)
    return result


if __name__ == "__main__":
    r = evaluate()
    print("Full-disk daily flare forecast, held-out partition %d:" % r["partition"])
    print("  %d days, %d flare days (%.1f%%)" % (r["days"], r["flare_days"], 100 * r["flare_rate"]))
    print("  AUC %.3f   TSS %.3f at the deployed threshold %.2f (picked on partition %s)"
          % (r["auc"], r["tss"], r["threshold"], r["threshold_source"]))
    print("  TSS %.3f if the threshold were refit on this partition (%.2f), which would be cheating"
          % (r["tss_in_hindsight"], r["best_threshold_in_hindsight"]))
    print("  mean forecast %.1f%% vs actual rate %.1f%% (full-disk calibration)"
          % (100 * r["mean_forecast"], 100 * r["flare_rate"]))
