import numpy as np


def confusion_counts(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    return tp, fp, fn, tn


def _safe_divide(numerator, denominator):
    if denominator == 0:
        return 0.0
    return numerator / denominator


def sensitivity(y_true, y_pred):
    tp, fp, fn, tn = confusion_counts(y_true, y_pred)
    return _safe_divide(tp, tp + fn)


def specificity(y_true, y_pred):
    tp, fp, fn, tn = confusion_counts(y_true, y_pred)
    return _safe_divide(tn, tn + fp)


def tss(y_true, y_pred):
    tp, fp, fn, tn = confusion_counts(y_true, y_pred)
    return _safe_divide(tp, tp + fn) - _safe_divide(fp, fp + tn)


def hss(y_true, y_pred):
    tp, fp, fn, tn = confusion_counts(y_true, y_pred)
    numerator = 2.0 * (tp * tn - fp * fn)
    denominator = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    return _safe_divide(numerator, denominator)


def accuracy(y_true, y_pred):
    tp, fp, fn, tn = confusion_counts(y_true, y_pred)
    return _safe_divide(tp + tn, tp + fp + fn + tn)


def auc(y_true, proba):
    positives = sum(1 for value in y_true if value)
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        return None
    pairs = sorted(zip(proba, y_true))
    total, i = 0.0, 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        total += rank * sum(1 for k in range(i, j + 1) if pairs[k][1])
        i = j + 1
    return (total - positives * (positives + 1) / 2.0) / (positives * negatives)


def best_threshold(y_true, proba, grid=None):
    grid = np.linspace(0.02, 0.98, 49) if grid is None else np.asarray(grid)
    proba = np.asarray(proba)
    scores = [tss(y_true, (proba >= t).astype(int)) for t in grid]
    return float(grid[int(np.argmax(scores))])


def all_scores(y_true, y_pred):
    return {
        "tss": tss(y_true, y_pred),
        "hss": hss(y_true, y_pred),
        "sensitivity": sensitivity(y_true, y_pred),
        "specificity": specificity(y_true, y_pred),
        "accuracy": accuracy(y_true, y_pred),
    }
