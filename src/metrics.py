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


def all_scores(y_true, y_pred):
    return {
        "tss": tss(y_true, y_pred),
        "hss": hss(y_true, y_pred),
        "sensitivity": sensitivity(y_true, y_pred),
        "specificity": specificity(y_true, y_pred),
        "accuracy": accuracy(y_true, y_pred),
    }
