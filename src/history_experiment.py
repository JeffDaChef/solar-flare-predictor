import numpy as np

from baseline import evaluate, logistic
from preprocess import features_for_partition


def load(parts, with_history):
    xs, ys = [], []
    for n in parts:
        X, y, _ = features_for_partition(n, with_history=with_history)
        xs.append(X)
        ys.append(y)
    return np.vstack(xs), np.concatenate(ys)


def run(with_history):
    x_tr, y_tr = load([1, 2], with_history)
    x_val, y_val = load([3], with_history)
    x_te, y_te = load([4, 5], with_history)
    return evaluate(logistic(), x_tr, y_tr, x_val, y_val, x_te, y_te)


def main():
    print("Does adding recent flare history help? (logistic regression, honest split)")
    for with_history in (False, True):
        result = run(with_history)
        tag = "with history (148 feat)" if with_history else "magnetic only (144 feat)"
        print("  %-26s TSS %.3f  HSS %.3f  sens %.3f  spec %.3f"
              % (tag, result["tss"], result["hss"], result["sens"], result["spec"]))


if __name__ == "__main__":
    main()
