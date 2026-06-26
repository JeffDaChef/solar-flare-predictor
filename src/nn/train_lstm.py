import numpy as np

from metrics import hss, sensitivity, specificity, tss
from nn.layers import LSTM, Linear, Sequential
from nn.losses import bce_with_logits, sigmoid
from nn.optim import Adam
from preprocess import sequences_for_partition


def load_seq(parts):
    xs, ys = [], []
    for n in parts:
        X, y = sequences_for_partition(n)
        xs.append(X)
        ys.append(y)
    return np.concatenate(xs), np.concatenate(ys)


def normalize_fit(X):
    flat = X.reshape(-1, X.shape[2]).astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        mean = flat.mean(axis=0)
        std = flat.std(axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > 0), std, 1.0)
    return mean, std


def normalize_apply(X, mean, std, clip=10.0):
    return np.clip((X - mean) / std, -clip, clip)


def predict_proba(model, X, batch=512):
    out = []
    for start in range(0, X.shape[0], batch):
        logits = model.forward(X[start:start + batch])
        out.append(sigmoid(logits.reshape(-1)))
    return np.concatenate(out)


def best_threshold(y_true, proba):
    thresholds = np.linspace(0.02, 0.98, 49)
    scores = [tss(y_true, (proba >= t).astype(int)) for t in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def train(model, X, y, epochs=12, batch=256, lr=1e-3, pos_weight=1.0, rng=None):
    rng = rng or np.random.default_rng(0)
    opt = Adam(model.parameters(), lr=lr)
    targets = y.reshape(-1, 1).astype(float)
    n = X.shape[0]
    for epoch in range(epochs):
        order = rng.permutation(n)
        total = 0.0
        for start in range(0, n, batch):
            idx = order[start:start + batch]
            opt.zero_grad()
            loss, grad = bce_with_logits(model.forward(X[idx]), targets[idx], pos_weight)
            model.backward(grad)
            opt.step()
            total += loss
        print("  epoch %2d  avg loss %.4f" % (epoch + 1, total / (n / batch)), flush=True)
    return model


def main():
    x_tr, y_tr = load_seq([1, 2])
    x_val, y_val = load_seq([3])
    x_te, y_te = load_seq([4, 5])
    mean, std = normalize_fit(x_tr)
    x_tr = normalize_apply(x_tr, mean, std)
    x_val = normalize_apply(x_val, mean, std)
    x_te = normalize_apply(x_te, mean, std)
    pos_weight = float((y_tr == 0).sum()) / float((y_tr == 1).sum())
    rng = np.random.default_rng(0)
    model = Sequential([LSTM(x_tr.shape[2], 32, rng), Linear(32, 1, rng)])
    print("training from-scratch LSTM on %d sequences..." % len(y_tr))
    train(model, x_tr, y_tr, epochs=12, batch=256, lr=1e-3, pos_weight=pos_weight, rng=rng)
    p_val = predict_proba(model, x_val)
    p_te = predict_proba(model, x_te)
    threshold = best_threshold(y_val, p_val)
    pred = (p_te >= threshold).astype(int)
    print()
    print("from-scratch LSTM (32 hidden) on real flare sequences, honest split:")
    print("  TSS %.3f  HSS %.3f  sens %.3f  spec %.3f  (threshold %.2f)"
          % (tss(y_te, pred), hss(y_te, pred), sensitivity(y_te, pred),
             specificity(y_te, pred), threshold))
    print("  (test is a balanced subsample; TSS is ratio-insensitive so it")
    print("   compares directly to the baseline numbers)")


if __name__ == "__main__":
    main()
