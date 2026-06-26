import numpy as np

from metrics import hss, sensitivity, specificity, tss
from nn.layers import Linear, ReLU, Sequential
from nn.losses import bce_with_logits, sigmoid
from nn.optim import Adam
from preprocess import Standardizer, features_for_partition


def load(parts):
    xs, ys = [], []
    for n in parts:
        X, y, _ = features_for_partition(n)
        xs.append(X)
        ys.append(y)
    return np.vstack(xs), np.concatenate(ys)


def build_mlp(in_dim, hidden=64, rng=None):
    rng = rng or np.random.default_rng(0)
    return Sequential([
        Linear(in_dim, hidden, rng), ReLU(),
        Linear(hidden, hidden, rng), ReLU(),
        Linear(hidden, 1, rng),
    ])


def predict_proba(model, X):
    return sigmoid(model.forward(X).reshape(-1))


def train(model, X, y, epochs=12, batch=512, lr=1e-3, pos_weight=1.0, rng=None):
    rng = rng or np.random.default_rng(0)
    opt = Adam(model.parameters(), lr=lr)
    targets = y.reshape(-1, 1).astype(float)
    n = X.shape[0]
    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch):
            idx = order[start:start + batch]
            opt.zero_grad()
            loss, grad = bce_with_logits(model.forward(X[idx]), targets[idx], pos_weight)
            model.backward(grad)
            opt.step()
    return model


def best_threshold(y_true, proba):
    thresholds = np.linspace(0.02, 0.98, 49)
    scores = [tss(y_true, (proba >= t).astype(int)) for t in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def main():
    x_tr, y_tr = load([1, 2])
    x_val, y_val = load([3])
    x_te, y_te = load([4, 5])
    scaler = Standardizer().fit(x_tr)
    x_tr, x_val, x_te = scaler.transform(x_tr), scaler.transform(x_val), scaler.transform(x_te)
    pos_weight = float((y_tr == 0).sum()) / float((y_tr == 1).sum())
    rng = np.random.default_rng(0)
    model = build_mlp(x_tr.shape[1], hidden=64, rng=rng)
    train(model, x_tr, y_tr, epochs=12, pos_weight=pos_weight, rng=rng)
    p_val = predict_proba(model, x_val)
    p_te = predict_proba(model, x_te)
    threshold = best_threshold(y_val, p_val)
    pred = (p_te >= threshold).astype(int)
    print("from-scratch numpy MLP (64-64 hidden), honest split:")
    print("  TSS %.3f  HSS %.3f  sens %.3f  spec %.3f  (threshold %.2f)"
          % (tss(y_te, pred), hss(y_te, pred), sensitivity(y_te, pred),
             specificity(y_te, pred), threshold))


if __name__ == "__main__":
    main()
