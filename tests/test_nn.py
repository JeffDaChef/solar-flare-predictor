import numpy as np

from nn.gradcheck import check_gradients
from nn.layers import LSTM, Linear, ReLU, Sequential, Tanh
from nn.losses import bce_with_logits, sigmoid
from nn.optim import Adam


def small_model(rng):
    return Sequential([
        Linear(5, 8, rng), Tanh(),
        Linear(8, 4, rng), ReLU(),
        Linear(4, 1, rng),
    ])


def test_sigmoid_is_stable_and_correct():
    z = np.array([-1000.0, 0.0, 1000.0])
    out = sigmoid(z)
    assert np.isfinite(out).all()
    assert out[1] == 0.5
    assert out[0] < 1e-6 and out[2] > 1 - 1e-6


def test_gradcheck_mlp():
    rng = np.random.default_rng(0)
    model = small_model(rng)
    x = rng.normal(size=(7, 5))
    y = (rng.uniform(size=(7, 1)) > 0.5).astype(float)
    assert check_gradients(model, bce_with_logits, x, y) < 1e-5


def test_gradcheck_with_pos_weight():
    rng = np.random.default_rng(1)
    model = small_model(rng)
    x = rng.normal(size=(6, 5))
    y = (rng.uniform(size=(6, 1)) > 0.5).astype(float)
    weighted = lambda logits, targets: bce_with_logits(logits, targets, pos_weight=8.0)
    assert check_gradients(model, weighted, x, y) < 1e-5


def test_gradcheck_lstm():
    rng = np.random.default_rng(4)
    model = Sequential([LSTM(3, 4, rng), Linear(4, 1, rng)])
    x = rng.normal(size=(5, 6, 3))
    y = (rng.uniform(size=(5, 1)) > 0.5).astype(float)
    assert check_gradients(model, bce_with_logits, x, y) < 1e-5


def test_lstm_learns_toy_sequence():
    rng = np.random.default_rng(3)
    n, steps, depth, hidden = 200, 8, 3, 8
    x = rng.normal(size=(n, steps, depth))
    y = (x[:, :, 0].sum(axis=1) > 0).astype(float).reshape(-1, 1)
    model = Sequential([LSTM(depth, hidden, rng), Linear(hidden, 1, rng)])
    opt = Adam(model.parameters(), lr=0.05)
    for _ in range(150):
        opt.zero_grad()
        loss, grad = bce_with_logits(model.forward(x), y)
        model.backward(grad)
        opt.step()
    pred = (sigmoid(model.forward(x).reshape(-1)) > 0.5).astype(float)
    assert (pred == y.reshape(-1)).mean() > 0.9


def test_training_reduces_loss():
    rng = np.random.default_rng(2)
    model = small_model(rng)
    x = rng.normal(size=(40, 5))
    y = (x[:, 0] > 0).astype(float).reshape(-1, 1)
    opt = Adam(model.parameters(), lr=0.05)
    first, last = None, None
    for _ in range(300):
        opt.zero_grad()
        loss, grad = bce_with_logits(model.forward(x), y)
        model.backward(grad)
        opt.step()
        first = loss if first is None else first
        last = loss
    assert last < first * 0.5
