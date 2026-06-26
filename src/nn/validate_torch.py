import numpy as np

from nn.layers import LSTM, Linear, ReLU, Sequential
from nn.losses import bce_with_logits

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _bce_torch(logits, targets):
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="mean")


def _grad_error(pairs):
    return max(float(np.max(np.abs(mine - theirs.numpy()))) for mine, theirs in pairs)


def compare_mlp(rng):
    model = Sequential([Linear(4, 6, rng), ReLU(), Linear(6, 1, rng)])
    first, _, second = model.layers
    x = rng.normal(size=(5, 4))
    y = (rng.uniform(size=(5, 1)) > 0.5).astype(float)
    logits = model.forward(x)
    _, grad = bce_with_logits(logits, y)
    model.backward(grad)

    tx = torch.tensor(x)
    ty = torch.tensor(y)
    w1 = torch.tensor(first.weight.value, requires_grad=True)
    b1 = torch.tensor(first.bias.value, requires_grad=True)
    w2 = torch.tensor(second.weight.value, requires_grad=True)
    b2 = torch.tensor(second.bias.value, requires_grad=True)
    t_logits = torch.relu(tx @ w1 + b1) @ w2 + b2
    _bce_torch(t_logits, ty).backward()

    return {
        "forward": float(np.max(np.abs(logits - t_logits.detach().numpy()))),
        "grad": _grad_error([
            (first.weight.grad, w1.grad), (first.bias.grad, b1.grad),
            (second.weight.grad, w2.grad), (second.bias.grad, b2.grad),
        ]),
    }


def compare_lstm(rng):
    n, steps, depth, hidden = 4, 5, 3, 6
    model = Sequential([LSTM(depth, hidden, rng), Linear(hidden, 1, rng)])
    lstm, linear = model.layers
    x = rng.normal(size=(n, steps, depth))
    y = (rng.uniform(size=(n, 1)) > 0.5).astype(float)
    logits = model.forward(x)
    _, grad = bce_with_logits(logits, y)
    model.backward(grad)

    tx = torch.tensor(x)
    ty = torch.tensor(y)
    weight = torch.tensor(lstm.weight.value, requires_grad=True)
    bias = torch.tensor(lstm.bias.value, requires_grad=True)
    w_out = torch.tensor(linear.weight.value, requires_grad=True)
    b_out = torch.tensor(linear.bias.value, requires_grad=True)
    h = torch.zeros(n, hidden, dtype=torch.float64)
    c = torch.zeros(n, hidden, dtype=torch.float64)
    for t in range(steps):
        z = torch.cat([tx[:, t, :], h], dim=1)
        a = z @ weight + bias
        a_i, a_f, a_o, a_g = torch.split(a, hidden, dim=1)
        i = torch.sigmoid(a_i)
        f = torch.sigmoid(a_f)
        o = torch.sigmoid(a_o)
        g = torch.tanh(a_g)
        c = f * c + i * g
        h = o * torch.tanh(c)
    t_logits = h @ w_out + b_out
    _bce_torch(t_logits, ty).backward()

    return {
        "forward": float(np.max(np.abs(logits - t_logits.detach().numpy()))),
        "grad": _grad_error([
            (lstm.weight.grad, weight.grad), (lstm.bias.grad, bias.grad),
            (linear.weight.grad, w_out.grad), (linear.bias.grad, b_out.grad),
        ]),
    }


def main():
    rng = np.random.default_rng(0)
    mlp = compare_mlp(rng)
    lstm = compare_lstm(rng)
    print("From-scratch nets validated against PyTorch autograd:")
    print("  MLP    max forward diff %.2e    max gradient diff %.2e" % (mlp["forward"], mlp["grad"]))
    print("  LSTM   max forward diff %.2e    max gradient diff %.2e" % (lstm["forward"], lstm["grad"]))


if __name__ == "__main__":
    main()
