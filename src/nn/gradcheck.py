import numpy as np


def check_gradients(model, loss_fn, x, y, eps=1e-6):
    model.zero_grad()
    _, dlogits = loss_fn(model.forward(x), y)
    model.backward(dlogits)
    max_error = 0.0
    for param in model.parameters():
        analytic = param.grad.copy()
        numeric = np.zeros_like(param.value)
        iterator = np.nditer(param.value, flags=["multi_index"])
        while not iterator.finished:
            idx = iterator.multi_index
            original = param.value[idx]
            param.value[idx] = original + eps
            loss_plus, _ = loss_fn(model.forward(x), y)
            param.value[idx] = original - eps
            loss_minus, _ = loss_fn(model.forward(x), y)
            param.value[idx] = original
            numeric[idx] = (loss_plus - loss_minus) / (2 * eps)
            iterator.iternext()
        max_error = max(max_error, float(np.max(np.abs(analytic - numeric))))
    return max_error
