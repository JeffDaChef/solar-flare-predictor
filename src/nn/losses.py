import numpy as np


def sigmoid(z):
    out = np.empty_like(z, dtype=float)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def bce_with_logits(logits, targets, pos_weight=1.0):
    z = logits.reshape(-1)
    y = targets.reshape(-1).astype(float)
    weight = np.where(y == 1.0, pos_weight, 1.0)
    per_sample = np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))
    loss = float(np.mean(weight * per_sample))
    grad = weight * (sigmoid(z) - y) / z.shape[0]
    return loss, grad.reshape(logits.shape)
