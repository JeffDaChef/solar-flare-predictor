import numpy as np

from nn.losses import sigmoid


class Parameter:
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class Linear:
    def __init__(self, in_dim, out_dim, rng):
        scale = np.sqrt(2.0 / in_dim)
        self.weight = Parameter(rng.normal(0.0, scale, size=(in_dim, out_dim)))
        self.bias = Parameter(np.zeros(out_dim))

    def forward(self, x):
        self.x = x
        return x @ self.weight.value + self.bias.value

    def backward(self, grad):
        self.weight.grad += self.x.T @ grad
        self.bias.grad += grad.sum(axis=0)
        return grad @ self.weight.value.T

    def parameters(self):
        return [self.weight, self.bias]


class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, grad):
        return grad * self.mask

    def parameters(self):
        return []


class Tanh:
    def forward(self, x):
        self.out = np.tanh(x)
        return self.out

    def backward(self, grad):
        return grad * (1.0 - self.out ** 2)

    def parameters(self):
        return []


class LSTM:
    def __init__(self, in_dim, hidden, rng):
        self.in_dim = in_dim
        self.hidden = hidden
        scale = 1.0 / np.sqrt(hidden)
        self.weight = Parameter(rng.uniform(-scale, scale, size=(in_dim + hidden, 4 * hidden)))
        self.bias = Parameter(np.zeros(4 * hidden))

    def forward(self, x):
        n, steps, _ = x.shape
        self.x = x
        self.steps = steps
        h = np.zeros((n, self.hidden))
        c = np.zeros((n, self.hidden))
        self.cache = []
        for t in range(steps):
            z = np.concatenate([x[:, t, :], h], axis=1)
            a = z @ self.weight.value + self.bias.value
            a_i, a_f, a_o, a_g = np.split(a, 4, axis=1)
            i = sigmoid(a_i)
            f = sigmoid(a_f)
            o = sigmoid(a_o)
            g = np.tanh(a_g)
            c_next = f * c + i * g
            tanh_c = np.tanh(c_next)
            h = o * tanh_c
            self.cache.append((z, i, f, o, g, c, tanh_c))
            c = c_next
        return h

    def backward(self, grad_h):
        n = self.x.shape[0]
        depth = self.in_dim
        d_weight = np.zeros_like(self.weight.value)
        d_bias = np.zeros_like(self.bias.value)
        d_x = np.zeros_like(self.x)
        d_c = np.zeros((n, self.hidden))
        for t in reversed(range(self.steps)):
            z, i, f, o, g, c_prev, tanh_c = self.cache[t]
            d_o = grad_h * tanh_c
            d_c = d_c + grad_h * o * (1.0 - tanh_c ** 2)
            d_f = d_c * c_prev
            d_i = d_c * g
            d_g = d_c * i
            d_c_prev = d_c * f
            a_i = d_i * i * (1.0 - i)
            a_f = d_f * f * (1.0 - f)
            a_o = d_o * o * (1.0 - o)
            a_g = d_g * (1.0 - g ** 2)
            grad_a = np.concatenate([a_i, a_f, a_o, a_g], axis=1)
            d_weight += z.T @ grad_a
            d_bias += grad_a.sum(axis=0)
            grad_z = grad_a @ self.weight.value.T
            d_x[:, t, :] = grad_z[:, :depth]
            grad_h = grad_z[:, depth:]
            d_c = d_c_prev
        self.weight.grad += d_weight
        self.bias.grad += d_bias
        return d_x

    def parameters(self):
        return [self.weight, self.bias]


class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def zero_grad(self):
        for param in self.parameters():
            param.grad = np.zeros_like(param.value)
