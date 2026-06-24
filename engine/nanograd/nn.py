"""Minimal neural-network layers and losses built on NumPy.

Kept deliberately small: a Linear layer and an MLP are enough to power the
classifier playground and to demonstrate that the optimisers train real models.
Gradients here are derived analytically (the autograd Value class covers the
scalar-graph demonstration; for batched training we use vectorised math).
"""
from __future__ import annotations
import numpy as np


def _xavier(fan_in, fan_out, rng):
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=(fan_in, fan_out))


class Linear:
    def __init__(self, fan_in, fan_out, rng=None):
        rng = rng or np.random.default_rng(0)
        self.W = _xavier(fan_in, fan_out, rng)
        self.b = np.zeros(fan_out)

    def __call__(self, X):
        self._X = X
        return X @ self.W + self.b


def relu(x):
    return np.maximum(0, x)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def mse_loss(pred, target):
    return float(np.mean((pred - target) ** 2))


def softmax_cross_entropy(logits, y):
    """Returns (loss, dlogits). ``y`` is an int array of class labels."""
    p = softmax(logits)
    n = logits.shape[0]
    loss = -np.mean(np.log(p[np.arange(n), y] + 1e-12))
    d = p.copy()
    d[np.arange(n), y] -= 1
    return loss, d / n


class MLP:
    """One hidden layer (ReLU) classifier trained with the engine's optimisers."""

    def __init__(self, n_in, n_hidden=16, n_out=2, seed=0):
        rng = np.random.default_rng(seed)
        self.l1 = Linear(n_in, n_hidden, rng)
        self.l2 = Linear(n_hidden, n_out, rng)

    def _params(self):
        return [self.l1.W, self.l1.b, self.l2.W, self.l2.b]

    def _flatten(self):
        return np.concatenate([p.ravel() for p in self._params()])

    def _unflatten(self, vec):
        i = 0
        for p in self._params():
            n = p.size
            p[...] = vec[i:i + n].reshape(p.shape)
            i += n

    def forward(self, X):
        self.z1 = X @ self.l1.W + self.l1.b
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.l2.W + self.l2.b
        return self.z2

    def _grads(self, X, y):
        logits = self.forward(X)
        loss, dz2 = softmax_cross_entropy(logits, y)
        dW2 = self.a1.T @ dz2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ self.l2.W.T
        dz1 = da1 * (self.z1 > 0)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0)
        grad = np.concatenate([dW1.ravel(), db1.ravel(), dW2.ravel(), db2.ravel()])
        return loss, grad

    def fit(self, X, y, optimizer, epochs=200):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        optimizer.reset()
        vec = self._flatten()
        history = []
        for _ in range(epochs):
            self._unflatten(vec)
            loss, grad = self._grads(X, y)
            vec = optimizer.step(vec, grad)
            history.append(loss)
        self._unflatten(vec)
        return history

    def predict(self, X):
        return self.forward(np.asarray(X, dtype=float)).argmax(axis=1)

    def predict_proba(self, X):
        return softmax(self.forward(np.asarray(X, dtype=float)))
