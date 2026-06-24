"""From-scratch Support Vector Machine (linear & Gaussian/RBF kernel).

Trained by sub-gradient descent on the soft-margin (hinge-loss) objective,
mirroring Experiment 3. Labels are internally mapped to {-1, +1}.
"""
from __future__ import annotations
import numpy as np


def _rbf_kernel(X, Z, gamma):
    sq = (X ** 2).sum(1)[:, None] + (Z ** 2).sum(1)[None, :] - 2 * X @ Z.T
    return np.exp(-gamma * np.maximum(sq, 0))


class SVM:
    def __init__(self, kernel="rbf", C=1.0, gamma=1.0, lr=0.01, epochs=300):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.where(np.asarray(y) <= 0, -1.0, 1.0)
        self.X_train = X
        self.y_train = y
        n = X.shape[0]
        if self.kernel == "rbf":
            K = _rbf_kernel(X, X, self.gamma)
            alpha = np.zeros(n)
            b = 0.0
            for _ in range(self.epochs):
                f = K @ (alpha * y) + b
                mask = (y * f) < 1
                grad_a = alpha - self.C * (K @ (mask * y) * y)
                alpha -= self.lr * grad_a / n
                b += self.lr * self.C * np.sum(mask * y) / n
            self.coef = alpha * y          # signed dual coefficients
            self.b = b
        else:  # linear
            w = np.zeros(X.shape[1])
            b = 0.0
            for _ in range(self.epochs):
                f = X @ w + b
                mask = (y * f) < 1
                grad_w = w - self.C * (X.T @ (mask * y))
                w -= self.lr * grad_w / n
                b += self.lr * self.C * np.sum(mask * y) / n
            self.w = w
            self.b = b
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        if self.kernel == "rbf":
            K = _rbf_kernel(X, self.X_train, self.gamma)
            return K @ self.coef + self.b
        return X @ self.w + self.b

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)
