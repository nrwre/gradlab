"""Benchmark loss surfaces and toy 2-D datasets for the visualiser."""
from __future__ import annotations
import numpy as np


def quadratic_surface(kappa: float = 20.0):
    """Strongly-convex quadratic f(x) = 0.5 xᵀAx with condition number ``kappa``.

    Reproduces the Experiment 1 setup: A = diag(1, kappa), so the ratio of the
    largest to smallest eigenvalue (the condition number) is exactly ``kappa``.
    Returns (value_fn, grad_fn).
    """
    A = np.diag([1.0, float(kappa)])

    def value(x):
        x = np.asarray(x, dtype=float)
        return 0.5 * x @ A @ x

    def grad(x):
        x = np.asarray(x, dtype=float)
        return A @ x

    return value, grad


def rosenbrock(a: float = 1.0, b: float = 100.0):
    """The classic non-convex banana valley. Returns (value_fn, grad_fn)."""
    def value(x):
        x = np.asarray(x, dtype=float)
        return (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2

    def grad(x):
        x = np.asarray(x, dtype=float)
        dx = -2 * (a - x[0]) - 4 * b * x[0] * (x[1] - x[0] ** 2)
        dy = 2 * b * (x[1] - x[0] ** 2)
        return np.array([dx, dy])

    return value, grad


SURFACES = {"quadratic": quadratic_surface, "rosenbrock": rosenbrock}


# ---- toy classification datasets ---------------------------------------
def make_blobs(n=200, seed=0):
    rng = np.random.default_rng(seed)
    n2 = n // 2
    a = rng.normal([-2, -2], 0.8, size=(n2, 2))
    b = rng.normal([2, 2], 0.8, size=(n - n2, 2))
    X = np.vstack([a, b])
    y = np.array([0] * n2 + [1] * (n - n2))
    return X, y


def make_moons(n=200, noise=0.2, seed=0):
    rng = np.random.default_rng(seed)
    n2 = n // 2
    t = np.linspace(0, np.pi, n2)
    outer = np.stack([np.cos(t), np.sin(t)], axis=1)
    inner = np.stack([1 - np.cos(t), 1 - np.sin(t) - 0.5], axis=1)
    X = np.vstack([outer, inner]) + rng.normal(0, noise, size=(n2 * 2, 2))
    y = np.array([0] * n2 + [1] * n2)
    return X, y


def make_circles(n=200, noise=0.1, factor=0.4, seed=0):
    rng = np.random.default_rng(seed)
    n2 = n // 2
    t = np.linspace(0, 2 * np.pi, n2)
    outer = np.stack([np.cos(t), np.sin(t)], axis=1)
    inner = factor * np.stack([np.cos(t), np.sin(t)], axis=1)
    X = np.vstack([outer, inner]) + rng.normal(0, noise, size=(n2 * 2, 2))
    y = np.array([0] * n2 + [1] * n2)
    return X, y


DATASETS = {"blobs": make_blobs, "moons": make_moons, "circles": make_circles}
