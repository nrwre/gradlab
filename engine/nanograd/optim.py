"""First-order optimisers, implemented from scratch.

All operate on plain NumPy parameter vectors and a gradient function, so they
can be used both for the 2-D visualiser surfaces and for training the neural
models. The update rules mirror Experiment 1 (gradient methods).
"""
from __future__ import annotations
import numpy as np
from typing import Callable, List


class Optimizer:
    name = "optimizer"

    def __init__(self, lr: float = 0.1):
        self.lr = lr

    def step(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def reset(self):
        pass


class SGD(Optimizer):
    name = "sgd"

    def step(self, x, grad):
        return x - self.lr * grad


class Momentum(Optimizer):
    name = "momentum"

    def __init__(self, lr=0.1, beta=0.9):
        super().__init__(lr)
        self.beta = beta
        self.v = None

    def reset(self):
        self.v = None

    def step(self, x, grad):
        if self.v is None:
            self.v = np.zeros_like(x)
        self.v = self.beta * self.v - self.lr * grad
        return x + self.v


class Nesterov(Optimizer):
    """Nesterov accelerated gradient. Expects ``grad_fn`` for the look-ahead."""
    name = "nesterov"

    def __init__(self, lr=0.1, beta=0.9):
        super().__init__(lr)
        self.beta = beta
        self.v = None

    def reset(self):
        self.v = None

    def step(self, x, grad, grad_fn: Callable[[np.ndarray], np.ndarray] | None = None):
        if self.v is None:
            self.v = np.zeros_like(x)
        look_ahead = x + self.beta * self.v
        g = grad_fn(look_ahead) if grad_fn is not None else grad
        self.v = self.beta * self.v - self.lr * g
        return x + self.v


class Adam(Optimizer):
    name = "adam"

    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(lr)
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.m = self.v = None
        self.t = 0

    def reset(self):
        self.m = self.v = None
        self.t = 0

    def step(self, x, grad):
        if self.m is None:
            self.m = np.zeros_like(x)
            self.v = np.zeros_like(x)
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        return x - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


OPTIMIZERS = {c.name: c for c in (SGD, Momentum, Nesterov, Adam)}


def optimise(grad_fn, x0, optimizer: Optimizer, n_steps: int = 100,
             value_fn: Callable | None = None) -> List[np.ndarray]:
    """Run an optimiser and return the trajectory of parameter vectors."""
    optimizer.reset()
    x = np.array(x0, dtype=float)
    traj = [x.copy()]
    for _ in range(n_steps):
        g = grad_fn(x)
        if isinstance(optimizer, Nesterov):
            x = optimizer.step(x, g, grad_fn=grad_fn)
        else:
            x = optimizer.step(x, g)
        traj.append(x.copy())
    return traj
