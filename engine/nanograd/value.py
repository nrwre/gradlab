"""Reverse-mode automatic differentiation over scalars.

A ``Value`` wraps a single float and records the operations applied to it so
that gradients can be propagated backwards through the resulting graph. This is
the natural generalisation of the hand-derived gradients from Experiment 1.
"""
from __future__ import annotations
import math
from typing import Callable, Set


class Value:
    __slots__ = ("data", "grad", "_backward", "_prev", "_op")

    def __init__(self, data: float, _children: tuple = (), _op: str = ""):
        self.data = float(data)
        self.grad = 0.0
        self._backward: Callable[[], None] = lambda: None
        self._prev: Set["Value"] = set(_children)
        self._op = _op

    # ---- core ops -------------------------------------------------------
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, p):
        assert isinstance(p, (int, float)), "only supports numeric powers"
        out = Value(self.data ** p, (self,), f"**{p}")

        def _backward():
            self.grad += (p * self.data ** (p - 1)) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(0.0 if self.data < 0 else self.data, (self,), "relu")

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t * t) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")

        def _backward():
            self.grad += e * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out

    # ---- backprop -------------------------------------------------------
    def backward(self):
        topo = []
        visited: Set["Value"] = set()

        def build(v: "Value"):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    # ---- sugar ----------------------------------------------------------
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other ** -1
    def __rtruediv__(self, other): return other * self ** -1
    def __repr__(self): return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"


def numerical_grad(f: Callable[[float], float], x: float, eps: float = 1e-6) -> float:
    """Central-difference gradient, used by the test-suite to verify autograd."""
    return (f(x + eps) - f(x - eps)) / (2 * eps)
