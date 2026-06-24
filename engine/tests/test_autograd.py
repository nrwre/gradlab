import math
import numpy as np
from nanograd.value import Value, numerical_grad


def test_add_mul_pow():
    a = Value(3.0); b = Value(4.0)
    c = a * b + a ** 2          # 3*4 + 9 = 21
    c.backward()
    assert abs(c.data - 21.0) < 1e-9
    assert abs(a.grad - (4.0 + 2 * 3.0)) < 1e-9   # b + 2a = 10
    assert abs(b.grad - 3.0) < 1e-9


def test_against_numerical():
    def f(x):
        v = Value(x)
        out = (v * v + v.exp()) * v.tanh()
        out.backward()
        return v.grad

    def fwd(x):
        v = Value(x)
        return ((v * v + v.exp()) * v.tanh()).data

    for x0 in [-1.3, 0.5, 2.0]:
        assert abs(f(x0) - numerical_grad(fwd, x0)) < 1e-4


def test_relu():
    a = Value(-2.0); b = a.relu(); b.backward()
    assert b.data == 0.0 and a.grad == 0.0
    a = Value(2.0); b = a.relu(); b.backward()
    assert b.data == 2.0 and a.grad == 1.0
