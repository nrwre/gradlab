import numpy as np
from nanograd.optim import SGD, Momentum, Nesterov, Adam, optimise
from nanograd.problems import quadratic_surface, rosenbrock


def _final(opt, lr, surface, x0, steps=400):
    value, grad = surface
    traj = optimise(grad, x0, opt(lr=lr), n_steps=steps, value_fn=value)
    return value(traj[-1])


def test_all_converge_on_quadratic():
    surf = quadratic_surface(kappa=20.0)
    for opt, lr in [(SGD, 0.05), (Momentum, 0.02), (Nesterov, 0.02), (Adam, 0.2)]:
        f_end = _final(opt, lr, surf, [5.0, 5.0])
        assert f_end < 1e-2, f"{opt.name} failed to converge ({f_end})"


def test_adam_handles_rosenbrock():
    surf = rosenbrock()
    f_end = _final(Adam, 0.02, surf, [-1.0, 1.0], steps=4000)
    assert f_end < 1.0
