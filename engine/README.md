# nanograd — the GradLab engine

A small, from-scratch deep-learning and classical-ML library in NumPy.
It is the compute core behind the GradLab playground.

- **Autograd** (`nanograd.value.Value`) — reverse-mode automatic differentiation over a scalar graph.
- **Optimisers** (`nanograd.optim`) — `SGD`, `Momentum`, `Nesterov`, `Adam`.
- **Neural nets** (`nanograd.nn`) — `Linear`, `MLP`, MSE / softmax-cross-entropy losses.
- **Classical models** — `SVM` (linear + RBF), `DecisionTree`, `RandomForest` (bagging + OOB).
- **Benchmarks** (`nanograd.problems`) — quadratic & Rosenbrock surfaces, `blobs` / `moons` / `circles`.

```bash
pip install -e ".[dev]"
pytest -q
```

```python
from nanograd import quadratic_surface, Adam
from nanograd.optim import optimise
value, grad = quadratic_surface(kappa=20)
traj = optimise(grad, [5, 5], Adam(lr=0.2), n_steps=200)
print("final loss:", value(traj[-1]))
```
