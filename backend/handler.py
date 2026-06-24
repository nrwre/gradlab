"""AWS Lambda handler for GradLab.

A single function routes three POST endpoints (works with a Lambda Function URL
or API Gateway proxy integration):

    POST /optimize  -> run an optimiser on a surface, return the trajectory
    POST /classify  -> train a from-scratch model, return a decision-boundary grid
    POST /explain   -> AI tutor explanation of a run

Run locally with ``python scripts/run_local.py`` (see that file).
"""
from __future__ import annotations
import json
import sys
import os
import numpy as np

# Make the engine importable whether running in Lambda (layer/vendored) or locally.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "engine"))

from nanograd.optim import OPTIMIZERS, optimise          # noqa: E402
from nanograd.problems import SURFACES, DATASETS          # noqa: E402
from nanograd.svm import SVM                              # noqa: E402
from nanograd.forest import DecisionTree, RandomForest    # noqa: E402
from nanograd.nn import MLP                               # noqa: E402
from nanograd.optim import Adam                           # noqa: E402
import ai_tutor                                           # noqa: E402

CORS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Content-Type": "application/json",
}


def _resp(body, code=200):
    return {"statusCode": code, "headers": CORS, "body": json.dumps(body)}


# ---- endpoint logic -----------------------------------------------------
def run_optimize(p):
    surface = p.get("surface", "quadratic")
    kappa = float(p.get("kappa", 20.0))
    opt_name = p.get("optimizer", "sgd")
    lr = float(p.get("lr", 0.05))
    steps = int(p.get("steps", 100))
    start = p.get("start", [4.0, 4.0])

    value, grad = (SURFACES[surface](kappa) if surface == "quadratic"
                   else SURFACES[surface]())
    opt = OPTIMIZERS[opt_name](lr=lr)
    traj = optimise(grad, start, opt, n_steps=steps, value_fn=value)
    losses = [float(value(x)) for x in traj]
    return {
        "trajectory": [[float(x[0]), float(x[1])] for x in traj],
        "losses": losses,
        "final_loss": losses[-1],
        "start_loss": losses[0],
    }


def run_classify(p):
    dataset = p.get("dataset", "moons")
    model_name = p.get("model", "svm")
    noise = float(p.get("noise", 0.2))
    X, y = DATASETS[dataset](int(p.get("n", 250)), seed=int(p.get("seed", 0)),
                             **({"noise": noise} if dataset != "blobs" else {}))
    if model_name == "svm":
        model = SVM(kernel=p.get("kernel", "rbf"), C=float(p.get("C", 1.0)),
                    gamma=float(p.get("gamma", 1.0)), epochs=int(p.get("epochs", 400)))
        model.fit(X, y)
    elif model_name == "tree":
        model = DecisionTree(max_depth=int(p.get("max_depth", 6))).fit(X, y)
    elif model_name == "forest":
        model = RandomForest(n_trees=int(p.get("n_trees", 15)),
                             max_depth=int(p.get("max_depth", 6))).fit(X, y)
    else:  # mlp
        model = MLP(2, n_hidden=int(p.get("hidden", 24)), n_out=2)
        model.fit(X, y, Adam(lr=0.05), epochs=int(p.get("epochs", 400)))

    # decision-boundary grid
    g = 50
    xs = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, g)
    ys = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, g)
    gx, gy = np.meshgrid(xs, ys)
    grid = np.c_[gx.ravel(), gy.ravel()]
    zz = model.predict(grid).reshape(g, g)
    acc = float(np.mean(model.predict(X) == y))
    out = {
        "points": X.tolist(), "labels": y.tolist(),
        "grid_x": xs.tolist(), "grid_y": ys.tolist(), "grid_z": zz.tolist(),
        "accuracy": acc,
    }
    if model_name == "forest":
        out["oob_score"] = model.oob_score_
    return out


def run_explain(p):
    return ai_tutor.explain(p.get("run", {}))


ROUTES = {"/optimize": run_optimize, "/classify": run_classify, "/explain": run_explain}


def handler(event, context=None):
    method = (event.get("requestContext", {}).get("http", {}).get("method")
              or event.get("httpMethod", "POST"))
    if method == "OPTIONS":
        return _resp({"ok": True})
    path = (event.get("rawPath") or event.get("path") or "/").rstrip("/") or "/"
    # allow stage prefixes like /default/optimize
    route = next((r for r in ROUTES if path.endswith(r)), None)
    if route is None:
        return _resp({"error": f"unknown path {path}"}, 404)
    try:
        body = event.get("body") or "{}"
        params = json.loads(body) if isinstance(body, str) else body
        return _resp(ROUTES[route](params or {}))
    except Exception as e:
        return _resp({"error": str(e)}, 400)
