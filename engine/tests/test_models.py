import numpy as np
from nanograd.problems import make_blobs, make_moons, make_circles
from nanograd.svm import SVM
from nanograd.forest import DecisionTree, RandomForest
from nanograd.nn import MLP
from nanograd.optim import Adam


def _acc(model, X, y):
    return float(np.mean(model.predict(X) == y))


def test_svm_linear_blobs():
    X, y = make_blobs(200, seed=1)
    svm = SVM(kernel="linear", C=1.0, epochs=400).fit(X, y)
    assert _acc(svm, X, y) > 0.95


def test_svm_rbf_circles():
    X, y = make_circles(200, noise=0.08, seed=1)
    svm = SVM(kernel="rbf", gamma=1.0, C=1.0, epochs=500).fit(X, y)
    assert _acc(svm, X, y) > 0.85


def test_tree_and_forest_moons():
    X, y = make_moons(300, noise=0.2, seed=2)
    tree = DecisionTree(max_depth=6).fit(X, y)
    forest = RandomForest(n_trees=15, max_depth=6, seed=2).fit(X, y)
    assert _acc(tree, X, y) > 0.85
    assert _acc(forest, X, y) > 0.85
    assert forest.oob_score_ is None or 0.0 <= forest.oob_score_ <= 1.0


def test_mlp_moons():
    X, y = make_moons(300, noise=0.2, seed=3)
    mlp = MLP(2, n_hidden=24, n_out=2, seed=0)
    hist = mlp.fit(X, y, Adam(lr=0.05), epochs=400)
    assert hist[-1] < hist[0]
    assert _acc(mlp, X, y) > 0.85
