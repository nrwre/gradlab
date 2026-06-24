"""From-scratch decision tree and random forest (CART-style, Gini impurity).

Mirrors Experiments 4 and 5: a recursive tree builder with Gini-based splits,
bootstrap aggregation, feature bagging and out-of-bag scoring.
"""
from __future__ import annotations
import numpy as np


def _gini(y):
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return 1.0 - np.sum(p ** 2)


class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "value")

    def __init__(self):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None


class DecisionTree:
    def __init__(self, max_depth=6, min_samples=2, n_features=None, rng=None):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.n_features = n_features          # feature bagging (for the forest)
        self.rng = rng or np.random.default_rng(0)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.n_classes = int(y.max()) + 1
        self.root = self._build(X, y, 0)
        return self

    def _build(self, X, y, depth):
        node = _Node()
        if (depth >= self.max_depth or len(y) < self.min_samples
                or len(np.unique(y)) == 1):
            node.value = np.bincount(y, minlength=self.n_classes).argmax()
            return node
        feat, thr, gain = self._best_split(X, y)
        if feat is None or gain <= 0:
            node.value = np.bincount(y, minlength=self.n_classes).argmax()
            return node
        left = X[:, feat] <= thr
        node.feature, node.threshold = feat, thr
        node.left = self._build(X[left], y[left], depth + 1)
        node.right = self._build(X[~left], y[~left], depth + 1)
        return node

    def _best_split(self, X, y):
        n_feat = X.shape[1]
        feats = range(n_feat)
        if self.n_features is not None and self.n_features < n_feat:
            feats = self.rng.choice(n_feat, self.n_features, replace=False)
        parent = _gini(y)
        best = (None, None, 0.0)
        for f in feats:
            for thr in np.unique(X[:, f]):
                left = X[:, f] <= thr
                if left.sum() == 0 or (~left).sum() == 0:
                    continue
                nl, nr = left.sum(), (~left).sum()
                child = (nl * _gini(y[left]) + nr * _gini(y[~left])) / len(y)
                gain = parent - child
                if gain > best[2]:
                    best = (f, thr, gain)
        return best

    def _predict_one(self, x, node):
        while node.value is None:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(x, self.root) for x in X])


class RandomForest:
    def __init__(self, n_trees=15, max_depth=6, sample_frac=0.8,
                 n_features="sqrt", seed=0):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.sample_frac = sample_frac
        self.n_features = n_features
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        nf = int(np.sqrt(d)) if self.n_features == "sqrt" else (self.n_features or d)
        self.trees = []
        self.oob_votes = np.zeros((n, int(y.max()) + 1))
        m = max(1, int(self.sample_frac * n))
        for t in range(self.n_trees):
            idx = rng.choice(n, m, replace=True)
            oob = np.setdiff1d(np.arange(n), idx)
            tree = DecisionTree(max_depth=self.max_depth, n_features=nf,
                                rng=np.random.default_rng(self.seed + t + 1))
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)
            if len(oob):
                for i, p in zip(oob, tree.predict(X[oob])):
                    self.oob_votes[i, p] += 1
        voted = self.oob_votes.sum(1) > 0
        self.oob_score_ = float(np.mean(
            self.oob_votes[voted].argmax(1) == y[voted])) if voted.any() else None
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        preds = np.stack([t.predict(X) for t in self.trees])
        out = np.zeros(X.shape[0], dtype=int)
        for i in range(X.shape[0]):
            out[i] = np.bincount(preds[:, i]).argmax()
        return out
