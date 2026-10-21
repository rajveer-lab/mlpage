"""Tiny models that score pages: logistic regression and a shallow decision tree.

Both are written with NumPy only, so the whole model is a handful of numbers
that can be saved as JSON and inspected by hand. ``predict_proba`` returns
the estimated probability that a page will NOT be needed soon.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .features import NUM_FEATURES

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.json"


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))


class LogisticModel:
    """Logistic regression: one weight per feature plus a bias."""

    kind = "logistic"

    def __init__(
        self,
        weights: Sequence[float] | None = None,
        bias: float = 0.0,
        features: Sequence[int] | None = None,
        meta: dict | None = None,
    ):
        self.features = list(range(NUM_FEATURES)) if features is None else list(features)
        self.weights = np.zeros(len(self.features)) if weights is None else np.asarray(weights, dtype=np.float64)
        self.bias = float(bias)
        self.meta = dict(meta or {})

    @property
    def n_params(self) -> int:
        return len(self.weights) + 1

    def fit(self, X: np.ndarray, y: np.ndarray, l2: float = 1e-4, iterations: int = 50) -> "LogisticModel":
        """Fit with Newton's method on standardised features, then fold the scaling into the weights."""
        Xf = X[:, self.features]
        mu = Xf.mean(axis=0)
        sd = Xf.std(axis=0)
        sd[sd == 0] = 1.0
        Z = np.column_stack(((Xf - mu) / sd, np.ones(len(Xf))))
        d = Z.shape[1]
        penalty = l2 * len(y) * np.r_[np.ones(d - 1), 0.0]
        beta = np.zeros(d)
        for _ in range(iterations):
            p = _sigmoid(Z @ beta)
            grad = Z.T @ (p - y) + penalty * beta
            hess = (Z * (p * (1 - p))[:, None]).T @ Z + np.diag(penalty) + 1e-9 * np.eye(d)
            step = np.linalg.solve(hess, grad)
            beta -= step
            if np.max(np.abs(step)) < 1e-7:
                break
        self.weights = beta[:-1] / sd
        self.bias = float(beta[-1] - np.sum(beta[:-1] * mu / sd))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return _sigmoid(X[:, self.features] @ self.weights + self.bias)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "features": self.features,
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LogisticModel":
        return cls(d["weights"], d["bias"], d["features"], d.get("meta"))


class TreeModel:
    """A shallow CART decision tree (Gini splits); leaves hold the positive rate."""

    kind = "tree"

    def __init__(self, nodes: dict | None = None, features: Sequence[int] | None = None, meta: dict | None = None):
        self.features = list(range(NUM_FEATURES)) if features is None else list(features)
        self.meta = dict(meta or {})
        nodes = nodes or {"feature": [-1], "threshold": [0.0], "left": [-1], "right": [-1], "value": [0.5]}
        self._set_nodes(nodes)

    def _set_nodes(self, nodes: dict) -> None:
        self.feature = np.asarray(nodes["feature"], dtype=np.int64)
        self.threshold = np.asarray(nodes["threshold"], dtype=np.float64)
        self.left = np.asarray(nodes["left"], dtype=np.int64)
        self.right = np.asarray(nodes["right"], dtype=np.int64)
        self.value = np.asarray(nodes["value"], dtype=np.float64)
        self.depth = self._depth(0) if len(self.feature) else 0

    def _depth(self, node: int) -> int:
        if self.feature[node] < 0:
            return 0
        return 1 + max(self._depth(self.left[node]), self._depth(self.right[node]))

    @property
    def n_params(self) -> int:
        internal = int(np.sum(self.feature >= 0))
        leaves = len(self.feature) - internal
        return 2 * internal + leaves  # (feature, threshold) per split, one value per leaf

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        max_depth: int = 4,
        min_leaf: int = 200,
        max_candidates: int = 256,
    ) -> "TreeModel":
        Xf = X[:, self.features]
        nodes: dict[str, list] = {"feature": [], "threshold": [], "left": [], "right": [], "value": []}

        def best_split(idx: np.ndarray) -> tuple[float, int, float] | None:
            ys = y[idx]
            n, pos = len(ys), ys.sum()
            parent = 2.0 * pos * (1.0 - pos / n)
            best = None
            for f in range(Xf.shape[1]):
                order = np.argsort(Xf[idx, f], kind="stable")
                xs = Xf[idx, f][order]
                cum = np.cumsum(ys[order])
                cut = np.nonzero(xs[1:] > xs[:-1])[0] + 1  # split before position cut
                cut = cut[(cut >= min_leaf) & (cut <= n - min_leaf)]
                if len(cut) == 0:
                    continue
                if len(cut) > max_candidates:
                    cut = cut[np.linspace(0, len(cut) - 1, max_candidates).astype(int)]
                nl = cut.astype(np.float64)
                pl = cum[cut - 1]
                nr = n - nl
                pr = pos - pl
                child = 2.0 * pl * (1.0 - pl / nl) + 2.0 * pr * (1.0 - pr / nr)
                i = int(np.argmin(child))
                gain = parent - child[i]
                if best is None or gain > best[0]:
                    best = (gain, f, float((xs[cut[i] - 1] + xs[cut[i]]) / 2.0))
            return best

        def build(idx: np.ndarray, depth: int) -> int:
            node = len(nodes["feature"])
            for key, init in (("feature", -1), ("threshold", 0.0), ("left", -1), ("right", -1)):
                nodes[key].append(init)
            nodes["value"].append(float(y[idx].mean()))
            if depth >= max_depth or len(idx) < 2 * min_leaf or nodes["value"][node] in (0.0, 1.0):
                return node
            split = best_split(idx)
            if split is None or split[0] <= 1e-9:
                return node
            _, f, thr = split
            mask = Xf[idx, f] <= thr
            nodes["feature"][node] = f
            nodes["threshold"][node] = thr
            nodes["left"][node] = build(idx[mask], depth + 1)
            nodes["right"][node] = build(idx[~mask], depth + 1)
            return node

        build(np.arange(len(y)), 0)
        self._set_nodes(nodes)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xf = X[:, self.features]
        rows = np.arange(len(Xf))
        node = np.zeros(len(Xf), dtype=np.int64)
        for _ in range(self.depth):
            f = self.feature[node]
            internal = f >= 0
            go_left = Xf[rows, np.maximum(f, 0)] <= self.threshold[node]
            node = np.where(internal, np.where(go_left, self.left[node], self.right[node]), node)
        return self.value[node]

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "features": self.features,
            "nodes": {
                "feature": self.feature.tolist(),
                "threshold": self.threshold.tolist(),
                "left": self.left.tolist(),
                "right": self.right.tolist(),
                "value": self.value.tolist(),
            },
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TreeModel":
        return cls(d["nodes"], d["features"], d.get("meta"))


MODEL_KINDS = {LogisticModel.kind: LogisticModel, TreeModel.kind: TreeModel}


def model_size_bytes(model) -> int:
    """Size of the parameters stored as float32."""
    return 4 * model.n_params


def save_model(model, path: str | Path = DEFAULT_MODEL_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model.to_dict(), indent=2) + "\n")
    return path


def load_model(path: str | Path | None = None):
    path = Path(path) if path is not None else DEFAULT_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"no trained model at {path}; run `python -m mlpr train` first")
    d = json.loads(path.read_text())
    return MODEL_KINDS[d["kind"]].from_dict(d)
