import numpy as np
import pytest

from mlpr.features import CAP, FEATURE_NAMES, MetadataTable, compute_features
from mlpr.models import LogisticModel, TreeModel, load_model, save_model
from mlpr.training import TrainConfig, collect_samples
from mlpr.traces import zipf_trace


def test_features_by_hand():
    frames = 4
    # last=10, count=3, gaps 2 and 6 (third unknown), gap sum 8; now t=18.
    raw = np.array([[10, 3, 2, 6, -1, 8]], dtype=float)
    x = compute_features(18, raw, frames)[0]
    age = 8 / frames
    mean_gap = 4 / frames
    expected = np.log2(1 + np.array([age, 3, 2 / frames, 6 / frames, 6 / frames, mean_gap, age / mean_gap]))
    assert np.allclose(x, expected)


def test_unseen_gaps_are_censored_by_age():
    # A page seen once, 40 references ago: every gap is "at least 40".
    x = compute_features(50, np.array([[10, 1, -1, -1, -1, 0]], dtype=float), 8)[0]
    assert np.allclose(x[[2, 3, 4, 5]], np.log2(1 + 40 / 8))
    assert np.isclose(x[6], np.log2(2))
    # Values are capped.
    x = compute_features(10**9, np.array([[0, 1, -1, -1, -1, 0]], dtype=float), 1)[0]
    assert x.max() <= np.log2(1 + CAP)


def test_metadata_table_restores_ghosts():
    table = MetadataTable(2)
    table.load(7, 0, 0)
    table.touch(7, 3)
    table.evict(0)
    assert 7 in table.ghost
    table.load(7, 10, 1)
    record = table.records[1]
    assert record[:3] == [10, 3, 7]  # last, count, newest gap
    assert 7 not in table.ghost


def test_logistic_regression_learns_a_linear_rule():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, len(FEATURE_NAMES)))
    y = (X[:, 0] - 2 * X[:, 3] > 0).astype(float)
    model = LogisticModel().fit(X, y)
    assert np.mean((model.predict_proba(X) > 0.5) == y) > 0.97


def test_tree_learns_an_interaction():
    rng = np.random.default_rng(1)
    X = rng.uniform(-1, 1, size=(4000, len(FEATURE_NAMES)))
    y = ((X[:, 0] > 0) & (X[:, 1] > 0.3)).astype(float)  # needs two levels; XOR defeats greedy splits
    model = TreeModel().fit(X, y, max_depth=2, min_leaf=50)
    assert np.mean((model.predict_proba(X) > 0.5) == y) > 0.97


@pytest.mark.parametrize("model", [
    LogisticModel(weights=np.arange(7) / 10, bias=-0.3, meta={"k": 8}),
    TreeModel().fit(np.random.default_rng(2).normal(size=(500, 7)), (np.arange(500) % 2).astype(float), min_leaf=20),
])
def test_save_and_load_round_trip(tmp_path, model):
    X = np.random.default_rng(3).normal(size=(50, 7))
    path = save_model(model, tmp_path / "m.json")
    again = load_model(path)
    assert again.kind == model.kind
    assert np.allclose(again.predict_proba(X), model.predict_proba(X))
    assert again.meta == model.meta


def test_collect_samples_labels_come_from_the_future():
    trace = zipf_trace(300, 5000, seed=4)
    cfg = TrainConfig(k=8, evictions_per_run=10**9)
    X, y = collect_samples(trace, 20, cfg)
    assert X.shape[1] == len(FEATURE_NAMES)
    assert len(X) == len(y) > 0
    assert set(np.unique(y)) <= {0.0, 1.0}
    assert 0.05 < y.mean() < 0.95
