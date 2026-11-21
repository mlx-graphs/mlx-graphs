"""Parity tests of mlx_graphs.metrics against torch_geometric.metrics (and
scikit-learn for AUC / average precision)."""

import mlx.core as mx
import numpy as np
import pytest

from mlx_graphs import metrics as M

torch = pytest.importorskip("torch")
pyg_metrics = pytest.importorskip("torch_geometric.metrics")

NUM_ITEMS, K = 15, 5


def _batch(seed, batch_size=8, num_labels=20, weighted=False, empty_rows=True):
    rng = np.random.default_rng(seed)
    pred = np.stack([rng.permutation(NUM_ITEMS)[: K + 2] for _ in range(batch_size)])
    rows = rng.integers(0, batch_size - (2 if empty_rows else 0), num_labels)
    cols = rng.integers(0, NUM_ITEMS, num_labels)
    # unique (row, col) pairs, as ground truth sets
    pairs = np.unique(np.stack([rows, cols], axis=1), axis=0)
    label_index = pairs.T
    weight = None
    if weighted:
        weight = rng.choice([-1.0, 0.5, 1.0, 2.0, 3.0], label_index.shape[1])
    return pred, label_index, weight


def _pyg(array):
    return None if array is None else torch.tensor(array)


UNWEIGHTED = [
    ("LinkPredPrecision", {}),
    ("LinkPredRecall", {}),
    ("LinkPredF1", {}),
    ("LinkPredMAP", {}),
    ("LinkPredNDCG", {}),
    ("LinkPredMRR", {}),
    ("LinkPredHitRatio", {}),
]
WEIGHTED = [
    ("LinkPredRecall", {"weighted": True}),
    ("LinkPredNDCG", {"weighted": True}),
]


@pytest.mark.parametrize("name, kwargs", UNWEIGHTED + WEIGHTED)
@pytest.mark.parametrize("k", [1, 3, K, K + 4])
def test_link_pred_metric_matches_pyg(name, kwargs, k):
    metric = getattr(M, name)(k=k, **kwargs)
    ref = getattr(pyg_metrics, name)(k=k, **kwargs)
    weighted = kwargs.get("weighted", False)
    for seed in range(3):
        pred, label_index, weight = _batch(seed, weighted=weighted)
        metric.update(mx.array(pred), mx.array(label_index), weight)
        ref.update(_pyg(pred), _pyg(label_index), _pyg(weight))
        assert np.isclose(metric.compute().item(), ref.compute().item(), atol=1e-6)
    # tuple ground truth and numpy inputs
    pred, label_index, weight = _batch(5, weighted=weighted)
    metric.update(pred, (label_index[0], label_index[1]), weight)
    ref.update(_pyg(pred), _pyg(label_index), _pyg(weight))
    assert np.isclose(metric.compute().item(), ref.compute().item(), atol=1e-6)
    metric.reset()
    assert metric.compute().item() == 0.0
    assert repr(metric) == repr(ref)


def test_weighted_metric_needs_weights():
    with pytest.raises(ValueError):
        M.LinkPredNDCG(k=2, weighted=True).update(mx.array([[0, 1]]), [[0], [1]])
    with pytest.raises(ValueError):
        M.LinkPredPrecision(k=0)


def test_empty_ground_truth():
    metric, ref = M.LinkPredMAP(k=3), pyg_metrics.LinkPredMAP(k=3)
    pred = np.array([[0, 1, 2]])
    empty = np.zeros((2, 0), dtype=np.int64)
    metric.update(pred, empty)
    ref.update(_pyg(pred), _pyg(empty))
    assert metric.compute().item() == ref.compute().item() == 0.0


def test_collection_matches_pyg():
    metrics = [
        M.LinkPredMAP(k=3),
        M.LinkPredPrecision(k=K),
        M.LinkPredNDCG(k=4, weighted=True),
        M.LinkPredRecall(k=2),
        M.LinkPredCoverage(k=2, num_dst_nodes=NUM_ITEMS),
    ]
    refs = [
        pyg_metrics.LinkPredMAP(k=3),
        pyg_metrics.LinkPredPrecision(k=K),
        pyg_metrics.LinkPredNDCG(k=4, weighted=True),
        pyg_metrics.LinkPredRecall(k=2),
        pyg_metrics.LinkPredCoverage(k=2, num_dst_nodes=NUM_ITEMS),
    ]
    collection = M.LinkPredMetricCollection(metrics)
    ref = pyg_metrics.LinkPredMetricCollection(refs)
    assert list(collection.keys()) == list(ref.keys())
    assert collection.max_k == ref.max_k and collection.weighted
    for seed in range(3):
        pred, label_index, weight = _batch(seed, weighted=True)
        collection.update(pred, label_index, weight)
        ref.update(_pyg(pred), _pyg(label_index), _pyg(weight))
    out, expected = collection.compute(), ref.compute()
    for name in expected:
        assert np.isclose(out[name].item(), expected[name].item(), atol=1e-6), name
    collection.reset()
    assert all(v.item() == 0.0 for v in collection.compute().values())


@pytest.mark.parametrize("k", [2, K])
def test_coverage_diversity_popularity_match_pyg(k):
    rng = np.random.default_rng(0)
    category = rng.integers(0, 3, NUM_ITEMS)
    popularity = rng.integers(0, 50, NUM_ITEMS)
    pairs = [
        (M.LinkPredCoverage(k, NUM_ITEMS), pyg_metrics.LinkPredCoverage(k, NUM_ITEMS)),
        (
            M.LinkPredDiversity(k, mx.array(category)),
            pyg_metrics.LinkPredDiversity(k, torch.tensor(category)),
        ),
        (
            M.LinkPredAveragePopularity(k, mx.array(popularity)),
            pyg_metrics.LinkPredAveragePopularity(k, torch.tensor(popularity)),
        ),
    ]
    for metric, ref in pairs:
        for seed in range(3):
            pred, label_index, _ = _batch(seed)
            metric.update(mx.array(pred), mx.array(label_index))
            ref.update(_pyg(pred), _pyg(label_index))
        assert np.isclose(metric.compute().item(), ref.compute().item(), atol=1e-6)


@pytest.mark.parametrize("max_src_nodes, batch_size", [(None, 7), (10, 2**16), (3, 1)])
def test_personalization_matches_pyg(max_src_nodes, batch_size):
    metric = M.LinkPredPersonalization(3, max_src_nodes, batch_size)
    ref = pyg_metrics.LinkPredPersonalization(3, max_src_nodes, batch_size)
    for seed in range(3):
        pred, label_index, _ = _batch(seed)
        metric.update(pred, label_index)
        ref.update(_pyg(pred), _pyg(label_index))
    assert np.isclose(metric.compute().item(), ref.compute().item(), atol=1e-6)


@pytest.mark.parametrize("seed", range(4))
def test_auc_and_average_precision_match_sklearn(seed):
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, 50)
    labels[:2] = [0, 1]
    # rounded scores give ties
    scores = np.round(rng.random(50), 1)
    auc = M.roc_auc_score(mx.array(labels), mx.array(scores))
    ap = M.average_precision_score(labels, scores)
    assert np.isclose(auc, sklearn_metrics.roc_auc_score(labels, scores))
    assert np.isclose(ap, sklearn_metrics.average_precision_score(labels, scores))
    with pytest.raises(ValueError):
        M.roc_auc_score(np.ones(3), scores[:3])
