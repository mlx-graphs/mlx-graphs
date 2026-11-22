import importlib
import types

import mlx.core as mx
import numpy as np
import pytest

from mlx_graphs.utils.assortativity import assortativity
from mlx_graphs.utils.normalized_cut import normalized_cut
from mlx_graphs.utils.ppr import get_ppr


def random_edges(seed: int, num_nodes: int = 10, num_edges: int = 30) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, num_nodes, size=(2, num_edges))


@pytest.mark.parametrize("seed", range(5))
def test_assortativity_matches_pyg(seed):
    torch = pytest.importorskip("torch")
    pyg_utils = pytest.importorskip("torch_geometric.utils")

    edges = random_edges(seed)
    out = assortativity(mx.array(edges))
    expected = pyg_utils.assortativity(torch.tensor(edges))
    assert out == pytest.approx(expected, abs=1e-5)


def test_assortativity_examples():
    # Star graph: hubs connect to leaves only.
    star = mx.array([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    assert assortativity(star) == pytest.approx(-1.0)
    assert assortativity(mx.array([[0, 1, 2, 3, 2], [1, 2, 0, 1, 3]])) == pytest.approx(
        -2 / 3
    )
    assert np.isnan(assortativity(mx.zeros((2, 0), dtype=mx.int32)))


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("weights_shape", [(), (2,)])
def test_normalized_cut_matches_pyg(seed, weights_shape):
    torch = pytest.importorskip("torch")
    pyg_utils = pytest.importorskip("torch_geometric.utils")

    edges = random_edges(seed)
    rng = np.random.default_rng(seed)
    weights = rng.random(size=(edges.shape[1], *weights_shape)).astype(np.float32)
    out = normalized_cut(mx.array(edges), mx.array(weights), num_nodes=12)
    if weights.ndim == 1:
        expected = pyg_utils.normalized_cut(
            torch.tensor(edges), torch.tensor(weights), num_nodes=12
        ).numpy()
    else:
        expected = np.stack(
            [
                pyg_utils.normalized_cut(
                    torch.tensor(edges), torch.tensor(weights[:, i]), num_nodes=12
                ).numpy()
                for i in range(weights.shape[1])
            ],
            axis=1,
        )
    assert out.shape == weights.shape
    assert np.allclose(np.asarray(out), expected, atol=1e-6)

    integer = normalized_cut(mx.array(edges), mx.ones((edges.shape[1],), mx.int32))
    assert integer.dtype == mx.float32


def _pyg_ppr(monkeypatch, edges: np.ndarray, num_nodes: int, alpha, eps, target):
    """Runs PyG's pure-Python PPR push (normally compiled with numba)."""
    ppr = importlib.import_module("torch_geometric.utils.ppr")
    numba = types.SimpleNamespace(prange=range, int64=int)
    monkeypatch.setattr(ppr, "numba", numba, raising=False)
    order = np.lexsort((edges[1], edges[0]))
    row, col = edges[0][order], edges[1][order]
    rowptr = np.concatenate([[0], np.cumsum(np.bincount(row, minlength=num_nodes))])
    return ppr._get_ppr(rowptr, col, alpha, eps, target)


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("use_target", [False, True])
def test_get_ppr_matches_pyg_algorithm(monkeypatch, seed, use_target):
    pytest.importorskip("torch_geometric")

    num_nodes = 10
    edges = np.unique(random_edges(seed, num_nodes, 40), axis=1)  # sorted
    target = np.array([3, 0, 7]) if use_target else None
    alpha, eps = 0.15, 1e-4

    ppr_index, ppr_weights = get_ppr(
        mx.array(edges),
        alpha,
        eps,
        None if target is None else mx.array(target),
        num_nodes,
    )
    js, vals = _pyg_ppr(monkeypatch, edges, num_nodes, alpha, eps, target)
    sources = np.arange(num_nodes) if target is None else target
    exp_row = np.concatenate([np.full(len(j), s) for s, j in zip(sources, js)])
    exp_col = np.concatenate([np.asarray(j) for j in js])
    exp_weights = np.concatenate([np.asarray(v) for v in vals])

    assert np.array_equal(np.asarray(ppr_index), np.stack([exp_row, exp_col]))
    assert np.allclose(np.asarray(ppr_weights), exp_weights, atol=1e-6)


def test_get_ppr_approximates_exact_ppr():
    # Exact PPR: pi_s = alpha * e_s @ inv(I - (1 - alpha) * P).
    num_nodes, alpha = 8, 0.2
    edges = np.unique(random_edges(0, num_nodes, 30), axis=1)
    edges = edges[:, edges[0] != edges[1]]
    adj = np.zeros((num_nodes, num_nodes))
    adj[edges[0], edges[1]] = 1
    trans = adj / np.maximum(adj.sum(1, keepdims=True), 1)
    exact = alpha * np.linalg.inv(np.eye(num_nodes) - (1 - alpha) * trans)

    ppr_index, ppr_weights = get_ppr(mx.array(edges), alpha, eps=1e-7)
    approx = np.zeros((num_nodes, num_nodes))
    idx = np.asarray(ppr_index)
    approx[idx[0], idx[1]] = np.asarray(ppr_weights)
    has_out = adj.sum(1) > 0
    assert np.allclose(approx[has_out], exact[has_out], atol=1e-3)

    empty_index, empty_weights = get_ppr(mx.array(edges), target=mx.array([], mx.int32))
    assert empty_index.shape == (2, 0) and empty_weights.shape == (0,)
