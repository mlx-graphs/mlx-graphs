import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch
from mlx.utils import tree_flatten
from torch_geometric.nn import EdgeConv as PyGEdgeConv

from mlx_graphs.nn import DynamicEdgeConv, EdgeConv

# MLX's GPU matmul is only accurate to ~1e-3 relative on some machines, while a
# wrong formula gives O(1) errors with these O(1) inputs.
TOL = dict(rtol=2e-2, atol=2e-2)

DIM, OUT_DIM, K = 3, 8, 4
BATCH = np.repeat(np.arange(3), [6, 3, 7]).astype(np.int64)

# PyG's DynamicEdgeConv needs pyg-lib for its k-NN search, which is not a test
# dependency: it is reproduced with PyG's EdgeConv on a brute-force k-NN graph.


def _knn_edges(x, y, k, bx, by):
    """Edges ``j -> i`` from the k nearest points ``j`` of x to each point ``i``
    of y (including itself), as in PyG's DynamicEdgeConv."""
    dist = np.linalg.norm(y[:, None, :] - x[None, :, :], axis=-1)
    src, dst = [], []
    for i in range(y.shape[0]):
        candidates = np.nonzero(bx == by[i])[0]
        nearest = candidates[np.argsort(dist[i, candidates], kind="stable")[:k]]
        src += nearest.tolist()
        dst += [i] * len(nearest)
    return np.array([src, dst], dtype=np.int64)


def _make_mlp():
    return nn.Sequential(nn.Linear(2 * DIM, 16), nn.ReLU(), nn.Linear(16, OUT_DIM))


def _make_pyg_conv(mlp, **kwargs):
    torch_mlp = torch.nn.Sequential(
        torch.nn.Linear(2 * DIM, 16), torch.nn.ReLU(), torch.nn.Linear(16, OUT_DIM)
    )
    pyg_conv = PyGEdgeConv(torch_mlp, **kwargs)
    with torch.no_grad():
        for idx in (0, 2):
            torch_mlp[idx].weight.copy_(
                torch.from_numpy(np.array(mlp.layers[idx].weight))
            )
            torch_mlp[idx].bias.copy_(torch.from_numpy(np.array(mlp.layers[idx].bias)))
    return pyg_conv


@pytest.mark.parametrize("aggr", ["max", "mean", "add"])
def test_dynamic_edge_conv_matches_pyg(aggr):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(len(BATCH), DIM)).astype(np.float32)
    mlp = _make_mlp()
    pyg_conv = _make_pyg_conv(mlp, aggr=aggr)
    conv = DynamicEdgeConv(mlp, k=K, aggr=aggr)
    assert isinstance(conv, EdgeConv)

    edge_index = _knn_edges(x, x, K, BATCH, BATCH)
    with torch.no_grad():
        expected = pyg_conv(torch.from_numpy(x), torch.from_numpy(edge_index))
    out = conv(mx.array(x), mx.array(BATCH))
    np.testing.assert_allclose(np.array(out), expected.numpy(), **TOL)

    # without batch: a single graph
    zeros = np.zeros_like(BATCH)
    edge_index = _knn_edges(x, x, K, zeros, zeros)
    with torch.no_grad():
        expected = pyg_conv(torch.from_numpy(x), torch.from_numpy(edge_index))
    np.testing.assert_allclose(np.array(conv(mx.array(x))), expected.numpy(), **TOL)


def test_dynamic_edge_conv_bipartite():
    rng = np.random.default_rng(1)
    x_src = rng.normal(size=(len(BATCH), DIM)).astype(np.float32)
    batch_dst = np.array([0, 0, 1, 2, 2], dtype=np.int64)
    x_dst = rng.normal(size=(len(batch_dst), DIM)).astype(np.float32)
    mlp = _make_mlp()
    pyg_conv = _make_pyg_conv(mlp)
    conv = DynamicEdgeConv(mlp, k=K)

    edge_index = _knn_edges(x_src, x_dst, K, BATCH, batch_dst)
    with torch.no_grad():
        expected = pyg_conv(
            (torch.from_numpy(x_src), torch.from_numpy(x_dst)),
            torch.from_numpy(edge_index),
        )
    out = conv(
        (mx.array(x_src), mx.array(x_dst)), (mx.array(BATCH), mx.array(batch_dst))
    )
    assert out.shape == (len(batch_dst), OUT_DIM)
    np.testing.assert_allclose(np.array(out), expected.numpy(), **TOL)


def test_dynamic_edge_conv_gradients():
    rng = np.random.default_rng(2)
    x = mx.array(rng.normal(size=(len(BATCH), DIM)).astype(np.float32))
    conv = DynamicEdgeConv(_make_mlp(), k=K)

    def loss_fn(params, x):
        conv.update(params)
        return (conv(x, mx.array(BATCH)) ** 2).sum()

    grads, x_grad = mx.grad(loss_fn, argnums=(0, 1))(conv.trainable_parameters(), x)
    for name, grad in tree_flatten(grads):
        assert mx.all(mx.isfinite(grad)).item(), name
        assert mx.any(grad != 0).item(), name
    assert mx.any(x_grad != 0).item()


def test_dynamic_edge_conv_rejects_static_graph():
    conv = DynamicEdgeConv(_make_mlp(), k=K)
    with pytest.raises(ValueError):
        conv(mx.zeros((4,)))
    assert repr(conv).startswith("DynamicEdgeConv(")
    assert repr(conv).endswith(f"k={K})")
