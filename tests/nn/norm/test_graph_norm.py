import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch
from mlx.utils import tree_flatten
from torch_geometric.nn import GraphNorm as PyGGraphNorm

import mlx_graphs.nn as gnn
from mlx_graphs.nn import GraphNorm, GraphNormalization

TOL = dict(rtol=1e-4, atol=1e-4)

DIM = 5
# 3 graphs with 4, 1 and 6 nodes, nodes of graphs interleaved
BATCH = np.array([0, 2, 0, 1, 2, 2, 0, 2, 2, 0, 2], dtype=np.int64)


def test_aliases():
    assert GraphNorm is GraphNormalization
    assert gnn.BatchNorm is gnn.BatchNormalization
    assert gnn.LayerNorm is gnn.LayerNormalization
    assert gnn.InstanceNorm is gnn.InstanceNormalization
    assert gnn.HeteroBatchNorm is gnn.HeteroBatchNormalization
    assert gnn.HeteroLayerNorm is gnn.HeteroLayerNormalization
    assert gnn.PairNorm is gnn.PairNormalization
    assert gnn.GraphSizeNorm is gnn.GraphSizeNormalization
    assert gnn.MeanSubtractionNorm is gnn.MeanSubtractionNormalization
    assert gnn.MessageNorm is gnn.MessageNormalization
    assert gnn.DiffGroupNorm is gnn.DiffGroupNormalization


@pytest.mark.parametrize("batched", [False, True])
def test_graph_norm_matches_pyg(batched):
    rng = np.random.default_rng(0)
    x = rng.normal(loc=2.0, size=(len(BATCH), DIM)).astype(np.float32)
    weight = rng.normal(size=DIM).astype(np.float32)
    bias = rng.normal(size=DIM).astype(np.float32)
    mean_scale = rng.uniform(size=DIM).astype(np.float32)

    pyg_norm = PyGGraphNorm(DIM)
    with torch.no_grad():
        pyg_norm.weight.copy_(torch.from_numpy(weight))
        pyg_norm.bias.copy_(torch.from_numpy(bias))
        pyg_norm.mean_scale.copy_(torch.from_numpy(mean_scale))
        batch = torch.from_numpy(BATCH) if batched else None
        expected = pyg_norm(torch.from_numpy(x), batch).numpy()

    norm = GraphNormalization(DIM)
    norm.weight, norm.bias = mx.array(weight), mx.array(bias)
    norm.mean_scale = mx.array(mean_scale)
    out = norm(mx.array(x), mx.array(BATCH) if batched else None)
    np.testing.assert_allclose(np.array(out), expected, **TOL)


def test_graph_norm_defaults_and_num_graphs():
    rng = np.random.default_rng(1)
    x = mx.array(rng.normal(size=(len(BATCH), DIM)).astype(np.float32))
    norm = GraphNormalization(DIM)
    out = np.array(norm(x, mx.array(BATCH)))
    # zero mean and unit variance per graph (except the 1-node graph -> 0)
    for g in (0, 2):
        np.testing.assert_allclose(out[BATCH == g].mean(0), 0, atol=1e-5)
        np.testing.assert_allclose(out[BATCH == g].var(0), 1, atol=1e-3)
    np.testing.assert_allclose(out[BATCH == 1], 0, atol=1e-5)

    padded = np.array(norm(x, mx.array(BATCH), num_graphs=5))
    np.testing.assert_allclose(padded, out, atol=1e-6)


def test_graph_norm_gradients():
    rng = np.random.default_rng(2)
    x = mx.array(rng.normal(size=(len(BATCH), DIM)).astype(np.float32))
    norm = GraphNormalization(DIM)

    def loss_fn():
        return (norm(x, mx.array(BATCH)) * mx.arange(DIM)).sum()

    _, grads = nn.value_and_grad(norm, loss_fn)()
    grads = dict(tree_flatten(grads))
    assert set(grads) == {"weight", "bias", "mean_scale"}
    for grad in grads.values():
        assert mx.all(mx.isfinite(grad)).item()
