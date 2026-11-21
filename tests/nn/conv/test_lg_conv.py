import mlx.core as mx
import numpy as np
import pytest
import torch
import torch_geometric.nn as pyg_nn
from mlx.utils import tree_flatten

import mlx_graphs.nn.message_passing as mp
from mlx_graphs.nn import fused_message_passing_mode
from mlx_graphs.nn.conv.lg_conv import LGConv

NUM_NODES, DIM = 30, 12
PYG_TOL = 2e-2
FUSED_TOL = 1e-4


def assert_close(actual, expected, tol):
    actual, expected = np.array(actual), np.array(expected)
    scale = max(1.0, float(np.abs(expected).max()))
    np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol * scale)


def make_graph(seed=0):
    """Random directed graph with duplicate edges, self-loops and isolated nodes."""
    rng = np.random.default_rng(seed)
    edges = rng.integers(0, NUM_NODES - 5, (2, 120))
    edges = edges[:, edges[0] != edges[1]]
    loops = np.array([[0, 1, 2], [0, 1, 2]])
    edges = np.concatenate([edges, edges[:, :10], loops], axis=1)
    x = rng.normal(size=(NUM_NODES, DIM)).astype(np.float32)
    weights = rng.uniform(0.1, 2.0, edges.shape[1]).astype(np.float32)
    return edges, x, weights


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("weighted", [False, True])
def test_lg_conv_shape(normalize, weighted):
    layer = LGConv(normalize=normalize)
    edge_index = mx.array([[0, 1, 2, 3, 50], [0, 0, 1, 1, 99]])
    weights = mx.random.uniform(shape=(5,)) if weighted else None

    out = layer(edge_index, mx.random.normal((100, DIM)), weights)

    assert out.shape == (100, DIM)
    assert len(tree_flatten(layer.parameters())) == 0


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("aggr", ["add", "mean"])
def test_lg_conv_matches_pyg(normalize, weighted, aggr):
    edges, x, weights = make_graph()
    ref = pyg_nn.LGConv(normalize=normalize, aggr=aggr)
    layer = LGConv(normalize=normalize, aggr=aggr)

    w = weights if weighted else None
    expected = ref(
        torch.tensor(x), torch.tensor(edges), None if w is None else torch.tensor(w)
    )
    out = layer(mx.array(edges), mx.array(x), None if w is None else mx.array(w))

    assert_close(out, expected.numpy(), PYG_TOL)


@pytest.mark.skipif(not mp.HAS_FUSED_KERNELS, reason="mlx_cluster SpMM not installed")
@pytest.mark.parametrize("weighted", [False, True])
def test_lg_conv_fused_matches_gather_scatter(weighted):
    edges, x, weights = make_graph()
    layer = LGConv()
    args = (mx.array(edges), mx.array(x), mx.array(weights) if weighted else None)

    fused = layer(*args)
    with fused_message_passing_mode(False):
        reference = layer(*args)

    assert_close(fused, reference, FUSED_TOL)
