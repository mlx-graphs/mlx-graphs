import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch
import torch_geometric.nn as pyg_nn
from mlx.utils import tree_flatten

import mlx_graphs.nn.message_passing as mp
from mlx_graphs.nn import MessagePassing, fused_message_passing_mode
from mlx_graphs.nn.conv.graph_conv import GraphConv

NUM_NODES, DIM = 30, 12
# MLX's GPU matmul is only accurate to ~1e-3 relative, a wrong formula is O(1) off
PYG_TOL = 2e-2
# Fused and gather-scatter only differ by the order of the sums
FUSED_TOL = 1e-4


def assert_close(actual, expected, tol):
    """Compares with a tolerance relative to the largest expected value, which
    is the size of the terms that are summed into every output entry."""
    actual, expected = np.array(actual), np.array(expected)
    scale = max(1.0, float(np.abs(expected).max()))
    np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol * scale)


def make_graph(seed=0):
    """Random directed graph with duplicate edges, a self-loop on nodes 0-2 and
    five isolated nodes (25-29)."""
    rng = np.random.default_rng(seed)
    edges = rng.integers(0, NUM_NODES - 5, (2, 120))
    edges = edges[:, edges[0] != edges[1]]
    loops = np.array([[0, 1, 2], [0, 1, 2]])
    edges = np.concatenate([edges, edges[:, :10], loops], axis=1)
    x = rng.normal(size=(NUM_NODES, DIM)).astype(np.float32)
    weights = rng.uniform(0.1, 2.0, edges.shape[1]).astype(np.float32)
    return edges, x, weights


def to_mx(tensor):
    return mx.array(tensor.detach().numpy())


def from_pyg(ref, **kwargs):
    in_dims = ref.in_channels
    layer = GraphConv(in_dims, ref.out_channels, aggr=ref.aggr, **kwargs)
    layer.neigh_proj.weight = to_mx(ref.lin_rel.weight)
    if ref.lin_rel.bias is not None:
        layer.neigh_proj.bias = to_mx(ref.lin_rel.bias)
    layer.self_proj.weight = to_mx(ref.lin_root.weight)
    return layer


@pytest.mark.parametrize(
    "edge_index, num_nodes, layer_args, expected",
    [
        ([[0, 1, 2, 3], [0, 0, 1, 1]], 6, dict(), (6, 20)),
        ([[0, 1, 2, 3, 50], [0, 0, 1, 1, 99]], 100, dict(bias=False), (100, 20)),
        ([[0, 1, 2, 3, 50], [0, 0, 1, 1, 99]], 100, dict(aggr="max"), (100, 20)),
    ],
)
@pytest.mark.parametrize("weighted", [False, True])
def test_graph_conv_shape(edge_index, num_nodes, layer_args, expected, weighted):
    layer = GraphConv(8, 20, **layer_args)
    edge_index = mx.array(edge_index)
    weights = mx.random.uniform(shape=(edge_index.shape[1],)) if weighted else None

    out = layer(edge_index, mx.random.normal((num_nodes, 8)), edge_weights=weights)

    assert out.shape == expected


# out_dim 6 projects before aggregating (add/mean), out_dim 20 after
@pytest.mark.parametrize("out_dim", [6, 20])
@pytest.mark.parametrize("aggr", ["add", "mean", "max"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("bias", [True, False])
def test_graph_conv_matches_pyg(out_dim, aggr, weighted, bias):
    edges, x, weights = make_graph()
    torch.manual_seed(0)
    ref = pyg_nn.GraphConv(DIM, out_dim, aggr=aggr, bias=bias)
    layer = from_pyg(ref, bias=bias)

    expected = ref(
        torch.tensor(x),
        torch.tensor(edges),
        torch.tensor(weights) if weighted else None,
    )
    out = layer(mx.array(edges), mx.array(x), mx.array(weights) if weighted else None)

    assert_close(out, expected.detach().numpy(), PYG_TOL)


@pytest.mark.parametrize("in_dims, out_dim", [((16, 8), 32), ((64, 8), 4)])
def test_graph_conv_bipartite_matches_pyg(in_dims, out_dim):
    rng = np.random.default_rng(0)
    num_src, num_dst = 9, 5
    # dst 4 gets no message, src 8 sends none, edge (0 -> 0) is duplicated
    edges = np.array([[0, 0, 1, 2, 3, 4, 5, 6, 7, 7], [0, 0, 0, 1, 1, 2, 2, 3, 3, 0]])
    x_src = rng.normal(size=(num_src, in_dims[0])).astype(np.float32)
    x_dst = rng.normal(size=(num_dst, in_dims[1])).astype(np.float32)
    weights = rng.uniform(0.1, 2.0, edges.shape[1]).astype(np.float32)

    torch.manual_seed(0)
    ref = pyg_nn.GraphConv(in_dims, out_dim)
    layer = from_pyg(ref)

    expected = ref(
        (torch.tensor(x_src), torch.tensor(x_dst)),
        torch.tensor(edges),
        torch.tensor(weights),
        size=(num_src, num_dst),
    )
    out = layer(mx.array(edges), (mx.array(x_src), mx.array(x_dst)), mx.array(weights))

    assert out.shape == (num_dst, out_dim)
    assert_close(out, expected.detach().numpy(), PYG_TOL)


def test_graph_conv_column_edge_weights():
    edges, x, weights = make_graph()
    layer = GraphConv(DIM, 8)

    flat = layer(mx.array(edges), mx.array(x), mx.array(weights))
    column = layer(mx.array(edges), mx.array(x), mx.array(weights)[:, None])

    np.testing.assert_allclose(np.array(column), np.array(flat), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("out_dim", [6, 20])
@pytest.mark.parametrize("weighted", [False, True])
def test_graph_conv_gradients(out_dim, weighted):
    edges, x, weights = make_graph()
    layer = GraphConv(DIM, out_dim)
    edge_weights = mx.array(weights) if weighted else None

    def loss_fn(features):
        return layer(mx.array(edges), features, edge_weights).square().sum()

    loss, grads = nn.value_and_grad(layer, loss_fn)(mx.array(x))
    grads = dict(tree_flatten(grads))

    assert np.isfinite(loss.item())
    assert set(grads) == set(dict(tree_flatten(layer.trainable_parameters())))
    assert set(grads) == {"neigh_proj.weight", "neigh_proj.bias", "self_proj.weight"}
    for name, grad in grads.items():
        grad = np.array(grad)
        assert np.isfinite(grad).all(), name
        assert np.abs(grad).max() > 0, name


@pytest.mark.skipif(not mp.HAS_FUSED_KERNELS, reason="mlx_cluster SpMM not installed")
@pytest.mark.parametrize("out_dim", [6, 20])
@pytest.mark.parametrize("aggr", ["add", "mean", "max"])
@pytest.mark.parametrize("weighted", [False, True])
def test_graph_conv_fused_matches_gather_scatter(out_dim, aggr, weighted, monkeypatch):
    calls = []
    original = MessagePassing.message_and_aggregate

    def spy(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(MessagePassing, "message_and_aggregate", spy)

    edges, x, weights = make_graph()
    layer = GraphConv(DIM, out_dim, aggr=aggr)
    args = (mx.array(edges), mx.array(x), mx.array(weights) if weighted else None)

    fused = layer(*args)
    assert len(calls) == 1
    with fused_message_passing_mode(False):
        reference = layer(*args)
    assert len(calls) == 1

    assert_close(fused, reference, FUSED_TOL)
