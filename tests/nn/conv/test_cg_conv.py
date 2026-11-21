import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch
import torch_geometric.nn as pyg_nn
from mlx.utils import tree_flatten

from mlx_graphs.nn.conv.cg_conv import CGConv

NUM_NODES, DIM, EDGE_DIM = 30, 12, 5
PYG_TOL = 2e-2


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
    edge_attr = rng.normal(size=(edges.shape[1], EDGE_DIM)).astype(np.float32)
    return edges, x, edge_attr


def to_mx(tensor):
    return mx.array(tensor.detach().numpy())


def from_pyg(ref, **kwargs):
    layer = CGConv(ref.channels, edge_features_dim=ref.dim, **kwargs)
    for name in ("lin_f", "lin_s"):
        src = getattr(ref, name)
        layer[name].weight = to_mx(src.weight)
        if src.bias is not None:
            layer[name].bias = to_mx(src.bias)
    return layer


@pytest.mark.parametrize(
    "layer_args, with_edges",
    [
        (dict(), False),
        (dict(edge_features_dim=EDGE_DIM), True),
        (dict(edge_features_dim=EDGE_DIM, batch_norm=True, bias=False), True),
    ],
)
def test_cg_conv_shape(layer_args, with_edges):
    layer = CGConv(8, **layer_args)
    edge_index = mx.array([[0, 1, 2, 3, 50], [0, 0, 1, 1, 99]])
    edge_features = mx.random.normal((5, EDGE_DIM)) if with_edges else None

    out = layer(edge_index, mx.random.normal((100, 8)), edge_features)

    assert out.shape == (100, 8)


@pytest.mark.parametrize("edge_dim", [0, EDGE_DIM])
@pytest.mark.parametrize("aggr", ["add", "mean", "max"])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("batch_norm", [False, True])
def test_cg_conv_matches_pyg(edge_dim, aggr, bias, batch_norm):
    edges, x, edge_attr = make_graph()
    torch.manual_seed(0)
    ref = pyg_nn.CGConv(DIM, dim=edge_dim, aggr=aggr, bias=bias, batch_norm=batch_norm)
    layer = from_pyg(ref, aggr=aggr, bias=bias, batch_norm=batch_norm)

    e = edge_attr if edge_dim else None
    # Both batch norms are in training mode (statistics of the batch).
    expected = ref(
        torch.tensor(x), torch.tensor(edges), None if e is None else torch.tensor(e)
    )
    out = layer(mx.array(edges), mx.array(x), None if e is None else mx.array(e))

    assert_close(out, expected.detach().numpy(), PYG_TOL)


def test_cg_conv_bipartite_matches_pyg():
    rng = np.random.default_rng(0)
    edges = np.array([[0, 0, 1, 2, 3, 4, 5, 6, 7, 7], [0, 0, 0, 1, 1, 2, 2, 3, 3, 0]])
    x_src = rng.normal(size=(9, 16)).astype(np.float32)
    x_dst = rng.normal(size=(5, 8)).astype(np.float32)
    edge_attr = rng.normal(size=(edges.shape[1], EDGE_DIM)).astype(np.float32)

    torch.manual_seed(0)
    ref = pyg_nn.CGConv((16, 8), dim=EDGE_DIM)
    layer = from_pyg(ref)

    expected = ref(
        (torch.tensor(x_src), torch.tensor(x_dst)),
        torch.tensor(edges),
        torch.tensor(edge_attr),
    )
    out = layer(
        mx.array(edges), (mx.array(x_src), mx.array(x_dst)), mx.array(edge_attr)
    )

    assert out.shape == (5, 8)
    assert_close(out, expected.detach().numpy(), PYG_TOL)


def test_cg_conv_gradients():
    edges, x, edge_attr = make_graph()
    layer = CGConv(DIM, edge_features_dim=EDGE_DIM)

    def loss_fn(features):
        out = layer(mx.array(edges), features, mx.array(edge_attr))
        return out.square().sum()

    loss, grads = nn.value_and_grad(layer, loss_fn)(mx.array(x))
    grads = dict(tree_flatten(grads))

    assert np.isfinite(loss.item())
    assert set(grads) == {"lin_f.weight", "lin_f.bias", "lin_s.weight", "lin_s.bias"}
    for name, grad in grads.items():
        grad = np.array(grad)
        assert np.isfinite(grad).all(), name
        assert np.abs(grad).max() > 0, name
