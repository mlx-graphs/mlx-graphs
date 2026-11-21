import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch
import torch_geometric.nn as pyg_nn
from mlx.utils import tree_flatten

from mlx_graphs.nn.conv.le_conv import LEConv

NUM_NODES, DIM = 30, 12
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
    weights = rng.uniform(0.1, 2.0, edges.shape[1]).astype(np.float32)
    return edges, x, weights


def to_mx(tensor):
    return mx.array(tensor.detach().numpy())


def from_pyg(ref, **kwargs):
    layer = LEConv(ref.in_channels, ref.out_channels, **kwargs)
    for name in ("lin1", "lin2", "lin3"):
        src = getattr(ref, name)
        layer[name].weight = to_mx(src.weight)
        if src.bias is not None:
            layer[name].bias = to_mx(src.bias)
    return layer


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("weighted", [False, True])
def test_le_conv_shape(bias, weighted):
    layer = LEConv(8, 20, bias=bias)
    edge_index = mx.array([[0, 1, 2, 3, 50], [0, 0, 1, 1, 99]])
    weights = mx.random.uniform(shape=(5,)) if weighted else None

    out = layer(edge_index, mx.random.normal((100, 8)), weights)

    assert out.shape == (100, 20)


@pytest.mark.parametrize("aggr", ["add", "mean", "max"])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("weighted", [False, True])
def test_le_conv_matches_pyg(aggr, bias, weighted):
    edges, x, weights = make_graph()
    torch.manual_seed(0)
    ref = pyg_nn.LEConv(DIM, 7, bias=bias, aggr=aggr)
    layer = from_pyg(ref, bias=bias, aggr=aggr)

    expected = ref(
        torch.tensor(x),
        torch.tensor(edges),
        torch.tensor(weights) if weighted else None,
    )
    out = layer(mx.array(edges), mx.array(x), mx.array(weights) if weighted else None)

    assert_close(out, expected.detach().numpy(), PYG_TOL)


def test_le_conv_bipartite_matches_pyg():
    rng = np.random.default_rng(0)
    edges = np.array([[0, 0, 1, 2, 3, 4, 5, 6, 7, 7], [0, 0, 0, 1, 1, 2, 2, 3, 3, 0]])
    x_src = rng.normal(size=(9, 16)).astype(np.float32)
    x_dst = rng.normal(size=(5, 8)).astype(np.float32)

    torch.manual_seed(0)
    ref = pyg_nn.LEConv((16, 8), 7)
    layer = from_pyg(ref)

    expected = ref((torch.tensor(x_src), torch.tensor(x_dst)), torch.tensor(edges))
    out = layer(mx.array(edges), (mx.array(x_src), mx.array(x_dst)))

    assert out.shape == (5, 7)
    assert_close(out, expected.detach().numpy(), PYG_TOL)


def test_le_conv_gradients():
    edges, x, weights = make_graph()
    layer = LEConv(DIM, 7)

    def loss_fn(features):
        return layer(mx.array(edges), features, mx.array(weights)).square().sum()

    loss, grads = nn.value_and_grad(layer, loss_fn)(mx.array(x))
    grads = dict(tree_flatten(grads))

    assert np.isfinite(loss.item())
    assert set(grads) == {
        "lin1.weight",
        "lin1.bias",
        "lin2.weight",
        "lin3.weight",
        "lin3.bias",
    }
    for name, grad in grads.items():
        grad = np.array(grad)
        assert np.isfinite(grad).all(), name
        assert np.abs(grad).max() > 0, name
