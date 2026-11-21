import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch
import torch_geometric.nn as pyg_nn
from mlx.utils import tree_flatten

import mlx_graphs.nn.message_passing as mp
from mlx_graphs.nn import fused_message_passing_mode
from mlx_graphs.nn.conv.gated_graph_conv import GatedGraphConv

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


def to_mx(tensor):
    return mx.array(tensor.detach().numpy())


def from_pyg(ref, bias=True, **kwargs):
    layer = GatedGraphConv(ref.out_channels, ref.num_layers, bias=bias, **kwargs)
    layer.weight = to_mx(ref.weight)
    rnn, H = ref.rnn, ref.out_channels
    layer.rnn.Wx = to_mx(rnn.weight_ih)
    layer.rnn.Wh = to_mx(rnn.weight_hh)
    if bias:
        # mlx's GRU folds the r/z parts of bias_hh into its input bias.
        bias_hh = to_mx(rnn.bias_hh)
        layer.rnn.b = to_mx(rnn.bias_ih) + mx.concatenate(
            [bias_hh[: 2 * H], mx.zeros((H,))]
        )
        layer.rnn.bhn = bias_hh[2 * H :]
    return layer


@pytest.mark.parametrize("in_dim", [8, 20])
@pytest.mark.parametrize("weighted", [False, True])
def test_gated_graph_conv_shape(in_dim, weighted):
    layer = GatedGraphConv(20, num_layers=3)
    edge_index = mx.array([[0, 1, 2, 3, 50], [0, 0, 1, 1, 99]])
    weights = mx.random.uniform(shape=(5,)) if weighted else None

    out = layer(edge_index, mx.random.normal((100, in_dim)), weights)

    assert out.shape == (100, 20)


def test_gated_graph_conv_too_many_features():
    with pytest.raises(ValueError):
        GatedGraphConv(4, 2)(mx.array([[0], [1]]), mx.zeros((2, 8)))


@pytest.mark.parametrize("out_dim", [DIM, 16])
@pytest.mark.parametrize("num_layers", [1, 3])
@pytest.mark.parametrize("aggr", ["add", "mean", "max"])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("weighted", [False, True])
def test_gated_graph_conv_matches_pyg(out_dim, num_layers, aggr, bias, weighted):
    edges, x, weights = make_graph()
    torch.manual_seed(0)
    ref = pyg_nn.GatedGraphConv(out_dim, num_layers, aggr=aggr, bias=bias)
    layer = from_pyg(ref, bias=bias, aggr=aggr)

    expected = ref(
        torch.tensor(x),
        torch.tensor(edges),
        torch.tensor(weights) if weighted else None,
    )
    out = layer(mx.array(edges), mx.array(x), mx.array(weights) if weighted else None)

    assert_close(out, expected.detach().numpy(), PYG_TOL)


def test_gated_graph_conv_gradients():
    edges, x, weights = make_graph()
    layer = GatedGraphConv(16, 2)

    def loss_fn(features):
        return layer(mx.array(edges), features, mx.array(weights)).square().sum()

    loss, grads = nn.value_and_grad(layer, loss_fn)(mx.array(x))
    grads = dict(tree_flatten(grads))

    assert np.isfinite(loss.item())
    assert set(grads) == {"weight", "rnn.Wx", "rnn.Wh", "rnn.b", "rnn.bhn"}
    for name, grad in grads.items():
        grad = np.array(grad)
        assert np.isfinite(grad).all(), name
        assert np.abs(grad).max() > 0, name


@pytest.mark.skipif(not mp.HAS_FUSED_KERNELS, reason="mlx_cluster SpMM not installed")
@pytest.mark.parametrize("weighted", [False, True])
def test_gated_graph_conv_fused_matches_gather_scatter(weighted):
    edges, x, weights = make_graph()
    layer = GatedGraphConv(16, 3)
    args = (mx.array(edges), mx.array(x), mx.array(weights) if weighted else None)

    fused = layer(*args)
    with fused_message_passing_mode(False):
        reference = layer(*args)

    assert_close(fused, reference, FUSED_TOL)
