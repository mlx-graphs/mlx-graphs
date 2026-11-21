import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch
import torch_geometric.nn as pyg_nn
from mlx.utils import tree_flatten

import mlx_graphs.nn.message_passing as mp
from mlx_graphs.nn import fused_message_passing_mode
from mlx_graphs.nn.conv.arma_conv import ARMAConv

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


@pytest.mark.parametrize(
    "layer_args",
    [
        dict(),
        dict(num_stacks=3, num_layers=2),
        dict(num_stacks=2, num_layers=3, shared_weights=True, bias=False),
        dict(act=None, dropout=0.5),
    ],
)
def test_arma_conv_shape(layer_args):
    layer = ARMAConv(8, 20, **layer_args)
    edge_index = mx.array([[0, 1, 2, 3, 50], [0, 0, 1, 1, 99]])

    out = layer(edge_index, mx.random.normal((100, 8)))

    assert out.shape == (100, 20)


@pytest.mark.parametrize("num_stacks, num_layers", [(1, 1), (3, 1), (2, 3)])
@pytest.mark.parametrize("shared_weights", [False, True])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("weighted", [False, True])
def test_arma_conv_matches_pyg(num_stacks, num_layers, shared_weights, bias, weighted):
    edges, x, weights = make_graph()
    args = dict(
        num_stacks=num_stacks,
        num_layers=num_layers,
        shared_weights=shared_weights,
        bias=bias,
    )
    torch.manual_seed(0)
    ref = pyg_nn.ARMAConv(DIM, 7, **args)
    layer = ARMAConv(DIM, 7, **args)
    layer.weight = to_mx(ref.weight)
    layer.init_weight = to_mx(ref.init_weight)
    layer.root_weight = to_mx(ref.root_weight)
    if bias:
        # PyG initializes the bias to zeros, which would hide a broadcast bug.
        torch.nn.init.normal_(ref.bias)
        layer.bias = to_mx(ref.bias)

    expected = ref(
        torch.tensor(x),
        torch.tensor(edges),
        torch.tensor(weights) if weighted else None,
    )
    out = layer(mx.array(edges), mx.array(x), mx.array(weights) if weighted else None)

    assert_close(out, expected.detach().numpy(), PYG_TOL)


def test_arma_conv_dropout_only_in_training():
    edges, x, _ = make_graph()
    layer = ARMAConv(DIM, 7, num_layers=2, dropout=0.5)
    layer.eval()
    a = layer(mx.array(edges), mx.array(x))
    b = layer(mx.array(edges), mx.array(x))
    np.testing.assert_allclose(np.array(a), np.array(b), rtol=1e-5, atol=1e-5)

    layer.train()
    mx.random.seed(0)
    c = layer(mx.array(edges), mx.array(x))
    assert not np.allclose(np.array(a), np.array(c), rtol=1e-3, atol=1e-3)


def test_arma_conv_gradients():
    edges, x, weights = make_graph()
    layer = ARMAConv(DIM, 7, num_stacks=2, num_layers=3)

    def loss_fn(features):
        return layer(mx.array(edges), features, mx.array(weights)).square().sum()

    loss, grads = nn.value_and_grad(layer, loss_fn)(mx.array(x))
    grads = dict(tree_flatten(grads))

    assert np.isfinite(loss.item())
    assert set(grads) == {"weight", "init_weight", "root_weight", "bias"}
    for name, grad in grads.items():
        grad = np.array(grad)
        assert np.isfinite(grad).all(), name
        assert np.abs(grad).max() > 0, name


@pytest.mark.skipif(not mp.HAS_FUSED_KERNELS, reason="mlx_cluster SpMM not installed")
@pytest.mark.parametrize("weighted", [False, True])
def test_arma_conv_fused_matches_gather_scatter(weighted):
    edges, x, weights = make_graph()
    layer = ARMAConv(DIM, 7, num_stacks=2, num_layers=2)
    args = (mx.array(edges), mx.array(x), mx.array(weights) if weighted else None)

    fused = layer(*args)
    with fused_message_passing_mode(False):
        reference = layer(*args)

    assert_close(fused, reference, FUSED_TOL)
