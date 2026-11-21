import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch
import torch_geometric.nn as pyg_nn
from mlx.utils import tree_flatten

import mlx_graphs.nn.message_passing as mp
from mlx_graphs.nn import fused_message_passing_mode
from mlx_graphs.nn.conv.ssg_conv import SSGConv

NUM_NODES, DIM = 30, 12
# MLX's GPU matmul is only accurate to ~1e-3 relative, a wrong formula is O(1) off
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


def from_pyg(ref, **kwargs):
    layer = SSGConv(
        ref.in_channels, ref.out_channels, alpha=ref.alpha, K=ref.K, **kwargs
    )
    layer.linear.weight = to_mx(ref.lin.weight)
    if ref.lin.bias is not None:
        layer.linear.bias = to_mx(ref.lin.bias)
    return layer


@pytest.mark.parametrize(
    "layer_args",
    [dict(), dict(K=3), dict(K=2, bias=False, add_self_loops=False), dict(cached=True)],
)
@pytest.mark.parametrize("weighted", [False, True])
def test_ssg_conv_shape(layer_args, weighted):
    layer = SSGConv(8, 20, alpha=0.1, **layer_args)
    edge_index = mx.array([[0, 1, 2, 3, 50], [0, 0, 1, 1, 99]])
    weights = mx.random.uniform(shape=(5,)) if weighted else None

    out = layer(edge_index, mx.random.normal((100, 8)), edge_weights=weights)

    assert out.shape == (100, 20)


# out_dim 6 projects before propagating, out_dim 20 after
@pytest.mark.parametrize("out_dim", [6, 20])
@pytest.mark.parametrize("K", [1, 3])
@pytest.mark.parametrize("add_self_loops", [True, False])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("bias", [True, False])
def test_ssg_conv_matches_pyg(out_dim, K, add_self_loops, weighted, bias):
    edges, x, weights = make_graph()
    torch.manual_seed(0)
    ref = pyg_nn.SSGConv(
        DIM, out_dim, alpha=0.3, K=K, add_self_loops=add_self_loops, bias=bias
    )
    layer = from_pyg(ref, add_self_loops=add_self_loops, bias=bias)

    expected = ref(
        torch.tensor(x),
        torch.tensor(edges),
        torch.tensor(weights) if weighted else None,
    )
    out = layer(mx.array(edges), mx.array(x), mx.array(weights) if weighted else None)

    assert_close(out, expected.detach().numpy(), PYG_TOL)


def test_ssg_conv_cached():
    edges, x, weights = make_graph()
    edge_index, x, weights = mx.array(edges), mx.array(x), mx.array(weights)
    layer = SSGConv(DIM, 6, alpha=0.1, K=2, cached=True)

    first = layer(edge_index, x, weights)
    assert layer._cached_features.shape == (NUM_NODES, DIM)
    assert set(dict(tree_flatten(layer.parameters()))) == {
        "linear.weight",
        "linear.bias",
    }
    again = layer(mx.array([[0], [1]]), mx.zeros_like(x))
    np.testing.assert_allclose(np.array(again), np.array(first), rtol=1e-5, atol=1e-5)

    layer.clear_cache()
    assert layer._cached_features is None


@pytest.mark.parametrize("out_dim", [6, 20])
def test_ssg_conv_gradients(out_dim):
    edges, x, weights = make_graph()
    layer = SSGConv(DIM, out_dim, alpha=0.1, K=2)

    def loss_fn(features):
        return layer(mx.array(edges), features, mx.array(weights)).square().sum()

    loss, grads = nn.value_and_grad(layer, loss_fn)(mx.array(x))
    grads = dict(tree_flatten(grads))

    assert np.isfinite(loss.item())
    assert set(grads) == {"linear.weight", "linear.bias"}
    for name, grad in grads.items():
        grad = np.array(grad)
        assert np.isfinite(grad).all(), name
        assert np.abs(grad).max() > 0, name


@pytest.mark.skipif(not mp.HAS_FUSED_KERNELS, reason="mlx_cluster SpMM not installed")
@pytest.mark.parametrize("out_dim", [6, 20])
@pytest.mark.parametrize("weighted", [False, True])
def test_ssg_conv_fused_matches_gather_scatter(out_dim, weighted):
    edges, x, weights = make_graph()
    layer = SSGConv(DIM, out_dim, alpha=0.1, K=3)
    args = (mx.array(edges), mx.array(x), mx.array(weights) if weighted else None)

    fused = layer(*args)
    with fused_message_passing_mode(False):
        reference = layer(*args)

    assert_close(fused, reference, FUSED_TOL)
