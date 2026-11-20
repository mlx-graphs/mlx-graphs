import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch
import torch_geometric.nn as pyg_nn
from mlx.utils import tree_flatten

import mlx_graphs.nn.message_passing as mp
from mlx_graphs.nn import MessagePassing, fused_message_passing_mode
from mlx_graphs.nn.conv.sg_conv import SGConv

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
    layer = SGConv(ref.in_channels, ref.out_channels, K=ref.K, **kwargs)
    layer.linear.weight = to_mx(ref.lin.weight)
    if ref.lin.bias is not None:
        layer.linear.bias = to_mx(ref.lin.bias)
    return layer


@pytest.mark.parametrize(
    "layer_args, expected",
    [
        (dict(), (100, 20)),
        (dict(K=3), (100, 20)),
        (dict(K=2, bias=False, add_self_loops=False), (100, 20)),
        (dict(K=2, cached=True), (100, 20)),
    ],
)
@pytest.mark.parametrize("weighted", [False, True])
def test_sg_conv_shape(layer_args, expected, weighted):
    layer = SGConv(8, 20, **layer_args)
    edge_index = mx.array([[0, 1, 2, 3, 50], [0, 0, 1, 1, 99]])
    weights = mx.random.uniform(shape=(5,)) if weighted else None

    out = layer(edge_index, mx.random.normal((100, 8)), edge_weights=weights)

    assert out.shape == expected


# out_dim 6 projects before propagating, out_dim 20 after
@pytest.mark.parametrize("out_dim", [6, 20])
@pytest.mark.parametrize("K", [1, 3])
@pytest.mark.parametrize("add_self_loops", [True, False])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("bias", [True, False])
def test_sg_conv_matches_pyg(out_dim, K, add_self_loops, weighted, bias):
    edges, x, weights = make_graph()
    torch.manual_seed(0)
    ref = pyg_nn.SGConv(DIM, out_dim, K=K, add_self_loops=add_self_loops, bias=bias)
    layer = from_pyg(ref, add_self_loops=add_self_loops, bias=bias)

    expected = ref(
        torch.tensor(x),
        torch.tensor(edges),
        torch.tensor(weights) if weighted else None,
    )
    out = layer(mx.array(edges), mx.array(x), mx.array(weights) if weighted else None)

    assert_close(out, expected.detach().numpy(), PYG_TOL)


@pytest.mark.parametrize("out_dim", [6, 20])
def test_sg_conv_cached(out_dim):
    edges, x, weights = make_graph()
    edge_index, x, weights = mx.array(edges), mx.array(x), mx.array(weights)

    mx.random.seed(0)
    plain = SGConv(DIM, out_dim, K=2)
    mx.random.seed(0)
    cached = SGConv(DIM, out_dim, K=2, cached=True)
    expected = np.array(plain(edge_index, x, weights))

    assert cached._cached_features is None
    first = cached(edge_index, x, weights)
    assert_close(first, expected, PYG_TOL)
    assert cached._cached_features.shape == (NUM_NODES, DIM)

    # The cache is not a parameter, and is used instead of the new inputs.
    assert set(dict(tree_flatten(cached.parameters()))) == {
        "linear.weight",
        "linear.bias",
    }
    again = cached(mx.array([[0], [1]]), mx.zeros_like(x))
    np.testing.assert_allclose(np.array(again), np.array(first), rtol=1e-5, atol=1e-5)

    # Without the cache the new inputs are used again.
    cached.clear_cache()
    assert cached._cached_features is None
    recomputed = cached(edge_index, 2 * x, weights)
    bias = np.array(cached.linear.bias)
    assert_close(np.array(recomputed) - bias, 2 * (expected - bias), PYG_TOL)

    # `cached=False` never stores anything.
    assert plain._cached_features is None


@pytest.mark.parametrize("out_dim", [6, 20])
@pytest.mark.parametrize("cached", [False, True])
def test_sg_conv_gradients(out_dim, cached):
    edges, x, weights = make_graph()
    layer = SGConv(DIM, out_dim, K=2, cached=cached)

    def loss_fn(features):
        out = layer(mx.array(edges), features, mx.array(weights))
        return out.square().sum()

    # The second step of a cached layer runs on the stored features.
    for _ in range(2):
        loss, grads = nn.value_and_grad(layer, loss_fn)(mx.array(x))
        grads = dict(tree_flatten(grads))

        assert np.isfinite(loss.item())
        assert set(grads) == set(dict(tree_flatten(layer.trainable_parameters())))
        assert set(grads) == {"linear.weight", "linear.bias"}
        for name, grad in grads.items():
            grad = np.array(grad)
            assert np.isfinite(grad).all(), name
            assert np.abs(grad).max() > 0, name


@pytest.mark.skipif(not mp.HAS_FUSED_KERNELS, reason="mlx_cluster SpMM not installed")
@pytest.mark.parametrize("out_dim", [6, 20])
@pytest.mark.parametrize("weighted", [False, True])
def test_sg_conv_fused_matches_gather_scatter(out_dim, weighted, monkeypatch):
    calls = []
    original = MessagePassing.message_and_aggregate

    def spy(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(MessagePassing, "message_and_aggregate", spy)

    edges, x, weights = make_graph()
    layer = SGConv(DIM, out_dim, K=3)
    args = (mx.array(edges), mx.array(x), mx.array(weights) if weighted else None)

    fused = layer(*args)
    assert len(calls) == 3  # one SpMM per propagation step
    with fused_message_passing_mode(False):
        reference = layer(*args)
    assert len(calls) == 3

    assert_close(fused, reference, FUSED_TOL)
