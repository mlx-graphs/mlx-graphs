import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch
import torch_geometric.nn as pyg_nn
from mlx.utils import tree_flatten

import mlx_graphs.nn.message_passing as mp
from mlx_graphs.nn import MessagePassing, fused_message_passing_mode
from mlx_graphs.nn.conv.tag_conv import TAGConv

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


def make_pair(out_dim, K, normalize=True, bias=True):
    """A PyG TAGConv and an mlx-graphs TAGConv with the same weights."""
    torch.manual_seed(0)
    ref = pyg_nn.TAGConv(DIM, out_dim, K=K, normalize=normalize, bias=bias)
    layer = TAGConv(DIM, out_dim, K=K, normalize=normalize, bias=bias)
    for lin, ref_lin in zip(layer.lins, ref.lins):
        lin.weight = to_mx(ref_lin.weight)
    if bias:
        torch.nn.init.normal_(ref.bias)  # zeros by default
        layer.bias = to_mx(ref.bias)
    return ref, layer


@pytest.mark.parametrize(
    "layer_args, num_lins",
    [
        (dict(), 4),
        (dict(K=1), 2),
        (dict(K=2, bias=False), 3),
        (dict(K=2, normalize=False), 3),
    ],
)
@pytest.mark.parametrize("weighted", [False, True])
def test_tag_conv_shape(layer_args, num_lins, weighted):
    layer = TAGConv(8, 20, **layer_args)
    edge_index = mx.array([[0, 1, 2, 3, 50], [0, 0, 1, 1, 99]])
    weights = mx.random.uniform(shape=(5,)) if weighted else None

    out = layer(edge_index, mx.random.normal((100, 8)), edge_weights=weights)

    assert out.shape == (100, 20)
    assert len(layer.lins) == num_lins


@pytest.mark.parametrize("K", [0, 1, 3])
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("bias", [True, False])
def test_tag_conv_matches_pyg(K, normalize, weighted, bias):
    edges, x, weights = make_graph()
    ref, layer = make_pair(7, K, normalize, bias)
    assert ("bias" in layer) == bias

    expected = ref(
        torch.tensor(x),
        torch.tensor(edges),
        torch.tensor(weights) if weighted else None,
    )
    out = layer(mx.array(edges), mx.array(x), mx.array(weights) if weighted else None)

    assert_close(out, expected.detach().numpy(), PYG_TOL)


@pytest.mark.parametrize("K", [0, 3])
@pytest.mark.parametrize("normalize", [True, False])
def test_tag_conv_gradients(K, normalize):
    edges, x, weights = make_graph()
    layer = TAGConv(DIM, 7, K=K, normalize=normalize)

    def loss_fn(features):
        out = layer(mx.array(edges), features, mx.array(weights))
        return out.square().sum()

    loss, grads = nn.value_and_grad(layer, loss_fn)(mx.array(x))
    grads = dict(tree_flatten(grads))

    assert np.isfinite(loss.item())
    assert set(grads) == set(dict(tree_flatten(layer.trainable_parameters())))
    assert set(grads) == {f"lins.{k}.weight" for k in range(K + 1)} | {"bias"}
    for name, grad in grads.items():
        grad = np.array(grad)
        assert np.isfinite(grad).all(), name
        assert np.abs(grad).max() > 0, name


@pytest.mark.skipif(not mp.HAS_FUSED_KERNELS, reason="mlx_cluster SpMM not installed")
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("weighted", [False, True])
def test_tag_conv_fused_matches_gather_scatter(normalize, weighted, monkeypatch):
    calls = []
    original = MessagePassing.message_and_aggregate

    def spy(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(MessagePassing, "message_and_aggregate", spy)

    edges, x, weights = make_graph()
    layer = TAGConv(DIM, 7, K=3, normalize=normalize)
    args = (mx.array(edges), mx.array(x), mx.array(weights) if weighted else None)

    fused = layer(*args)
    assert len(calls) == 3  # one SpMM per hop
    with fused_message_passing_mode(False):
        reference = layer(*args)
    assert len(calls) == 3

    assert_close(fused, reference, FUSED_TOL)
