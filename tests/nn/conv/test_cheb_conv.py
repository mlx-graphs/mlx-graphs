import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch
import torch_geometric.nn as pyg_nn
from mlx.utils import tree_flatten

import mlx_graphs.nn.message_passing as mp
from mlx_graphs.nn import MessagePassing, fused_message_passing_mode
from mlx_graphs.nn.conv.cheb_conv import ChebConv

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


def make_pair(out_dim, K, normalization="sym", bias=True):
    """A PyG ChebConv and an mlx-graphs ChebConv with the same weights."""
    torch.manual_seed(0)
    ref = pyg_nn.ChebConv(DIM, out_dim, K=K, normalization=normalization, bias=bias)
    layer = ChebConv(DIM, out_dim, K=K, normalization=normalization, bias=bias)
    for lin, ref_lin in zip(layer.lins, ref.lins):
        lin.weight = to_mx(ref_lin.weight)
    if bias:
        torch.nn.init.normal_(ref.bias)  # zeros by default
        layer.bias = to_mx(ref.bias)
    return ref, layer


@pytest.mark.parametrize(
    "layer_args, expected",
    [
        (dict(K=1), (100, 20)),
        (dict(K=3), (100, 20)),
        (dict(K=3, normalization="rw", bias=False), (100, 20)),
        (dict(K=2, normalization=None), (100, 20)),
    ],
)
@pytest.mark.parametrize("weighted", [False, True])
def test_cheb_conv_shape(layer_args, expected, weighted):
    layer = ChebConv(8, 20, **layer_args)
    edge_index = mx.array([[0, 1, 2, 3, 50], [0, 0, 1, 1, 99]])
    weights = mx.random.uniform(shape=(5,)) if weighted else None

    out = layer(edge_index, mx.random.normal((100, 8)), edge_weights=weights)

    assert out.shape == expected
    assert len(layer.lins) == layer_args["K"]


@pytest.mark.parametrize("K", [1, 2, 4])
@pytest.mark.parametrize("normalization", ["sym", "rw", None])
@pytest.mark.parametrize("lambda_max", [None, 3.0])
@pytest.mark.parametrize("weighted", [False, True])
def test_cheb_conv_matches_pyg(K, normalization, lambda_max, weighted):
    edges, x, weights = make_graph()
    ref, layer = make_pair(7, K, normalization)

    expected = ref(
        torch.tensor(x),
        torch.tensor(edges),
        torch.tensor(weights) if weighted else None,
        lambda_max=lambda_max,
    )
    out = layer(
        mx.array(edges),
        mx.array(x),
        mx.array(weights) if weighted else None,
        lambda_max=lambda_max,
    )

    assert_close(out, expected.detach().numpy(), PYG_TOL)


def test_cheb_conv_no_bias_matches_pyg():
    edges, x, weights = make_graph()
    ref, layer = make_pair(7, 3, bias=False)
    assert "bias" not in layer

    expected = ref(torch.tensor(x), torch.tensor(edges), torch.tensor(weights))
    out = layer(mx.array(edges), mx.array(x), mx.array(weights))

    assert_close(out, expected.detach().numpy(), PYG_TOL)


def test_cheb_conv_lambda_max_per_graph_matches_pyg():
    # Two graphs in one batch, each with its own largest eigenvalue.
    edges, x, weights = make_graph()
    batch = (np.arange(NUM_NODES) >= 12).astype(np.int64)
    edges = edges[:, batch[edges[0]] == batch[edges[1]]]
    lambda_max = np.array([1.5, 2.5], dtype=np.float32)
    ref, layer = make_pair(7, 3)

    expected = ref(
        torch.tensor(x),
        torch.tensor(edges),
        batch=torch.tensor(batch),
        lambda_max=torch.tensor(lambda_max),
    )
    out = layer(
        mx.array(edges),
        mx.array(x),
        batch=mx.array(batch),
        lambda_max=mx.array(lambda_max),
    )

    assert_close(out, expected.detach().numpy(), PYG_TOL)

    with pytest.raises(ValueError):
        layer(mx.array(edges), mx.array(x), lambda_max=mx.array(lambda_max))


def test_cheb_conv_ignores_input_self_loops():
    edges, x, weights = make_graph()
    no_loops = edges[0] != edges[1]
    assert not no_loops.all()
    layer = ChebConv(DIM, 7, K=3)

    out = layer(mx.array(edges), mx.array(x), mx.array(weights))
    expected = layer(
        mx.array(edges[:, no_loops]), mx.array(x), mx.array(weights[no_loops])
    )

    np.testing.assert_allclose(np.array(out), np.array(expected), rtol=1e-4, atol=1e-4)


def test_cheb_conv_invalid_arguments():
    with pytest.raises(ValueError):
        ChebConv(8, 16, K=0)
    with pytest.raises(ValueError):
        ChebConv(8, 16, K=2, normalization="row")


@pytest.mark.parametrize("K", [1, 4])
@pytest.mark.parametrize("normalization", ["sym", "rw", None])
def test_cheb_conv_gradients(K, normalization):
    edges, x, weights = make_graph()
    layer = ChebConv(DIM, 7, K=K, normalization=normalization)

    def loss_fn(features):
        out = layer(mx.array(edges), features, mx.array(weights))
        return out.square().sum()

    loss, grads = nn.value_and_grad(layer, loss_fn)(mx.array(x))
    grads = dict(tree_flatten(grads))

    assert np.isfinite(loss.item())
    assert set(grads) == set(dict(tree_flatten(layer.trainable_parameters())))
    assert set(grads) == {f"lins.{k}.weight" for k in range(K)} | {"bias"}
    for name, grad in grads.items():
        grad = np.array(grad)
        assert np.isfinite(grad).all(), name
        assert np.abs(grad).max() > 0, name


@pytest.mark.skipif(not mp.HAS_FUSED_KERNELS, reason="mlx_cluster SpMM not installed")
@pytest.mark.parametrize("normalization", ["sym", "rw", None])
@pytest.mark.parametrize("weighted", [False, True])
def test_cheb_conv_fused_matches_gather_scatter(normalization, weighted, monkeypatch):
    calls = []
    original = MessagePassing.message_and_aggregate

    def spy(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(MessagePassing, "message_and_aggregate", spy)

    edges, x, weights = make_graph()
    layer = ChebConv(DIM, 7, K=4, normalization=normalization)
    args = (mx.array(edges), mx.array(x), mx.array(weights) if weighted else None)

    fused = layer(*args)
    assert len(calls) == 3  # K - 1 SpMMs
    with fused_message_passing_mode(False):
        reference = layer(*args)
    assert len(calls) == 3

    assert_close(fused, reference, FUSED_TOL)
