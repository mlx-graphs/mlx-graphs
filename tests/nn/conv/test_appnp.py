import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch
import torch_geometric.nn as pyg_nn
from mlx.utils import tree_flatten

import mlx_graphs.nn.message_passing as mp
from mlx_graphs.nn import MessagePassing, fused_message_passing_mode
from mlx_graphs.nn.conv.appnp import APPNP

NUM_NODES, DIM = 30, 12
# APPNP has no dense matmul, so it matches PyG to float32 precision
TOL = dict(rtol=1e-4, atol=1e-5)


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


@pytest.mark.parametrize("weighted", [False, True])
def test_appnp_shape(weighted):
    layer = APPNP(K=4, alpha=0.1)
    edge_index = mx.array([[0, 1, 2, 3, 50], [0, 0, 1, 1, 99]])
    weights = mx.random.uniform(shape=(5,)) if weighted else None

    out = layer(edge_index, mx.random.normal((100, 8)), edge_weights=weights)

    assert out.shape == (100, 8)
    assert tree_flatten(layer.parameters()) == []


@pytest.mark.parametrize("K, alpha", [(1, 0.1), (5, 0.2), (10, 0.5)])
@pytest.mark.parametrize("add_self_loops", [True, False])
@pytest.mark.parametrize("weighted", [False, True])
def test_appnp_matches_pyg(K, alpha, add_self_loops, weighted):
    edges, x, weights = make_graph()
    ref = pyg_nn.APPNP(K=K, alpha=alpha, add_self_loops=add_self_loops)
    layer = APPNP(K=K, alpha=alpha, add_self_loops=add_self_loops)

    expected = ref(
        torch.tensor(x),
        torch.tensor(edges),
        torch.tensor(weights) if weighted else None,
    )
    out = layer(mx.array(edges), mx.array(x), mx.array(weights) if weighted else None)

    np.testing.assert_allclose(np.array(out), expected.numpy(), **TOL)


def test_appnp_teleport_only_returns_input():
    edges, x, _ = make_graph()

    out = APPNP(K=3, alpha=1.0)(mx.array(edges), mx.array(x))
    np.testing.assert_allclose(np.array(out), x, rtol=1e-6, atol=1e-6)

    out = APPNP(K=0, alpha=0.1)(mx.array(edges), mx.array(x))
    np.testing.assert_allclose(np.array(out), x, rtol=1e-6, atol=1e-6)


def test_appnp_dropout_only_in_training():
    edges, x, weights = make_graph()
    args = (mx.array(edges), mx.array(x), mx.array(weights))
    reference = np.array(APPNP(K=3, alpha=0.2)(*args))
    layer = APPNP(K=3, alpha=0.2, dropout=0.5)

    layer.eval()
    np.testing.assert_allclose(np.array(layer(*args)), reference, **TOL)

    layer.train()
    mx.random.seed(0)
    dropped = np.array(layer(*args))
    assert np.isfinite(dropped).all()
    assert np.abs(dropped - reference).max() > 1e-2
    # A different set of edges is dropped on every call.
    assert np.abs(np.array(layer(*args)) - dropped).max() > 1e-2


def test_appnp_dropout_is_unbiased():
    # One step without teleport is linear in the dropped weights, so the mean
    # over many draws is the output without dropout.
    edges, x, weights = make_graph()
    args = (mx.array(edges), mx.array(x), mx.array(weights))
    reference = np.array(APPNP(K=1, alpha=0.0)(*args))
    layer = APPNP(K=1, alpha=0.0, dropout=0.5)
    layer.train()

    mx.random.seed(0)
    draws = np.stack([np.array(layer(*args)) for _ in range(1000)])

    # Within 5 standard errors of the mean, for every output entry.
    standard_error = draws.std(axis=0) / np.sqrt(len(draws))
    assert (np.abs(draws.mean(axis=0) - reference) <= 5 * standard_error + 1e-4).all()
    assert standard_error.max() > 1e-3  # edges were dropped


@pytest.mark.parametrize("dropout", [-0.1, 1.0])
def test_appnp_invalid_dropout(dropout):
    with pytest.raises(ValueError):
        APPNP(K=2, alpha=0.1, dropout=dropout)


@pytest.mark.parametrize("dropout", [0.0, 0.5])
def test_appnp_gradients(dropout):
    edges, x, weights = make_graph()

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(DIM, 7)
            self.prop = APPNP(K=4, alpha=0.1, dropout=dropout)

        def __call__(self, edge_index, features, edge_weights):
            return self.prop(edge_index, self.linear(features), edge_weights)

    model = Model()
    model.train()

    def loss_fn(features):
        out = model(mx.array(edges), features, mx.array(weights))
        return out.square().sum()

    loss, grads = nn.value_and_grad(model, loss_fn)(mx.array(x))
    grads = dict(tree_flatten(grads))

    assert np.isfinite(loss.item())
    assert set(grads) == {"linear.weight", "linear.bias"}
    for name, grad in grads.items():
        grad = np.array(grad)
        assert np.isfinite(grad).all(), name
        assert np.abs(grad).max() > 0, name

    # Gradients also flow to the input features.
    input_grad = np.array(mx.grad(loss_fn)(mx.array(x)))
    assert np.isfinite(input_grad).all()
    assert np.abs(input_grad).max() > 0


@pytest.mark.skipif(not mp.HAS_FUSED_KERNELS, reason="mlx_cluster SpMM not installed")
@pytest.mark.parametrize("weighted", [False, True])
def test_appnp_fused_matches_gather_scatter(weighted, monkeypatch):
    calls = []
    original = MessagePassing.message_and_aggregate

    def spy(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(MessagePassing, "message_and_aggregate", spy)

    edges, x, weights = make_graph()
    layer = APPNP(K=5, alpha=0.2)
    args = (mx.array(edges), mx.array(x), mx.array(weights) if weighted else None)

    fused = layer(*args)
    assert len(calls) == 5  # one SpMM per propagation step
    with fused_message_passing_mode(False):
        reference = layer(*args)
    assert len(calls) == 5

    np.testing.assert_allclose(np.array(fused), np.array(reference), **TOL)
