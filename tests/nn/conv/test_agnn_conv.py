import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch
import torch_geometric.nn as pyg_nn
from mlx.utils import tree_flatten

import mlx_graphs.nn.message_passing as mp
from mlx_graphs.nn import fused_message_passing_mode
from mlx_graphs.nn.conv.agnn_conv import AGNNConv

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
    return edges, x


@pytest.mark.parametrize("requires_grad", [True, False])
@pytest.mark.parametrize("add_self_loops", [True, False])
def test_agnn_conv_shape(requires_grad, add_self_loops):
    layer = AGNNConv(requires_grad=requires_grad, add_self_loops=add_self_loops)
    edge_index = mx.array([[0, 1, 2, 3, 50], [0, 0, 1, 1, 99]])

    out = layer(edge_index, mx.random.normal((100, DIM)))

    assert out.shape == (100, DIM)
    trainable = dict(tree_flatten(layer.trainable_parameters()))
    assert set(trainable) == ({"beta"} if requires_grad else set())


@pytest.mark.parametrize("requires_grad", [True, False])
@pytest.mark.parametrize("add_self_loops", [True, False])
def test_agnn_conv_matches_pyg(requires_grad, add_self_loops):
    edges, x = make_graph()
    ref = pyg_nn.AGNNConv(requires_grad=requires_grad, add_self_loops=add_self_loops)
    layer = AGNNConv(requires_grad=requires_grad, add_self_loops=add_self_loops)
    if requires_grad:
        ref.beta.data.fill_(2.5)
        layer.beta = mx.array([2.5])

    expected = ref(torch.tensor(x), torch.tensor(edges))
    out = layer(mx.array(edges), mx.array(x))

    assert_close(out, expected.detach().numpy(), PYG_TOL)


def test_agnn_conv_gradients():
    edges, x = make_graph()
    layer = AGNNConv()

    def loss_fn(features):
        return layer(mx.array(edges), features).square().sum()

    loss, grads = nn.value_and_grad(layer, loss_fn)(mx.array(x))
    grads = dict(tree_flatten(grads))
    _, input_grad = mx.value_and_grad(loss_fn)(mx.array(x))

    assert np.isfinite(loss.item())
    assert set(grads) == {"beta"}
    assert np.isfinite(np.array(grads["beta"])).all()
    assert np.abs(np.array(grads["beta"])).max() > 0
    # The masked self-loops must not produce NaNs in the backward pass.
    assert np.isfinite(np.array(input_grad)).all()


@pytest.mark.skipif(not mp.HAS_FUSED_KERNELS, reason="mlx_cluster SpMM not installed")
def test_agnn_conv_fused_matches_gather_scatter():
    edges, x = make_graph()
    layer = AGNNConv()

    fused = layer(mx.array(edges), mx.array(x))
    with fused_message_passing_mode(False):
        reference = layer(mx.array(edges), mx.array(x))

    assert_close(fused, reference, FUSED_TOL)


def _zero_row_graph():
    """A small graph and features where node 2's features are all zero (as
    after a ReLU or dropout)."""
    edge_index = np.array([[0, 1, 2, 3, 1, 2], [1, 2, 3, 0, 0, 0]])
    x = np.random.default_rng(0).normal(size=(4, 6)).astype(np.float32)
    x[2] = 0.0
    return edge_index, x


def _grads_match_pyg(mlx_fn, torch_fn, x):
    """Input gradients of sum(f(x) ** 2) in both frameworks."""
    import torch

    g = mx.grad(lambda h: (mlx_fn(h) ** 2).sum())(mx.array(x))
    h = torch.from_numpy(x).requires_grad_()
    (torch_fn(h) ** 2).sum().backward()
    assert mx.all(mx.isfinite(g)).item(), "NaN / inf gradient"
    np.testing.assert_allclose(np.array(g), h.grad.numpy(), rtol=2e-2, atol=2e-3)


def test_agnn_conv_zero_features_gradient_matches_pyg():
    """Regression: the cosine similarity divided by sqrt(sum(x ** 2)), whose
    derivative is infinite at zero: a node with all-zero features gave NaN
    gradients (seen when training on PubMed). PyG's are finite."""
    edge_index, x = _zero_row_graph()
    conv = AGNNConv()
    pyg_conv = pyg_nn.AGNNConv()
    ei_m, ei_t = mx.array(edge_index), torch.from_numpy(edge_index)
    _grads_match_pyg(lambda h: conv(ei_m, h), lambda h: pyg_conv(h, ei_t), x)
