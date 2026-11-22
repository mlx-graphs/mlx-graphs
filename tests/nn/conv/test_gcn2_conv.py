import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch
import torch_geometric.nn as pyg_nn
from mlx.utils import tree_flatten

import mlx_graphs.nn.message_passing as mp
from mlx_graphs.nn import fused_message_passing_mode
from mlx_graphs.nn.conv.gcn2_conv import GCN2Conv

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
    x_0 = rng.normal(size=(NUM_NODES, DIM)).astype(np.float32)
    weights = rng.uniform(0.1, 2.0, edges.shape[1]).astype(np.float32)
    return edges, x, x_0, weights


def to_mx(tensor):
    return mx.array(tensor.detach().numpy())


@pytest.mark.parametrize(
    "layer_args",
    [
        dict(alpha=0.1),
        dict(alpha=0.2, theta=0.5, layer=2),
        dict(alpha=0.1, shared_weights=False, add_self_loops=False),
        dict(alpha=0.1, normalize=False),
        dict(alpha=0.1, cached=True),
    ],
)
@pytest.mark.parametrize("weighted", [False, True])
def test_gcn2_conv_shape(layer_args, weighted):
    layer = GCN2Conv(DIM, **layer_args)
    edge_index = mx.array([[0, 1, 2, 3, 50], [0, 0, 1, 1, 99]])
    weights = mx.random.uniform(shape=(5,)) if weighted else None
    x = mx.random.normal((100, DIM))

    out = layer(edge_index, x, x, edge_weights=weights)

    assert out.shape == (100, DIM)


@pytest.mark.parametrize("theta, layer_idx", [(None, None), (0.5, 3)])
@pytest.mark.parametrize("shared_weights", [True, False])
@pytest.mark.parametrize("add_self_loops", [True, False])
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("weighted", [False, True])
def test_gcn2_conv_matches_pyg(
    theta, layer_idx, shared_weights, add_self_loops, normalize, weighted
):
    edges, x, x_0, weights = make_graph()
    args = dict(
        alpha=0.2,
        theta=theta,
        layer=layer_idx,
        shared_weights=shared_weights,
        add_self_loops=add_self_loops,
        normalize=normalize,
    )
    torch.manual_seed(0)
    ref = pyg_nn.GCN2Conv(DIM, **args)
    layer = GCN2Conv(DIM, **args)
    layer.weight1 = to_mx(ref.weight1)
    if not shared_weights:
        layer.weight2 = to_mx(ref.weight2)

    expected = ref(
        torch.tensor(x),
        torch.tensor(x_0),
        torch.tensor(edges),
        torch.tensor(weights) if weighted else None,
    )
    out = layer(
        mx.array(edges),
        mx.array(x),
        mx.array(x_0),
        mx.array(weights) if weighted else None,
    )

    assert_close(out, expected.detach().numpy(), PYG_TOL)


def test_gcn2_conv_theta_requires_layer():
    with pytest.raises(ValueError):
        GCN2Conv(DIM, alpha=0.1, theta=0.5)


def test_gcn2_conv_cached():
    edges, x, x_0, weights = make_graph()
    layer = GCN2Conv(DIM, alpha=0.1, cached=True)
    first = layer(mx.array(edges), mx.array(x), mx.array(x_0), mx.array(weights))
    assert layer._cached_graph is not None
    assert set(dict(tree_flatten(layer.parameters()))) == {"weight1"}

    # The cached graph is used instead of the new one.
    again = layer(mx.array([[0], [1]]), mx.array(x), mx.array(x_0))
    np.testing.assert_allclose(np.array(again), np.array(first), rtol=1e-5, atol=1e-5)

    layer.clear_cache()
    assert layer._cached_graph is None


@pytest.mark.parametrize("shared_weights", [True, False])
def test_gcn2_conv_gradients(shared_weights):
    edges, x, x_0, weights = make_graph()
    layer = GCN2Conv(DIM, alpha=0.1, theta=0.5, layer=1, shared_weights=shared_weights)

    def loss_fn(features):
        out = layer(mx.array(edges), features, mx.array(x_0), mx.array(weights))
        return out.square().sum()

    loss, grads = nn.value_and_grad(layer, loss_fn)(mx.array(x))
    grads = dict(tree_flatten(grads))

    assert np.isfinite(loss.item())
    expected = {"weight1"} if shared_weights else {"weight1", "weight2"}
    assert set(grads) == expected
    for name, grad in grads.items():
        grad = np.array(grad)
        assert np.isfinite(grad).all(), name
        assert np.abs(grad).max() > 0, name


@pytest.mark.skipif(not mp.HAS_FUSED_KERNELS, reason="mlx_cluster SpMM not installed")
@pytest.mark.parametrize("weighted", [False, True])
def test_gcn2_conv_fused_matches_gather_scatter(weighted):
    edges, x, x_0, weights = make_graph()
    layer = GCN2Conv(DIM, alpha=0.1)
    args = (
        mx.array(edges),
        mx.array(x),
        mx.array(x_0),
        mx.array(weights) if weighted else None,
    )

    fused = layer(*args)
    with fused_message_passing_mode(False):
        reference = layer(*args)

    assert_close(fused, reference, FUSED_TOL)
