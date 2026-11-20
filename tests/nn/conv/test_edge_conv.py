import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch
from mlx.utils import tree_flatten
from torch_geometric.nn import EdgeConv as PyGEdgeConv

from mlx_graphs.nn.conv.edge_conv import EdgeConv

# MLX's GPU matmul is only accurate to ~1e-3 relative on some machines, while a
# wrong formula gives O(1) errors with these O(1) inputs.
TOL = dict(rtol=2e-2, atol=2e-2)


def _random_graph(rng, num_src=12, num_dst=12, num_edges=40):
    """Random graph with duplicate edges and isolated nodes (the last 3 nodes
    of each side never appear in an edge)."""
    src = rng.integers(0, num_src - 3, size=num_edges)
    dst = rng.integers(0, num_dst - 3, size=num_edges)
    # duplicate the first 5 edges
    src = np.concatenate([src, src[:5]])
    dst = np.concatenate([dst, dst[:5]])
    return np.stack([src, dst]).astype(np.int64)


def _make_mlp(in_dim=6, hidden_dim=16, out_dim=8):
    return nn.Sequential(
        nn.Linear(2 * in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    )


def _make_pyg_conv(mlp, **kwargs):
    """PyG EdgeConv with the same MLP (and weights) as the MLX ``mlp``."""
    hidden_dim, in_dim = mlp.layers[0].weight.shape
    out_dim = mlp.layers[2].weight.shape[0]
    torch_mlp = torch.nn.Sequential(
        torch.nn.Linear(in_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, out_dim),
    )
    # PyG's EdgeConv re-initializes the MLP in its constructor, so the weights
    # are copied afterwards.
    pyg_conv = PyGEdgeConv(torch_mlp, **kwargs)
    with torch.no_grad():
        for idx in (0, 2):
            torch_mlp[idx].weight.copy_(
                torch.from_numpy(np.array(mlp.layers[idx].weight))
            )
            torch_mlp[idx].bias.copy_(torch.from_numpy(np.array(mlp.layers[idx].bias)))
    return pyg_conv


def _assert_finite_grads(model, loss_fn):
    _, grads = nn.value_and_grad(model, loss_fn)()
    params = dict(tree_flatten(model.trainable_parameters()))
    grads = dict(tree_flatten(grads))
    assert set(grads) == set(params)
    for name, grad in grads.items():
        assert grad.shape == params[name].shape, name
        assert mx.all(mx.isfinite(grad)).item(), f"non-finite gradient for {name}"
    assert any(mx.any(grad != 0).item() for grad in grads.values())


@pytest.mark.parametrize("aggr", ["max", "add", "mean", "min"])
def test_edge_conv_shapes(aggr):
    rng = np.random.default_rng(0)
    edge_index = mx.array(_random_graph(rng))
    node_features = mx.array(rng.normal(size=(12, 6)).astype(np.float32))

    mlp = _make_mlp()
    out = EdgeConv(mlp, aggr=aggr)(edge_index, node_features)

    assert out.shape == (12, 8)
    assert mx.all(mx.isfinite(out)).item()
    # nodes 9, 10, 11 have no incoming edge
    assert mx.all(out[9:] == 0).item()


def test_edge_conv_default_aggr_is_max():
    mlp = _make_mlp()
    assert EdgeConv(mlp).aggr == "max"
    with pytest.raises(ValueError):
        EdgeConv(mlp, aggr="softmax")


@pytest.mark.parametrize("aggr", [None, "max", "add", "mean", "min"])
def test_edge_conv_matches_pyg(aggr):
    mx.random.seed(0)
    rng = np.random.default_rng(1)
    edge_index = _random_graph(rng)
    x = rng.normal(size=(12, 6)).astype(np.float32)

    mlp = _make_mlp()
    kwargs = {} if aggr is None else {"aggr": aggr}

    with torch.no_grad():
        expected = _make_pyg_conv(mlp, **kwargs)(
            torch.from_numpy(x), torch.from_numpy(edge_index)
        ).numpy()

    out = EdgeConv(mlp, **kwargs)(mx.array(edge_index), mx.array(x))

    assert out.shape == expected.shape
    np.testing.assert_allclose(np.array(out), expected, **TOL)
    # sanity check: the comparison is not trivially satisfied
    assert np.abs(expected).max() > 0.2


def test_edge_conv_bipartite_matches_pyg():
    mx.random.seed(0)
    rng = np.random.default_rng(2)
    edge_index = _random_graph(rng, num_src=10, num_dst=7, num_edges=30)
    x_src = rng.normal(size=(10, 6)).astype(np.float32)
    x_dst = rng.normal(size=(7, 6)).astype(np.float32)

    mlp = _make_mlp()

    with torch.no_grad():
        expected = _make_pyg_conv(mlp)(
            (torch.from_numpy(x_src), torch.from_numpy(x_dst)),
            torch.from_numpy(edge_index),
        ).numpy()

    out = EdgeConv(mlp)(mx.array(edge_index), (mx.array(x_src), mx.array(x_dst)))

    assert out.shape == (7, 8)
    np.testing.assert_allclose(np.array(out), expected, **TOL)


@pytest.mark.parametrize("aggr", ["max", "mean"])
def test_edge_conv_gradients(aggr):
    mx.random.seed(0)
    rng = np.random.default_rng(3)
    edge_index = mx.array(_random_graph(rng))
    node_features = mx.array(rng.normal(size=(12, 6)).astype(np.float32))

    mlp = _make_mlp()
    conv = EdgeConv(mlp, aggr=aggr)

    def loss_fn():
        return (conv(edge_index, node_features) ** 2).sum()

    _assert_finite_grads(conv, loss_fn)
