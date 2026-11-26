import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch
from mlx.utils import tree_flatten
from torch_geometric.nn import TransformerConv as PyGTransformerConv

from mlx_graphs.nn import fused_message_passing_mode
from mlx_graphs.nn.conv.transformer_conv import TransformerConv
from mlx_graphs.nn.message_passing import HAS_EDGE_SCORES

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


def _copy_linear(mlx_linear, torch_linear):
    mlx_linear.weight = mx.array(torch_linear.weight.detach().numpy())
    if torch_linear.bias is not None:
        mlx_linear.bias = mx.array(torch_linear.bias.detach().numpy())


def _copy_weights(conv, pyg_conv):
    for name in ("lin_key", "lin_query", "lin_value", "lin_edge", "lin_skip"):
        if name in conv:
            _copy_linear(getattr(conv, name), getattr(pyg_conv, name))
    if "lin_beta" in conv:
        _copy_linear(conv.lin_beta, pyg_conv.lin_beta)


def _assert_finite_grads(model, loss_fn):
    _, grads = nn.value_and_grad(model, loss_fn)()
    params = dict(tree_flatten(model.trainable_parameters()))
    grads = dict(tree_flatten(grads))
    assert set(grads) == set(params)
    for name, grad in grads.items():
        assert grad.shape == params[name].shape, name
        assert mx.all(mx.isfinite(grad)).item(), f"non-finite gradient for {name}"
    assert any(mx.any(grad != 0).item() for grad in grads.values())


@pytest.mark.parametrize(
    "kwargs, expected_dim",
    [
        (dict(), 8),
        (dict(heads=3), 24),
        (dict(heads=3, concat=False), 8),
        (dict(heads=2, beta=True), 16),
        (dict(heads=2, edge_features_dim=5), 16),
        (dict(heads=2, root_weight=False, bias=False), 16),
        (dict(heads=2, dropout=0.5), 16),
    ],
)
def test_transformer_conv_shapes(kwargs, expected_dim):
    rng = np.random.default_rng(0)
    edge_index = mx.array(_random_graph(rng))
    node_features = mx.array(rng.normal(size=(12, 6)).astype(np.float32))
    edge_features = None
    if "edge_features_dim" in kwargs:
        edge_features = mx.array(
            rng.normal(size=(edge_index.shape[1], 5)).astype(np.float32)
        )

    conv = TransformerConv(6, 8, **kwargs)
    out = conv(edge_index, node_features, edge_features)

    assert out.shape == (12, expected_dim)
    assert mx.all(mx.isfinite(out)).item()


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(),
        dict(heads=3),
        dict(heads=3, concat=False),
        dict(heads=2, beta=True),
        dict(heads=2, beta=True, concat=False),
        dict(heads=2, edge_dim=5),
        dict(heads=2, edge_dim=5, concat=False, beta=True),
        dict(heads=2, bias=False),
        dict(heads=2, root_weight=False),
        dict(heads=2, root_weight=False, beta=True),  # beta ignored without root
        dict(heads=2, aggr="mean"),
        dict(heads=2, aggr="max"),
    ],
)
def test_transformer_conv_matches_pyg(kwargs):
    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    edge_index = _random_graph(rng)
    x = rng.normal(size=(12, 6)).astype(np.float32)
    edge_dim = kwargs.get("edge_dim")
    edge_attr = None
    if edge_dim is not None:
        edge_attr = rng.normal(size=(edge_index.shape[1], edge_dim)).astype(np.float32)

    pyg_conv = PyGTransformerConv(6, 8, **kwargs).eval()
    with torch.no_grad():
        expected = pyg_conv(
            torch.from_numpy(x),
            torch.from_numpy(edge_index),
            None if edge_attr is None else torch.from_numpy(edge_attr),
        ).numpy()

    mlx_kwargs = dict(kwargs)
    mlx_kwargs["edge_features_dim"] = mlx_kwargs.pop("edge_dim", None)
    conv = TransformerConv(6, 8, **mlx_kwargs)
    _copy_weights(conv, pyg_conv)

    out = conv(
        mx.array(edge_index),
        mx.array(x),
        None if edge_attr is None else mx.array(edge_attr),
    )

    assert out.shape == expected.shape
    np.testing.assert_allclose(np.array(out), expected, **TOL)
    # sanity check: the comparison is not trivially satisfied
    assert np.abs(expected).max() > 0.2


def test_transformer_conv_isolated_nodes_only_get_skip():
    mx.random.seed(0)
    rng = np.random.default_rng(2)
    edge_index = mx.array(_random_graph(rng))
    node_features = mx.array(rng.normal(size=(12, 6)).astype(np.float32))

    conv = TransformerConv(6, 8, heads=2)
    out = conv(edge_index, node_features)
    skip = conv.lin_skip(node_features)

    # nodes 9, 10, 11 have no incoming edge
    np.testing.assert_allclose(np.array(out[9:]), np.array(skip[9:]), **TOL)

    conv = TransformerConv(6, 8, heads=2, root_weight=False)
    out = conv(edge_index, node_features)
    assert mx.all(out[9:] == 0).item()


def test_transformer_conv_bipartite_matches_pyg():
    torch.manual_seed(0)
    rng = np.random.default_rng(3)
    edge_index = _random_graph(rng, num_src=10, num_dst=7, num_edges=30)
    x_src = rng.normal(size=(10, 6)).astype(np.float32)
    x_dst = rng.normal(size=(7, 4)).astype(np.float32)
    edge_attr = rng.normal(size=(edge_index.shape[1], 3)).astype(np.float32)

    pyg_conv = PyGTransformerConv((6, 4), 8, heads=2, edge_dim=3, beta=True).eval()
    with torch.no_grad():
        expected = pyg_conv(
            (torch.from_numpy(x_src), torch.from_numpy(x_dst)),
            torch.from_numpy(edge_index),
            torch.from_numpy(edge_attr),
        ).numpy()

    conv = TransformerConv((6, 4), 8, heads=2, edge_features_dim=3, beta=True)
    _copy_weights(conv, pyg_conv)
    out = conv(
        mx.array(edge_index), (mx.array(x_src), mx.array(x_dst)), mx.array(edge_attr)
    )

    assert out.shape == (7, 16)
    np.testing.assert_allclose(np.array(out), expected, **TOL)


def test_transformer_conv_dropout():
    rng = np.random.default_rng(4)
    edge_index = mx.array(_random_graph(rng))
    node_features = mx.array(rng.normal(size=(12, 6)).astype(np.float32))

    mx.random.seed(0)
    conv = TransformerConv(6, 8, heads=2, dropout=0.5)
    reference = TransformerConv(6, 8, heads=2)
    conv.update(reference.parameters())

    # dropout is only active in training mode
    conv.eval()
    np.testing.assert_allclose(
        np.array(conv(edge_index, node_features)),
        np.array(reference(edge_index, node_features)),
        **TOL,
    )

    conv.train()
    out = conv(edge_index, node_features)
    assert out.shape == (12, 16)
    assert mx.all(mx.isfinite(out)).item()
    assert not np.allclose(
        np.array(out), np.array(reference(edge_index, node_features)), atol=1e-3
    )


def test_transformer_conv_edge_features_validation():
    edge_index = mx.array([[0, 1, 2], [1, 2, 0]])
    node_features = mx.random.normal((3, 6))

    with pytest.raises(ValueError):
        TransformerConv(6, 8)(edge_index, node_features, mx.random.normal((3, 5)))
    with pytest.raises(ValueError):
        TransformerConv(6, 8, edge_features_dim=5)(edge_index, node_features)

    # 1D edge features are treated as a single feature per edge
    conv = TransformerConv(6, 8, edge_features_dim=1)
    assert conv(edge_index, node_features, mx.random.normal((3,))).shape == (3, 8)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(heads=2),
        dict(heads=2, concat=False, beta=True, edge_features_dim=5),
        dict(heads=2, root_weight=False, bias=False),
    ],
)
def test_transformer_conv_gradients(kwargs):
    mx.random.seed(0)
    rng = np.random.default_rng(5)
    edge_index = mx.array(_random_graph(rng))
    node_features = mx.array(rng.normal(size=(12, 6)).astype(np.float32))
    edge_features = None
    if "edge_features_dim" in kwargs:
        edge_features = mx.array(
            rng.normal(size=(edge_index.shape[1], 5)).astype(np.float32)
        )

    conv = TransformerConv(6, 8, **kwargs)

    def loss_fn():
        return (conv(edge_index, node_features, edge_features) ** 2).sum()

    _assert_finite_grads(conv, loss_fn)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(),
        dict(heads=3),
        dict(heads=3, concat=False),
        dict(heads=2, beta=True),
        dict(heads=2, edge_features_dim=5),
        dict(heads=2, aggr="mean"),
    ],
)
@pytest.mark.parametrize("bipartite", [False, True])
def test_transformer_conv_fused_matches_gather_scatter(kwargs, bipartite):
    """The fused path (edge_scores + SpMM) gives the outputs and gradients of
    gather-scatter, which is what the layer computes without mlx_cluster."""
    rng = np.random.default_rng(6)
    num_src, num_dst = (15, 9) if bipartite else (12, 12)
    edge_index = mx.array(_random_graph(rng, num_src, num_dst))
    src = mx.array(rng.normal(size=(num_src, 6)).astype(np.float32))
    dst = mx.array(rng.normal(size=(num_dst, 6)).astype(np.float32))
    node_features = (src, dst) if bipartite else src
    edge_features = None
    if "edge_features_dim" in kwargs:
        edge_features = mx.array(
            rng.normal(size=(edge_index.shape[1], 5)).astype(np.float32)
        )

    mx.random.seed(0)
    conv = TransformerConv(6, 8, **kwargs)

    def loss_fn():
        return (conv(edge_index, node_features, edge_features) ** 2).sum()

    out = conv(edge_index, node_features, edge_features)
    loss, grads = nn.value_and_grad(conv, loss_fn)()
    with fused_message_passing_mode(False):
        ref_out = conv(edge_index, node_features, edge_features)
        ref_loss, ref_grads = nn.value_and_grad(conv, loss_fn)()

    np.testing.assert_allclose(np.array(out), np.array(ref_out), **TOL)
    np.testing.assert_allclose(loss.item(), ref_loss.item(), rtol=1e-3)
    ref_grads = dict(tree_flatten(ref_grads))
    for name, grad in tree_flatten(grads):
        np.testing.assert_allclose(
            np.array(grad), np.array(ref_grads[name]), err_msg=name, **TOL
        )


@pytest.mark.skipif(not HAS_EDGE_SCORES, reason="needs mlx_cluster.edge_scores")
def test_transformer_conv_uses_fused_kernels(monkeypatch):
    """Without edge features, neither the scores nor the messages are
    computed per edge in Python."""
    rng = np.random.default_rng(7)
    edge_index = mx.array(_random_graph(rng))
    node_features = mx.array(rng.normal(size=(12, 6)).astype(np.float32))
    conv = TransformerConv(6, 8, heads=2)

    def fail(*args, **kwargs):
        raise AssertionError("message() called on the fused path")

    monkeypatch.setattr(conv, "message", fail)
    assert conv(edge_index, node_features).shape == (12, 16)
