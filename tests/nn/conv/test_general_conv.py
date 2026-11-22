import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch
from mlx.utils import tree_flatten
from torch_geometric.nn import GeneralConv as PyGGeneralConv

from mlx_graphs.nn.conv.general_conv import GeneralConv

# MLX's GPU matmul is only accurate to ~1e-3 relative on some machines, while a
# wrong formula gives O(1) errors with these O(1) inputs.
TOL = dict(rtol=2e-2, atol=2e-2)


def _random_graph(rng, num_src=12, num_dst=12, num_edges=40):
    """Random graph with duplicate edges and isolated nodes (the last 3 nodes
    of each side never appear in an edge)."""
    src = rng.integers(0, num_src - 3, size=num_edges)
    dst = rng.integers(0, num_dst - 3, size=num_edges)
    src = np.concatenate([src, src[:5]])
    dst = np.concatenate([dst, dst[:5]])
    return np.stack([src, dst]).astype(np.int64)


def _copy_linear(mlx_linear, torch_linear):
    mlx_linear.weight = mx.array(torch_linear.weight.detach().numpy())
    if torch_linear.bias is not None:
        mlx_linear.bias = mx.array(torch_linear.bias.detach().numpy())


def _copy_weights(conv, pyg_conv):
    for name in ("lin_msg", "lin_msg_i", "lin_edge", "lin_self"):
        if name in conv:
            _copy_linear(getattr(conv, name), getattr(pyg_conv, name))
        elif hasattr(pyg_conv, name):
            assert isinstance(getattr(pyg_conv, name), torch.nn.Identity), name
    if "att_msg" in conv:
        conv.att_msg = mx.array(pyg_conv.att_msg.detach().numpy())


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(),
        dict(heads=3),
        dict(heads=2, attention=True),
        dict(heads=2, attention=True, attention_type="dot_product"),
        dict(directed_msg=False, skip_linear=True, l2_normalize=True),
        dict(edge_features_dim=5, aggr="mean", bias=False),
    ],
)
def test_general_conv_shapes(kwargs):
    rng = np.random.default_rng(0)
    edge_index = mx.array(_random_graph(rng))
    x = mx.array(rng.normal(size=(12, 6)).astype(np.float32))
    edge_features = mx.array(
        rng.normal(size=(edge_index.shape[1], 5)).astype(np.float32)
    )

    conv = GeneralConv(6, 8, **kwargs)
    out = conv(edge_index, x, edge_features if "edge_features_dim" in kwargs else None)
    assert out.shape == (12, 8)
    assert mx.all(mx.isfinite(out)).item()


def test_general_conv_identity_skip():
    conv = GeneralConv(8, 8)
    assert "lin_self" not in conv
    assert "lin_self" in GeneralConv(8, 8, skip_linear=True)
    assert "lin_self" in GeneralConv((8, 8), 8)
    with pytest.raises(ValueError):
        GeneralConv(8, 8, attention=True, attention_type="cosine")


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(),
        dict(heads=3),
        dict(heads=2, attention=True),
        dict(heads=2, attention=True, attention_type="dot_product"),
        dict(heads=2, attention=True, directed_msg=False),
        dict(
            heads=2,
            attention=True,
            attention_type="dot_product",
            directed_msg=False,
        ),
        dict(skip_linear=True, l2_normalize=True),
        dict(aggr="mean", bias=False),
        dict(aggr="max", heads=2),
        dict(in_edge_channels=5, heads=2, attention=True),
    ],
)
@pytest.mark.parametrize("in_dim", [6, 8])
def test_general_conv_matches_pyg(kwargs, in_dim):
    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    edge_index = _random_graph(rng)
    x = rng.normal(size=(12, in_dim)).astype(np.float32)
    edge_attr = rng.normal(size=(edge_index.shape[1], 5)).astype(np.float32)
    use_edges = "in_edge_channels" in kwargs

    pyg_conv = PyGGeneralConv(in_dim, 8, **kwargs)
    mlx_kwargs = dict(kwargs)
    if use_edges:
        mlx_kwargs["edge_features_dim"] = mlx_kwargs.pop("in_edge_channels")
    conv = GeneralConv(in_dim, 8, **mlx_kwargs)
    _copy_weights(conv, pyg_conv)

    with torch.no_grad():
        expected = pyg_conv(
            torch.from_numpy(x),
            torch.from_numpy(edge_index),
            torch.from_numpy(edge_attr) if use_edges else None,
        ).numpy()

    out = conv(
        mx.array(edge_index), mx.array(x), mx.array(edge_attr) if use_edges else None
    )
    assert out.shape == expected.shape
    np.testing.assert_allclose(np.array(out), expected, **TOL)
    assert np.abs(expected).max() > 0.2


def test_general_conv_bipartite_matches_pyg():
    torch.manual_seed(0)
    rng = np.random.default_rng(2)
    edge_index = _random_graph(rng, num_src=10, num_dst=7)
    x_src = rng.normal(size=(10, 6)).astype(np.float32)
    x_dst = rng.normal(size=(7, 4)).astype(np.float32)

    pyg_conv = PyGGeneralConv((6, 4), 8, heads=2, attention=True)
    conv = GeneralConv((6, 4), 8, heads=2, attention=True)
    _copy_weights(conv, pyg_conv)

    with torch.no_grad():
        expected = pyg_conv(
            (torch.from_numpy(x_src), torch.from_numpy(x_dst)),
            torch.from_numpy(edge_index),
        ).numpy()
    out = conv(mx.array(edge_index), (mx.array(x_src), mx.array(x_dst)))
    assert out.shape == (7, 8)
    np.testing.assert_allclose(np.array(out), expected, **TOL)


def test_general_conv_gradients():
    mx.random.seed(0)
    rng = np.random.default_rng(3)
    edge_index = mx.array(_random_graph(rng))
    x = mx.array(rng.normal(size=(12, 6)).astype(np.float32))
    conv = GeneralConv(6, 8, heads=2, attention=True)

    def loss_fn():
        return (conv(edge_index, x) ** 2).sum()

    _, grads = nn.value_and_grad(conv, loss_fn)()
    grads = dict(tree_flatten(grads))
    params = dict(tree_flatten(conv.trainable_parameters()))
    assert set(grads) == set(params)
    for name, grad in grads.items():
        assert mx.all(mx.isfinite(grad)).item(), name
        assert mx.any(grad != 0).item(), name
