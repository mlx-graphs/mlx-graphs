import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch
from mlx.utils import tree_flatten
from torch_geometric.nn import RGATConv as PyGRGATConv

from mlx_graphs.nn.conv.rgat_conv import RGATConv

# MLX's GPU matmul is only accurate to ~1e-3 relative on some machines, while a
# wrong formula gives O(1) errors with these O(1) inputs.
TOL = dict(rtol=2e-2, atol=2e-2)

NUM_RELATIONS = 4


def _random_graph(rng, num_nodes=12, num_edges=50):
    """Random multi-relational graph with duplicate edges, isolated nodes (the
    last 3) and one relation without any edge (relation 2)."""
    src = rng.integers(0, num_nodes - 3, size=num_edges)
    dst = rng.integers(0, num_nodes - 3, size=num_edges)
    edge_type = rng.choice([0, 1, 3], size=num_edges)
    src = np.concatenate([src, src[:5]])
    dst = np.concatenate([dst, dst[:5]])
    edge_type = np.concatenate([edge_type, (edge_type[:5] + 1) % 2])
    return np.stack([src, dst]).astype(np.int64), edge_type.astype(np.int64)


def _copy_weights(conv, pyg_conv, rng):
    """Copies PyG's parameters, after giving random values to those that PyG
    initializes to constants (or, for ``l2``, leaves uninitialized)."""
    with torch.no_grad():
        for name in ("bias", "w", "l1", "b1", "l2", "b2"):
            param = getattr(pyg_conv, name)
            if param is not None:
                param.copy_(torch.from_numpy(rng.normal(size=param.shape)))
    for name in (
        "q",
        "k",
        "e",
        "att",
        "basis",
        "weight",
        "bias",
        "w",
        "l1",
        "b1",
        "l2",
        "b2",
    ):
        if name in conv:
            setattr(conv, name, mx.array(getattr(pyg_conv, name).detach().numpy()))
    if "lin_edge" in conv:
        conv.lin_edge.weight = mx.array(pyg_conv.lin_edge.weight.detach().numpy())


def _force_path(monkeypatch, path):
    if path is not None:
        monkeypatch.setattr(RGATConv, "_use_node_path", lambda self, *a: path == "node")


@pytest.mark.parametrize(
    "kwargs, expected_dim",
    [
        (dict(), 8),
        (dict(heads=3), 24),
        (dict(heads=3, concat=False), 8),
        (dict(attention_mode="multiplicative-self-attention", heads=2, dim=3), 48),
        (
            dict(
                attention_mode="multiplicative-self-attention",
                heads=2,
                dim=3,
                concat=False,
            ),
            24,
        ),
        (dict(num_bases=2, mod="scaled"), 8),
        (dict(num_blocks=2, attention_mechanism="within-relation"), 8),
        (dict(edge_features_dim=3, dropout=0.2), 8),
    ],
)
def test_rgat_conv_shapes(kwargs, expected_dim):
    rng = np.random.default_rng(0)
    edge_index, edge_type = _random_graph(rng)
    x = mx.array(rng.normal(size=(12, 6)).astype(np.float32))
    edge_features = None
    if "edge_features_dim" in kwargs:
        edge_features = mx.array(
            rng.normal(size=(edge_index.shape[1], 3)).astype(np.float32)
        )

    conv = RGATConv(6, 8, NUM_RELATIONS, **kwargs)
    out, (ei, alpha) = conv(
        mx.array(edge_index),
        x,
        mx.array(edge_type),
        edge_features,
        return_attention_weights=True,
    )
    assert out.shape == (12, expected_dim)
    assert mx.all(mx.isfinite(out)).item()
    heads, dim = kwargs.get("heads", 1), kwargs.get("dim", 1)
    assert alpha.shape == (edge_index.shape[1], heads * dim)


def test_rgat_conv_invalid_arguments():
    with pytest.raises(ValueError):
        RGATConv(6, 8, 3, dim=2)  # additive attention with dim > 1
    with pytest.raises(ValueError):
        RGATConv(6, 8, 3, num_bases=2, num_blocks=2)
    with pytest.raises(ValueError):
        RGATConv(6, 8, 3, mod="additive", dropout=0.5)
    with pytest.raises(ValueError):
        RGATConv(6, 8, 3, attention_mechanism="global")
    with pytest.raises(ValueError):
        RGATConv(6, 8, 3, num_blocks=4)  # 6 is not a multiple of 4


MULT = dict(attention_mode="multiplicative-self-attention")


@pytest.mark.parametrize("path", ["node", "edge"])
@pytest.mark.parametrize(
    "kwargs",
    [
        dict(),
        dict(heads=2, concat=False),
        dict(attention_mechanism="within-relation", heads=2),
        dict(**MULT, heads=2, dim=2),
        dict(**MULT, heads=2, dim=2, concat=False),
        dict(**MULT, attention_mechanism="within-relation", dim=3),
        dict(num_bases=2, heads=2),
        dict(num_blocks=2, heads=2),
        dict(num_blocks=2, **MULT, dim=2),
        dict(mod="additive", heads=2),
        dict(mod="scaled", heads=2),
        dict(mod="f-additive", heads=2),
        dict(mod="f-scaled", heads=2),
        dict(**MULT, mod="additive", heads=2, dim=2),
        dict(**MULT, mod="scaled", heads=2, dim=2),
        dict(**MULT, mod="f-additive", dim=2),
        dict(**MULT, mod="f-scaled", dim=2),
        dict(aggr="mean", heads=2),
        dict(edge_dim=3, heads=2),
        dict(**MULT, edge_dim=3, heads=2, dim=2),
        dict(bias=False),
    ],
)
def test_rgat_conv_matches_pyg(monkeypatch, path, kwargs):
    _force_path(monkeypatch, path)
    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    edge_index, edge_type = _random_graph(rng)
    x = rng.normal(size=(12, 6)).astype(np.float32)
    edge_attr = None
    mlx_kwargs = dict(kwargs)
    if "edge_dim" in kwargs:
        edge_attr = rng.normal(size=(edge_index.shape[1], 3)).astype(np.float32)
        mlx_kwargs["edge_features_dim"] = mlx_kwargs.pop("edge_dim")

    pyg_conv = PyGRGATConv(6, 8, NUM_RELATIONS, **kwargs)
    conv = RGATConv(6, 8, NUM_RELATIONS, **mlx_kwargs)
    _copy_weights(conv, pyg_conv, rng)

    with torch.no_grad():
        expected, (_, expected_alpha) = pyg_conv(
            torch.from_numpy(x),
            torch.from_numpy(edge_index),
            torch.from_numpy(edge_type),
            None if edge_attr is None else torch.from_numpy(edge_attr),
            return_attention_weights=True,
        )

    out, (_, alpha) = conv(
        mx.array(edge_index),
        mx.array(x),
        mx.array(edge_type),
        None if edge_attr is None else mx.array(edge_attr),
        return_attention_weights=True,
    )

    assert out.shape == tuple(expected.shape)
    np.testing.assert_allclose(np.array(alpha), expected_alpha.numpy(), **TOL)
    expected = expected.numpy()
    # relative to the scale of the output (the multiplicative modes can give
    # large values)
    scale = max(1.0, np.abs(expected).max())
    np.testing.assert_allclose(
        np.array(out) / scale, expected / scale, rtol=2e-2, atol=2e-2
    )
    assert np.abs(expected).max() > 0.2


@pytest.mark.parametrize("kwargs", [dict(), dict(num_bases=2), dict(num_blocks=2)])
def test_rgat_conv_paths_agree(monkeypatch, kwargs):
    mx.random.seed(0)
    rng = np.random.default_rng(2)
    edge_index, edge_type = _random_graph(rng)
    edge_index, edge_type = mx.array(edge_index), mx.array(edge_type)
    x = mx.array(rng.normal(size=(12, 6)).astype(np.float32))
    conv = RGATConv(6, 8, NUM_RELATIONS, heads=2, **kwargs)

    _force_path(monkeypatch, "node")
    node = np.array(conv(edge_index, x, edge_type))
    _force_path(monkeypatch, "edge")
    edge = np.array(conv(edge_index, x, edge_type))
    np.testing.assert_allclose(node, edge, **TOL)


@pytest.mark.parametrize("kwargs", [dict(), dict(num_bases=2, mod="scaled")])
def test_rgat_conv_gradients(kwargs):
    mx.random.seed(0)
    rng = np.random.default_rng(3)
    edge_index, edge_type = _random_graph(rng)
    edge_index, edge_type = mx.array(edge_index), mx.array(edge_type)
    x = mx.array(rng.normal(size=(12, 6)).astype(np.float32))
    conv = RGATConv(6, 8, NUM_RELATIONS, heads=2, **kwargs)

    def loss_fn():
        return (conv(edge_index, x, edge_type) ** 2).sum()

    _, grads = nn.value_and_grad(conv, loss_fn)()
    grads = dict(tree_flatten(grads))
    params = dict(tree_flatten(conv.trainable_parameters()))
    assert set(grads) == set(params)
    for name, grad in grads.items():
        assert mx.all(mx.isfinite(grad)).item(), name
        assert mx.any(grad != 0).item(), name


@pytest.mark.parametrize("path", ["node", "edge"])
@pytest.mark.parametrize("kwargs", [dict(), dict(num_bases=2), dict(num_blocks=2)])
def test_rgat_conv_zero_edges(monkeypatch, path, kwargs):
    # Block-diagonal weights were reshaped with an inferred size, which fails
    # on empty arrays.
    _force_path(monkeypatch, path)
    conv = RGATConv(6, 8, NUM_RELATIONS, heads=2, **kwargs)
    edge_index = mx.zeros((2, 0), dtype=mx.int32)
    edge_type = mx.zeros((0,), dtype=mx.int32)
    out = conv(edge_index, mx.ones((10, 6)), edge_type)
    assert out.shape == (10, 16)
    assert mx.all(mx.isfinite(out)).item()
