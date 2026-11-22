import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch
from mlx.utils import tree_flatten
from torch_geometric.nn import GENConv as PyGGENConv

from mlx_graphs.nn.conv.gen_conv import GENConv

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


def _copy_weights(conv, pyg_conv, rng):
    for name in ("lin_src", "lin_edge", "lin_aggr_out", "lin_dst"):
        assert (name in conv) == hasattr(pyg_conv, name), name
        if name in conv:
            _copy_linear(getattr(conv, name), getattr(pyg_conv, name))

    # Linear and affine normalization layers, in order
    torch_layers = [m for m in pyg_conv.mlp if getattr(m, "weight", None) is not None]
    mlx_layers = [m for m in conv.mlp.layers if "weight" in m]
    assert len(torch_layers) == len(mlx_layers)
    for mlx_layer, torch_layer in zip(mlx_layers, torch_layers):
        assert type(mlx_layer).__name__ in type(torch_layer).__name__
        if isinstance(mlx_layer, nn.Linear):
            _copy_linear(mlx_layer, torch_layer)
        else:
            # random affine parameters, so that they are actually tested
            weight = rng.normal(size=torch_layer.weight.shape).astype(np.float32)
            bias = rng.normal(size=torch_layer.bias.shape).astype(np.float32)
            with torch.no_grad():
                torch_layer.weight.copy_(torch.from_numpy(weight))
                torch_layer.bias.copy_(torch.from_numpy(bias))
            mlx_layer.weight = mx.array(weight)
            mlx_layer.bias = mx.array(bias)


def _run_pyg(pyg_conv, x, edge_index, edge_attr=None):
    with torch.no_grad():
        if isinstance(x, tuple):
            x = tuple(torch.from_numpy(v) for v in x)
        else:
            x = torch.from_numpy(x)
        edge_attr = None if edge_attr is None else torch.from_numpy(edge_attr)
        return pyg_conv(x, torch.from_numpy(edge_index), edge_attr).numpy()


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(),
        dict(aggr="powermean", p=2.0, learn_p=True),
        dict(aggr="softmax", t=0.5, learn_t=True, msg_norm=True),
        dict(aggr="mean", norm="layer", num_layers=3),
        dict(aggr="max", norm=None, bias=True),
    ],
)
def test_gen_conv_shapes(kwargs):
    rng = np.random.default_rng(0)
    edge_index = mx.array(_random_graph(rng))
    x = mx.array(rng.normal(size=(12, 6)).astype(np.float32))

    conv = GENConv(6, 8, **kwargs)
    out = conv(edge_index, x)
    assert out.shape == (12, 8)
    assert mx.all(mx.isfinite(out)).item()

    # bipartite
    x_dst = mx.array(rng.normal(size=(12, 4)).astype(np.float32))
    conv = GENConv((6, 4), 8, **kwargs)
    assert conv(edge_index, (x, x_dst)).shape == (12, 8)


def test_gen_conv_parameters():
    conv = GENConv(8, 8)
    assert "lin_src" not in conv and "lin_dst" not in conv and "lin_edge" not in conv
    assert "t" not in dict(tree_flatten(conv.parameters()))
    conv = GENConv((6, 4), 8, edge_features_dim=3, learn_t=True)
    assert "lin_src" in conv and "lin_dst" in conv and "lin_edge" in conv
    assert conv.t.shape == (1,)
    assert "bias" not in conv.lin_src  # bias=False by default, as in PyG
    with pytest.raises(ValueError):
        GENConv(8, 8, aggr="std")

    # A list of aggregations (once unsupported) is concatenated and projected
    # back to out_features_dim, as in PyG.
    conv = GENConv(8, 8, aggr=["softmax", "mean"])
    assert conv.lin_aggr_out.weight.shape == (8, 16)
    assert "lin_aggr_out" not in GENConv(8, 8, aggr="mean")


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(),
        dict(aggr="softmax", t=0.3, learn_t=True),
        dict(aggr="softmax_sg"),
        dict(aggr="powermean", p=2.5, learn_p=True),
        dict(aggr="power", p=1.0),
        dict(aggr="add"),
        dict(aggr="mean"),
        dict(aggr="max"),
        dict(msg_norm=True, learn_msg_scale=True),
        dict(norm="layer", num_layers=3, expansion=3),
        dict(norm="instance"),
        dict(norm=None, bias=True),
        # lists of aggregations: concatenated, then projected by lin_aggr_out
        dict(aggr=["softmax", "mean"]),
        dict(aggr=["mean", "max", "std"], msg_norm=True, bias=True),
        dict(aggr=["powermean", "min"], norm="layer"),
    ],
)
@pytest.mark.parametrize("in_dim", [6, 8])
def test_gen_conv_matches_pyg(kwargs, in_dim):
    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    edge_index = _random_graph(rng)
    x = rng.normal(size=(12, in_dim)).astype(np.float32)

    pyg_conv = PyGGENConv(in_dim, 8, **kwargs)
    conv = GENConv(in_dim, 8, **kwargs)
    _copy_weights(conv, pyg_conv, rng)

    expected = _run_pyg(pyg_conv, x, edge_index)
    out = conv(mx.array(edge_index), mx.array(x))

    assert out.shape == expected.shape
    np.testing.assert_allclose(np.array(out), expected, **TOL)
    assert np.abs(expected).max() > 0.2


@pytest.mark.parametrize("edge_dim", [3, 8])
def test_gen_conv_edge_features_bipartite_matches_pyg(edge_dim):
    torch.manual_seed(0)
    rng = np.random.default_rng(2)
    edge_index = _random_graph(rng, num_src=10, num_dst=7)
    x_src = rng.normal(size=(10, 6)).astype(np.float32)
    x_dst = rng.normal(size=(7, 4)).astype(np.float32)
    edge_attr = rng.normal(size=(edge_index.shape[1], edge_dim)).astype(np.float32)

    kwargs = dict(edge_dim=edge_dim, msg_norm=True, bias=True)
    pyg_conv = PyGGENConv((6, 4), 8, **kwargs)
    conv = GENConv((6, 4), 8, edge_features_dim=edge_dim, msg_norm=True, bias=True)
    _copy_weights(conv, pyg_conv, rng)

    expected = _run_pyg(pyg_conv, (x_src, x_dst), edge_index, edge_attr)
    out = conv(
        mx.array(edge_index),
        (mx.array(x_src), mx.array(x_dst)),
        mx.array(edge_attr),
    )
    assert out.shape == (7, 8)
    np.testing.assert_allclose(np.array(out), expected, **TOL)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(learn_t=True),
        dict(aggr="powermean", p=2.0, learn_p=True),
        dict(msg_norm=True, learn_msg_scale=True, edge_features_dim=3),
    ],
)
def test_gen_conv_gradients(kwargs):
    mx.random.seed(0)
    rng = np.random.default_rng(3)
    edge_index = mx.array(_random_graph(rng))
    x = mx.array(rng.normal(size=(12, 6)).astype(np.float32))
    edge_features = None
    if "edge_features_dim" in kwargs:
        edge_features = mx.array(
            rng.normal(size=(edge_index.shape[1], 3)).astype(np.float32)
        )

    conv = GENConv(6, 8, **kwargs)

    def loss_fn():
        return (conv(edge_index, x, edge_features) ** 2).sum()

    _, grads = nn.value_and_grad(conv, loss_fn)()
    grads = dict(tree_flatten(grads))
    params = dict(tree_flatten(conv.trainable_parameters()))
    assert set(grads) == set(params)
    for name in ("t", "p", "msg_scale"):
        if name in params:
            assert mx.any(grads[name] != 0).item(), name
    for name, grad in grads.items():
        assert mx.all(mx.isfinite(grad)).item(), name


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


def test_gen_conv_msg_norm_zero_features_gradient_matches_pyg():
    """Regression: GENConv(msg_norm=True) gave NaN gradients for nodes with
    all-zero features (frequent after ReLU + dropout), see MessageNorm."""
    torch.manual_seed(0)
    edge_index, x = _zero_row_graph()
    kwargs = dict(msg_norm=True, learn_msg_scale=True, norm=None)
    pyg_conv = PyGGENConv(6, 6, **kwargs)
    conv = GENConv(6, 6, **kwargs)
    _copy_weights(conv, pyg_conv, np.random.default_rng(2))
    ei_m, ei_t = mx.array(edge_index), torch.from_numpy(edge_index)
    _grads_match_pyg(lambda h: conv(ei_m, h), lambda h: pyg_conv(h, ei_t), x)
