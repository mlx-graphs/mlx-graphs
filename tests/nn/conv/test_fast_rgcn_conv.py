import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch
from mlx.utils import tree_flatten
from torch_geometric.nn import FastRGCNConv as PyGFastRGCNConv

from mlx_graphs.nn.conv.fast_rgcn_conv import FastRGCNConv
from mlx_graphs.nn.conv.rgcn_conv import RGCNConv

# MLX's GPU matmul is only accurate to ~1e-3 relative on some machines, while a
# wrong formula gives O(1) errors with these O(1) inputs.
TOL = dict(rtol=2e-2, atol=2e-2)

NUM_RELATIONS = 5


def _random_graph(rng, num_src=12, num_dst=12, num_edges=60):
    """Random multi-relational graph with duplicate edges, isolated nodes and
    one relation without any edge (relation 2)."""
    src = rng.integers(0, num_src - 3, size=num_edges)
    dst = rng.integers(0, num_dst - 3, size=num_edges)
    edge_type = rng.choice([0, 1, 3, 4], size=num_edges)
    src = np.concatenate([src, src[:10]])
    dst = np.concatenate([dst, dst[:10]])
    edge_type = np.concatenate([edge_type, edge_type[:5], (edge_type[5:10] + 1) % 2])
    return np.stack([src, dst]).astype(np.int64), edge_type.astype(np.int64)


def _copy_weights(conv, pyg_conv, rng):
    for name in ("weight", "comp", "root"):
        if name in conv:
            setattr(conv, name, mx.array(getattr(pyg_conv, name).detach().numpy()))
    if "bias" in conv:
        bias = rng.normal(size=pyg_conv.bias.shape).astype(np.float32)
        with torch.no_grad():
            pyg_conv.bias.copy_(torch.from_numpy(bias))
        conv.bias = mx.array(bias)


@pytest.mark.parametrize(
    "kwargs",
    [dict(), dict(num_bases=2), dict(root_weight=False, bias=False), dict(aggr="add")],
)
def test_fast_rgcn_conv_shapes(kwargs):
    rng = np.random.default_rng(0)
    edge_index, edge_type = _random_graph(rng)
    x = mx.array(rng.normal(size=(12, 6)).astype(np.float32))

    conv = FastRGCNConv(6, 8, NUM_RELATIONS, **kwargs)
    out = conv(mx.array(edge_index), x, mx.array(edge_type))
    assert out.shape == (12, 8)
    assert isinstance(conv, RGCNConv)

    with pytest.raises(ValueError):
        FastRGCNConv(6, 8, NUM_RELATIONS, aggr="max")


@pytest.mark.parametrize("aggr", ["mean", "add"])
@pytest.mark.parametrize("num_bases", [None, 2])
@pytest.mark.parametrize("bipartite", [False, True])
def test_fast_rgcn_conv_matches_pyg(aggr, num_bases, bipartite):
    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    num_dst = 7 if bipartite else 12
    edge_index, edge_type = _random_graph(rng, num_src=12, num_dst=num_dst)
    x = rng.normal(size=(12, 6)).astype(np.float32)
    x_dst = rng.normal(size=(num_dst, 4)).astype(np.float32)
    in_dim = (6, 4) if bipartite else 6

    pyg_conv = PyGFastRGCNConv(in_dim, 8, NUM_RELATIONS, num_bases=num_bases, aggr=aggr)
    conv = FastRGCNConv(in_dim, 8, NUM_RELATIONS, num_bases=num_bases, aggr=aggr)
    _copy_weights(conv, pyg_conv, rng)

    with torch.no_grad():
        torch_x = torch.from_numpy(x)
        if bipartite:
            torch_x = (torch_x, torch.from_numpy(x_dst))
        expected = pyg_conv(
            torch_x, torch.from_numpy(edge_index), torch.from_numpy(edge_type)
        ).numpy()

    mlx_x = (mx.array(x), mx.array(x_dst)) if bipartite else mx.array(x)
    out = conv(mx.array(edge_index), mlx_x, mx.array(edge_type))

    assert out.shape == expected.shape == (num_dst, 8)
    np.testing.assert_allclose(np.array(out), expected, **TOL)
    assert np.abs(expected).max() > 0.2

    # same result as RGCNConv with the same weights
    rgcn = RGCNConv(in_dim, 8, NUM_RELATIONS, num_bases=num_bases, aggr=aggr)
    rgcn.update(conv.parameters())
    np.testing.assert_allclose(
        np.array(rgcn(mx.array(edge_index), mlx_x, mx.array(edge_type))),
        np.array(out),
        **TOL,
    )


def test_fast_rgcn_conv_gradients():
    mx.random.seed(0)
    rng = np.random.default_rng(2)
    edge_index, edge_type = _random_graph(rng)
    edge_index, edge_type = mx.array(edge_index), mx.array(edge_type)
    x = mx.array(rng.normal(size=(12, 6)).astype(np.float32))
    conv = FastRGCNConv(6, 8, NUM_RELATIONS, num_bases=2)

    def loss_fn():
        return (conv(edge_index, x, edge_type) ** 2).sum()

    _, grads = nn.value_and_grad(conv, loss_fn)()
    grads = dict(tree_flatten(grads))
    assert set(grads) == {"weight", "comp", "root", "bias"}
    for name, grad in grads.items():
        assert mx.all(mx.isfinite(grad)).item(), name
        assert mx.any(grad != 0).item(), name
