import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch
from mlx.utils import tree_flatten, tree_unflatten
from torch_geometric.nn import MixHopConv as PyGMixHopConv

from mlx_graphs.nn import fused_message_passing_mode
from mlx_graphs.nn.conv.mixhop_conv import MixHopConv

# MLX's GPU matmul is only accurate to ~1e-3 relative on some machines, while a
# wrong formula gives O(1) errors with these O(1) inputs.
TOL = dict(rtol=2e-2, atol=2e-2)


def _random_graph(rng, num_nodes=12, num_edges=40):
    """Random graph with duplicate edges, two (single) self-loops and isolated
    nodes (the last 3 nodes)."""
    src = rng.integers(0, num_nodes - 3, size=num_edges)
    dst = (src + rng.integers(1, num_nodes - 3, size=num_edges)) % (num_nodes - 3)
    src = np.concatenate([src, src[:5], [1, 4]])
    dst = np.concatenate([dst, dst[:5], [1, 4]])
    return np.stack([src, dst]).astype(np.int64)


def _copy_weights(conv, pyg_conv, rng):
    for p in conv.powers:
        conv.lins[p].weight = mx.array(pyg_conv.lins[p].weight.detach().numpy())
    if "bias" in conv:
        bias = rng.normal(size=pyg_conv.bias.shape).astype(np.float32)
        with torch.no_grad():
            pyg_conv.bias.copy_(torch.from_numpy(bias))
        conv.bias = mx.array(bias)


@pytest.mark.parametrize(
    "kwargs, expected_dim",
    [
        (dict(), 24),
        (dict(powers=[1, 3]), 16),
        (dict(powers=[0]), 8),
        (dict(add_self_loops=False, bias=False), 24),
    ],
)
def test_mixhop_conv_shapes(kwargs, expected_dim):
    rng = np.random.default_rng(0)
    edge_index = mx.array(_random_graph(rng))
    x = mx.array(rng.normal(size=(12, 6)).astype(np.float32))

    conv = MixHopConv(6, 8, **kwargs)
    out = conv(edge_index, x)
    assert out.shape == (12, expected_dim)
    assert mx.all(mx.isfinite(out)).item()
    # only the used powers have parameters
    names = {k.split(".")[1] for k, _ in tree_flatten(conv.parameters()) if "lins" in k}
    assert names == {str(p) for p in conv.powers}


@pytest.mark.parametrize("fused", [True, False])
@pytest.mark.parametrize(
    "kwargs",
    [
        dict(),
        dict(powers=[1, 3]),
        dict(powers=[2, 0, 1]),
        dict(add_self_loops=False),
        dict(bias=False),
    ],
)
@pytest.mark.parametrize("weighted", [False, True])
def test_mixhop_conv_matches_pyg(fused, kwargs, weighted):
    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    edge_index = _random_graph(rng)
    x = rng.normal(size=(12, 6)).astype(np.float32)
    edge_weight = (
        rng.uniform(0.5, 2.0, size=edge_index.shape[1]).astype(np.float32)
        if weighted
        else None
    )

    pyg_conv = PyGMixHopConv(6, 8, **kwargs)
    conv = MixHopConv(6, 8, **kwargs)
    _copy_weights(conv, pyg_conv, rng)

    with torch.no_grad():
        expected = pyg_conv(
            torch.from_numpy(x),
            torch.from_numpy(edge_index),
            None if edge_weight is None else torch.from_numpy(edge_weight),
        ).numpy()

    with fused_message_passing_mode(fused):
        out = conv(
            mx.array(edge_index),
            mx.array(x),
            None if edge_weight is None else mx.array(edge_weight),
        )

    assert out.shape == expected.shape
    np.testing.assert_allclose(np.array(out), expected, **TOL)
    assert np.abs(expected).max() > 0.2


def test_mixhop_conv_gradients():
    mx.random.seed(0)
    rng = np.random.default_rng(2)
    edge_index = mx.array(_random_graph(rng))
    x = mx.array(rng.normal(size=(12, 6)).astype(np.float32))
    conv = MixHopConv(6, 8, powers=[0, 2])

    def loss_fn():
        return (conv(edge_index, x) ** 2).sum()

    _, grads = nn.value_and_grad(conv, loss_fn)()
    grads = dict(tree_flatten(grads))
    assert set(grads) == {"lins.0.weight", "lins.2.weight", "bias"}
    for name, grad in grads.items():
        assert mx.all(mx.isfinite(grad)).item(), name
        assert mx.any(grad != 0).item(), name


@pytest.mark.parametrize("powers", [[0, 1, 2], [1, 3]])
def test_mixhop_conv_save_load_weights(tmp_path, powers):
    """Regression: the weights were in a dict keyed by "0", "1", ..., which
    MLX's ``load_weights`` (``tree_unflatten``) reads back as list indices, so
    a saved MixHopConv could not be loaded (KeyError) and generic parameter
    tree code (e.g. weight decay with ``tree_unflatten``) failed."""
    rng = np.random.default_rng(3)
    edge_index = mx.array(_random_graph(rng))
    x = mx.array(rng.normal(size=(12, 6)).astype(np.float32))
    conv = MixHopConv(6, 8, powers=powers)
    path = str(tmp_path / "mixhop.safetensors")
    conv.save_weights(path)

    loaded = MixHopConv(6, 8, powers=powers)
    loaded.load_weights(path)
    np.testing.assert_allclose(
        np.array(loaded(edge_index, x)),
        np.array(conv(edge_index, x)),
        rtol=1e-5,
        atol=1e-6,
    )
    # parameter trees survive a flatten / unflatten round trip
    loaded.update(tree_unflatten(tree_flatten(conv.parameters())))
