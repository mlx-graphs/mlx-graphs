import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch
from mlx.utils import tree_flatten
from torch_geometric.nn import FAConv as PyGFAConv

from mlx_graphs.nn import fused_message_passing_mode
from mlx_graphs.nn.conv.fa_conv import FAConv

# MLX's GPU matmul is only accurate to ~1e-3 relative on some machines, while a
# wrong formula gives O(1) errors with these O(1) inputs.
TOL = dict(rtol=2e-2, atol=2e-2)


def _random_graph(rng, num_nodes=12, num_edges=40):
    """Random graph without self-loops, with duplicate edges and isolated
    nodes (the last 3 nodes)."""
    src = rng.integers(0, num_nodes - 3, size=num_edges)
    dst = (src + rng.integers(1, num_nodes - 3, size=num_edges)) % (num_nodes - 3)
    src = np.concatenate([src, src[:5]])
    dst = np.concatenate([dst, dst[:5]])
    return np.stack([src, dst]).astype(np.int64)


def _copy_weights(conv, pyg_conv):
    conv.att_l.weight = mx.array(pyg_conv.att_l.weight.detach().numpy())
    conv.att_r.weight = mx.array(pyg_conv.att_r.weight.detach().numpy())


def test_fa_conv_shapes():
    rng = np.random.default_rng(0)
    edge_index = mx.array(_random_graph(rng))
    x = mx.array(rng.normal(size=(12, 6)).astype(np.float32))
    x_0 = mx.array(rng.normal(size=(12, 6)).astype(np.float32))

    conv = FAConv(6)
    out = conv(edge_index, x, x_0)
    assert out.shape == (12, 6)

    out, (ei, alpha) = conv(edge_index, x, x_0, return_attention_weights=True)
    assert out.shape == (12, 6)
    # self-loops are added
    assert ei.shape == (2, edge_index.shape[1] + 12)
    assert alpha.shape == (edge_index.shape[1] + 12,)
    assert mx.all(mx.abs(alpha) <= 1).item()

    with pytest.raises(ValueError):
        conv(edge_index, x, x_0, edge_weights=mx.ones((edge_index.shape[1],)))
    with pytest.raises(ValueError):
        FAConv(6, normalize=False)(edge_index, x, x_0)


@pytest.mark.parametrize("fused", [True, False])
@pytest.mark.parametrize(
    "kwargs",
    [dict(), dict(eps=0.0), dict(eps=0.5, add_self_loops=False)],
)
def test_fa_conv_matches_pyg(fused, kwargs):
    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    edge_index = _random_graph(rng)
    x = rng.normal(size=(12, 6)).astype(np.float32)
    x_0 = rng.normal(size=(12, 6)).astype(np.float32)

    pyg_conv = PyGFAConv(6, **kwargs)
    conv = FAConv(6, **kwargs)
    _copy_weights(conv, pyg_conv)

    with torch.no_grad():
        expected, (pyg_ei, pyg_alpha) = pyg_conv(
            torch.from_numpy(x),
            torch.from_numpy(x_0),
            torch.from_numpy(edge_index),
            return_attention_weights=True,
        )

    with fused_message_passing_mode(fused):
        out, (ei, alpha) = conv(
            mx.array(edge_index),
            mx.array(x),
            mx.array(x_0),
            return_attention_weights=True,
        )

    np.testing.assert_allclose(np.array(out), expected.numpy(), **TOL)
    assert np.abs(expected.numpy()).max() > 0.2
    # Without self-loops in the input, both add one loop per node at the end.
    np.testing.assert_array_equal(np.array(ei), pyg_ei.numpy())
    np.testing.assert_allclose(np.array(alpha), pyg_alpha.numpy(), **TOL)


def test_fa_conv_no_normalize_matches_pyg():
    torch.manual_seed(0)
    rng = np.random.default_rng(2)
    edge_index = _random_graph(rng)
    x = rng.normal(size=(12, 6)).astype(np.float32)
    x_0 = rng.normal(size=(12, 6)).astype(np.float32)
    edge_weight = rng.uniform(0.1, 1.0, size=edge_index.shape[1]).astype(np.float32)

    pyg_conv = PyGFAConv(6, normalize=False)
    conv = FAConv(6, normalize=False)
    _copy_weights(conv, pyg_conv)

    with torch.no_grad():
        expected = pyg_conv(
            torch.from_numpy(x),
            torch.from_numpy(x_0),
            torch.from_numpy(edge_index),
            torch.from_numpy(edge_weight),
        ).numpy()

    out = conv(mx.array(edge_index), mx.array(x), mx.array(x_0), mx.array(edge_weight))
    np.testing.assert_allclose(np.array(out), expected, **TOL)


def test_fa_conv_gradients():
    mx.random.seed(0)
    rng = np.random.default_rng(3)
    edge_index = mx.array(_random_graph(rng))
    x = mx.array(rng.normal(size=(12, 6)).astype(np.float32))
    conv = FAConv(6, dropout=0.1)

    def loss_fn():
        return (conv(edge_index, x, x) ** 2).sum()

    _, grads = nn.value_and_grad(conv, loss_fn)()
    grads = dict(tree_flatten(grads))
    assert set(grads) == {"att_l.weight", "att_r.weight"}
    for name, grad in grads.items():
        assert mx.all(mx.isfinite(grad)).item(), name
        assert mx.any(grad != 0).item(), name
