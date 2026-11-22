import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch
from mlx.utils import tree_flatten
from torch_geometric.nn import HANConv as PyGHANConv

from mlx_graphs.nn.conv.han_conv import HANConv

# MLX's GPU matmul is only accurate to ~1e-3 relative on some machines, while a
# wrong formula gives O(1) errors with these O(1) inputs.
TOL = dict(rtol=2e-2, atol=2e-2)

NUM_NODES = {"author": 7, "paper": 9, "venue": 3}
DIMS = {"author": 6, "paper": 8, "venue": 4}
EDGE_TYPES = [
    ("author", "writes", "paper"),
    ("paper", "cites", "paper"),
    ("paper", "rev_writes", "author"),
    ("venue", "hosts", "paper"),  # venue has no incoming edges
]
METADATA = (list(NUM_NODES), EDGE_TYPES)


def _hetero_graph(rng, num_edges=20):
    x_dict = {
        nt: rng.normal(size=(n, DIMS[nt])).astype(np.float32)
        for nt, n in NUM_NODES.items()
    }
    edge_index_dict = {}
    for src, rel, dst in EDGE_TYPES:
        # the last node of each destination type has no incoming edge
        ei = np.stack(
            [
                rng.integers(0, NUM_NODES[src], size=num_edges),
                rng.integers(0, NUM_NODES[dst] - 1, size=num_edges),
            ]
        )
        edge_index_dict[(src, rel, dst)] = ei.astype(np.int64)
    return x_dict, edge_index_dict


def _to_mlx(d):
    return {k: mx.array(v) for k, v in d.items()}


def _to_torch(d):
    return {k: torch.from_numpy(v) for k, v in d.items()}


def _copy_linear(mlx_linear, torch_linear):
    mlx_linear.weight = mx.array(torch_linear.weight.detach().numpy())
    if torch_linear.bias is not None:
        mlx_linear.bias = mx.array(torch_linear.bias.detach().numpy())


def _copy_weights(conv, pyg_conv):
    for nt in conv.proj:
        _copy_linear(conv.proj[nt], pyg_conv.proj[nt])
    for key in conv.lin_src:
        conv.lin_src[key] = mx.array(pyg_conv.lin_src[key].detach().numpy())
        conv.lin_dst[key] = mx.array(pyg_conv.lin_dst[key].detach().numpy())
    _copy_linear(conv.k_lin, pyg_conv.k_lin)
    conv.q = mx.array(pyg_conv.q.detach().numpy())


@pytest.mark.parametrize("heads", [1, 2])
def test_han_conv_shapes(heads):
    rng = np.random.default_rng(0)
    x_dict, edge_index_dict = _hetero_graph(rng)
    conv = HANConv(DIMS, 8, METADATA, heads=heads)
    out, attn = conv(
        _to_mlx(x_dict),
        _to_mlx(edge_index_dict),
        return_semantic_attention_weights=True,
    )
    # venue is the destination of no edge type
    assert set(out) == {"author", "paper"}
    assert out["author"].shape == (7, 8) and out["paper"].shape == (9, 8)
    # three edge types point to paper, one to author
    assert attn["paper"].shape == (3,) and attn["author"].shape == (1,)
    np.testing.assert_allclose(np.array(attn["paper"]).sum(), 1.0, rtol=1e-5)

    # a single input size for all node types
    conv = HANConv(4, 8, METADATA, heads=heads)
    x_dict = {nt: mx.ones((n, 4)) for nt, n in NUM_NODES.items()}
    out = conv(x_dict, _to_mlx(edge_index_dict))
    assert out["paper"].shape == (9, 8)

    with pytest.raises(ValueError):
        HANConv(DIMS, 8, METADATA, heads=3)


@pytest.mark.parametrize("heads", [1, 2, 4])
@pytest.mark.parametrize("negative_slope", [0.2, 0.5])
def test_han_conv_matches_pyg(heads, negative_slope):
    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    x_dict, edge_index_dict = _hetero_graph(rng)

    pyg_conv = PyGHANConv(DIMS, 8, METADATA, heads=heads, negative_slope=negative_slope)
    conv = HANConv(DIMS, 8, METADATA, heads=heads, negative_slope=negative_slope)
    _copy_weights(conv, pyg_conv)

    with torch.no_grad():
        expected, expected_attn = pyg_conv(
            _to_torch(x_dict),
            _to_torch(edge_index_dict),
            return_semantic_attention_weights=True,
        )
    out, attn = conv(
        _to_mlx(x_dict),
        _to_mlx(edge_index_dict),
        return_semantic_attention_weights=True,
    )

    assert expected["venue"] is None and "venue" not in out
    for nt in ("author", "paper"):
        np.testing.assert_allclose(np.array(out[nt]), expected[nt].numpy(), **TOL)
        np.testing.assert_allclose(np.array(attn[nt]), expected_attn[nt].numpy(), **TOL)
    assert np.abs(expected["paper"].numpy()).max() > 0.2


def test_han_conv_gradients():
    mx.random.seed(0)
    rng = np.random.default_rng(2)
    x_dict, edge_index_dict = _hetero_graph(rng)
    x_dict, edge_index_dict = _to_mlx(x_dict), _to_mlx(edge_index_dict)
    conv = HANConv(DIMS, 8, METADATA, heads=2, dropout=0.1)

    def loss_fn():
        out = conv(x_dict, edge_index_dict)
        return sum((v**2).sum() for v in out.values())

    _, grads = nn.value_and_grad(conv, loss_fn)()
    grads = dict(tree_flatten(grads))
    params = dict(tree_flatten(conv.trainable_parameters()))
    assert set(grads) == set(params)
    assert "lin_src.author__writes__paper" in grads
    for name, grad in grads.items():
        assert mx.all(mx.isfinite(grad)).item(), name
