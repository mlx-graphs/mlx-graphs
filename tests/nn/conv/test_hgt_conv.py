import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch
from mlx.utils import tree_flatten
from torch_geometric.nn import HGTConv as PyGHGTConv

from mlx_graphs.nn.conv.hgt_conv import HGTConv

# MLX's GPU matmul is only accurate to ~1e-3 relative on some machines, while a
# wrong formula gives O(1) errors with these O(1) inputs.
TOL = dict(rtol=2e-2, atol=2e-2)

NUM_NODES = {"author": 7, "paper": 9, "venue": 3}
# paper has as many input as output features: its output is gated with its input
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


def _copy_weights(conv, pyg_conv, rng):
    for nt in conv.kqv_lin.lins:
        _copy_linear(conv.kqv_lin.lins[nt], pyg_conv.kqv_lin.lins[nt])
    for nt in conv.out_lin.lins:
        _copy_linear(conv.out_lin.lins[nt], pyg_conv.out_lin.lins[nt])

    H, E = conv.heads, len(EDGE_TYPES)
    D = conv.out_features_dim // H
    # PyG's HeteroLinear type of (head h, edge type e) is h * E + e
    k_rel = pyg_conv.k_rel.weight.detach().numpy().reshape(H, E, D, D)
    v_rel = pyg_conv.v_rel.weight.detach().numpy().reshape(H, E, D, D)
    for e, edge_type in enumerate(EDGE_TYPES):
        key = "__".join(edge_type)
        conv.k_rel[key] = mx.array(k_rel[:, e])
        conv.v_rel[key] = mx.array(v_rel[:, e])

    # random priors and skip gates (PyG initializes them to ones)
    with torch.no_grad():
        for key, param in pyg_conv.p_rel.items():
            param.copy_(torch.from_numpy(rng.uniform(0.5, 1.5, size=param.shape)))
            conv.p_rel[key] = mx.array(param.numpy())
        for nt, param in pyg_conv.skip.items():
            param.copy_(torch.from_numpy(rng.normal(size=param.shape)))
            conv.skip[nt] = mx.array(param.numpy())


@pytest.mark.parametrize("heads", [1, 2])
def test_hgt_conv_shapes(heads):
    rng = np.random.default_rng(0)
    x_dict, edge_index_dict = _hetero_graph(rng)
    conv = HGTConv(DIMS, 8, METADATA, heads=heads)
    out = conv(_to_mlx(x_dict), _to_mlx(edge_index_dict))
    assert set(out) == {"author", "paper"}
    assert out["author"].shape == (7, 8) and out["paper"].shape == (9, 8)

    # a missing edge type: its destination type still gets an output
    del edge_index_dict[("paper", "rev_writes", "author")]
    out = conv(_to_mlx(x_dict), _to_mlx(edge_index_dict))
    assert out["author"].shape == (7, 8)

    conv = HGTConv(4, 8, METADATA, heads=heads)
    assert conv.node_features_dim == {"author": 4, "paper": 4, "venue": 4}
    with pytest.raises(ValueError):
        HGTConv(DIMS, 8, METADATA, heads=3)


@pytest.mark.parametrize("heads", [1, 2, 4])
@pytest.mark.parametrize("drop_edge_type", [False, True])
def test_hgt_conv_matches_pyg(heads, drop_edge_type):
    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    x_dict, edge_index_dict = _hetero_graph(rng)
    if drop_edge_type:
        del edge_index_dict[("author", "writes", "paper")]

    pyg_conv = PyGHGTConv(DIMS, 8, METADATA, heads=heads)
    conv = HGTConv(DIMS, 8, METADATA, heads=heads)
    _copy_weights(conv, pyg_conv, rng)

    with torch.no_grad():
        expected = pyg_conv(_to_torch(x_dict), _to_torch(edge_index_dict))
    out = conv(_to_mlx(x_dict), _to_mlx(edge_index_dict))

    assert set(out) == set(expected)
    for nt in out:
        assert out[nt].shape == tuple(expected[nt].shape)
        np.testing.assert_allclose(np.array(out[nt]), expected[nt].numpy(), **TOL)
    assert np.abs(expected["paper"].numpy()).max() > 0.2


def test_hgt_conv_gradients():
    mx.random.seed(0)
    rng = np.random.default_rng(2)
    x_dict, edge_index_dict = _hetero_graph(rng)
    x_dict, edge_index_dict = _to_mlx(x_dict), _to_mlx(edge_index_dict)
    conv = HGTConv(DIMS, 8, METADATA, heads=2)

    def loss_fn():
        out = conv(x_dict, edge_index_dict)
        return sum((v**2).sum() for v in out.values())

    _, grads = nn.value_and_grad(conv, loss_fn)()
    grads = dict(tree_flatten(grads))
    params = dict(tree_flatten(conv.trainable_parameters()))
    assert set(grads) == set(params)
    for key in (
        "k_rel.author__writes__paper",
        "p_rel.paper__cites__paper",
        "skip.paper",
    ):
        assert mx.any(grads[key] != 0).item(), key
    for name, grad in grads.items():
        assert mx.all(mx.isfinite(grad)).item(), name
