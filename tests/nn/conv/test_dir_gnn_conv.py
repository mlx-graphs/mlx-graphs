import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch
from mlx.utils import tree_flatten
from torch_geometric.nn import DirGNNConv as PyGDirGNNConv
from torch_geometric.nn import GCNConv as PyGGCNConv

from mlx_graphs.nn import GCNConv, SAGEConv, TransformerConv
from mlx_graphs.nn.conv.dir_gnn_conv import DirGNNConv

# MLX's GPU matmul is only accurate to ~1e-3 relative on some machines, while a
# wrong formula gives O(1) errors with these O(1) inputs.
TOL = dict(rtol=2e-2, atol=2e-2)


def _random_graph(rng, num_nodes=12, num_edges=40):
    """Random directed graph without self-loops, with isolated nodes."""
    src = rng.integers(0, num_nodes - 3, size=num_edges)
    dst = (src + rng.integers(1, num_nodes - 3, size=num_edges)) % (num_nodes - 3)
    return np.stack([src, dst]).astype(np.int64)


def _copy_gcn(conv, pyg_conv):
    conv.linear.weight = mx.array(pyg_conv.lin.weight.detach().numpy())
    conv.bias = mx.array(pyg_conv.bias.detach().numpy())


def test_dir_gnn_conv_shapes():
    rng = np.random.default_rng(0)
    edge_index = mx.array(_random_graph(rng))
    x = mx.array(rng.normal(size=(12, 6)).astype(np.float32))

    conv = DirGNNConv(SAGEConv(6, 8))
    assert conv(edge_index, x).shape == (12, 8)

    # extra per-edge arguments are passed to both layers
    conv = DirGNNConv(GCNConv(6, 8), root_weight=False)
    weights = mx.ones((edge_index.shape[1],))
    assert conv(edge_index, x, edge_weights=weights).shape == (12, 8)
    assert "lin" not in conv


def test_dir_gnn_conv_copies():
    conv = DirGNNConv(SAGEConv(6, 8))
    # deep copies, with the same initial weights
    assert conv.conv_in.neigh_proj.weight is not conv.conv_out.neigh_proj.weight
    assert mx.array_equal(
        conv.conv_in.neigh_proj.weight, conv.conv_out.neigh_proj.weight
    )
    params = dict(tree_flatten(conv.trainable_parameters()))
    assert "lin.weight" in params and "conv_in.neigh_proj.weight" in params

    # a factory initializes both layers independently
    conv = DirGNNConv(lambda: SAGEConv(6, 8))
    assert not mx.array_equal(
        conv.conv_in.neigh_proj.weight, conv.conv_out.neigh_proj.weight
    )

    # self-loops and root weights of the wrapped layers are turned off
    conv = DirGNNConv(
        GCNConv(6, 8, add_self_loops=True), node_features_dim=6, out_features_dim=8
    )
    assert not conv.conv_in._add_self_loops and not conv.conv_out._add_self_loops
    conv = DirGNNConv(TransformerConv(6, 8))
    assert not conv.conv_in.root_weight and not conv.conv_out.root_weight

    # Regression: GCNConv did not store its sizes, so DirGNNConv(GCNConv(...))
    # (PyG's usual way to build it) raised unless the sizes were passed again.
    conv = DirGNNConv(GCNConv(6, 8))
    assert conv.lin.weight.shape == (8, 6)
    conv = DirGNNConv(lambda: GCNConv(6, 8))
    assert (conv.node_features_dim, conv.out_features_dim) == (6, 8)

    # a layer without stored sizes still needs them
    class NoSizes(nn.Module):
        def __call__(self, edge_index, x):
            return x

    with pytest.raises(ValueError):
        DirGNNConv(NoSizes())
    assert DirGNNConv(NoSizes(), node_features_dim=6, out_features_dim=6).lin


@pytest.mark.parametrize("alpha", [0.5, 0.2])
@pytest.mark.parametrize("root_weight", [True, False])
def test_dir_gnn_conv_matches_pyg(alpha, root_weight):
    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    edge_index = _random_graph(rng)
    x = rng.normal(size=(12, 6)).astype(np.float32)

    pyg_conv = PyGDirGNNConv(
        PyGGCNConv(6, 8, add_self_loops=True), alpha=alpha, root_weight=root_weight
    )
    with torch.no_grad():  # PyG initializes the GCN biases to zero
        for bias in (pyg_conv.conv_in.bias, pyg_conv.conv_out.bias):
            bias.copy_(torch.randn_like(bias))
    conv = DirGNNConv(
        GCNConv(6, 8, add_self_loops=True),
        alpha=alpha,
        root_weight=root_weight,
        node_features_dim=6,
        out_features_dim=8,
    )
    _copy_gcn(conv.conv_in, pyg_conv.conv_in)
    _copy_gcn(conv.conv_out, pyg_conv.conv_out)
    if root_weight:
        conv.lin.weight = mx.array(pyg_conv.lin.weight.detach().numpy())
        conv.lin.bias = mx.array(pyg_conv.lin.bias.detach().numpy())

    with torch.no_grad():
        expected = pyg_conv(torch.from_numpy(x), torch.from_numpy(edge_index)).numpy()
    out = conv(mx.array(edge_index), mx.array(x))

    assert out.shape == expected.shape
    np.testing.assert_allclose(np.array(out), expected, **TOL)
    assert np.abs(expected).max() > 0.2


def test_dir_gnn_conv_gradients():
    mx.random.seed(0)
    rng = np.random.default_rng(2)
    edge_index = mx.array(_random_graph(rng))
    x = mx.array(rng.normal(size=(12, 6)).astype(np.float32))
    conv = DirGNNConv(lambda: SAGEConv(6, 8))

    def loss_fn():
        return (conv(edge_index, x) ** 2).sum()

    _, grads = nn.value_and_grad(conv, loss_fn)()
    grads = dict(tree_flatten(grads))
    for name, grad in grads.items():
        assert mx.all(mx.isfinite(grad)).item(), name
        assert mx.any(grad != 0).item(), name


def _copy_sage(conv, pyg_conv):
    """PyG's SAGEConv without root weight keeps its bias on ``lin_l``."""
    conv.neigh_proj.weight = mx.array(pyg_conv.lin_l.weight.detach().numpy())
    conv.bias = mx.array(pyg_conv.lin_l.bias.detach().numpy())


def test_dir_gnn_conv_sage_drops_the_root_terms_like_pyg():
    """Regression: SAGEConv had no ``root_weight``, so both copies kept their
    own root projection (three root terms in total instead of PyG's one)."""
    from torch_geometric.nn import SAGEConv as PyGSAGEConv

    rng = np.random.default_rng(0)
    edge_index = _random_graph(rng)
    x = rng.normal(size=(12, 6)).astype(np.float32)

    torch.manual_seed(0)
    pyg = PyGDirGNNConv(PyGSAGEConv(6, 8), alpha=0.3)
    conv = DirGNNConv(lambda: SAGEConv(6, 8), alpha=0.3)
    assert "self_proj" not in conv.conv_in and "self_proj" not in conv.conv_out
    _copy_sage(conv.conv_in, pyg.conv_in)
    _copy_sage(conv.conv_out, pyg.conv_out)
    conv.lin.weight = mx.array(pyg.lin.weight.detach().numpy())
    conv.lin.bias = mx.array(pyg.lin.bias.detach().numpy())

    x_t = torch.from_numpy(x).requires_grad_()
    expected = pyg(x_t, torch.from_numpy(edge_index))
    expected.pow(2).sum().backward()

    def loss(x):
        return (conv(mx.array(edge_index), x) ** 2).sum()

    out = conv(mx.array(edge_index), mx.array(x))
    grad = mx.grad(loss)(mx.array(x))
    np.testing.assert_allclose(np.array(out), expected.detach().numpy(), **TOL)
    np.testing.assert_allclose(np.array(grad), x_t.grad.numpy(), **TOL)
