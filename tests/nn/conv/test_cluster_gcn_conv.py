import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch
import torch_geometric.nn as pyg_nn
from mlx.utils import tree_flatten

import mlx_graphs.nn.message_passing as mp
from mlx_graphs.nn import fused_message_passing_mode
from mlx_graphs.nn.conv.cluster_gcn_conv import ClusterGCNConv

NUM_NODES, DIM = 30, 12
PYG_TOL = 2e-2
FUSED_TOL = 1e-4


def assert_close(actual, expected, tol):
    actual, expected = np.array(actual), np.array(expected)
    scale = max(1.0, float(np.abs(expected).max()))
    np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol * scale)


def make_graph(seed=0, all_loops=False):
    """Random directed graph with duplicate edges, self-loops and isolated nodes.
    With ``all_loops``, every node has exactly one self-loop, in node order."""
    rng = np.random.default_rng(seed)
    edges = rng.integers(0, NUM_NODES - 5, (2, 120))
    edges = edges[:, edges[0] != edges[1]]
    loops = np.arange(NUM_NODES if all_loops else 3)
    edges = np.concatenate([edges, edges[:, :10], np.stack([loops, loops])], axis=1)
    x = rng.normal(size=(NUM_NODES, DIM)).astype(np.float32)
    return edges, x


def to_mx(tensor):
    return mx.array(tensor.detach().numpy())


def from_pyg(ref, **kwargs):
    layer = ClusterGCNConv(ref.in_channels, ref.out_channels, **kwargs)
    layer.lin_out.weight = to_mx(ref.lin_out.weight)
    if ref.lin_out.bias is not None:
        layer.lin_out.bias = to_mx(ref.lin_out.bias)
    layer.lin_root.weight = to_mx(ref.lin_root.weight)
    return layer


@pytest.mark.parametrize(
    "layer_args",
    [dict(), dict(diag_lambda=1.0, bias=False), dict(add_self_loops=False)],
)
def test_cluster_gcn_conv_shape(layer_args):
    layer = ClusterGCNConv(8, 20, **layer_args)
    edge_index = mx.array([[0, 1, 2, 3, 50], [0, 0, 1, 1, 99]])

    out = layer(edge_index, mx.random.normal((100, 8)))

    assert out.shape == (100, 20)


# out_dim 6 projects before propagating, out_dim 20 after
@pytest.mark.parametrize("out_dim", [6, 20])
@pytest.mark.parametrize("diag_lambda", [0.0, 0.7])
@pytest.mark.parametrize("add_self_loops", [True, False])
@pytest.mark.parametrize("bias", [True, False])
def test_cluster_gcn_conv_matches_pyg(out_dim, diag_lambda, add_self_loops, bias):
    # PyG's add_self_loops=False path needs exactly one self-loop per node.
    edges, x = make_graph(all_loops=not add_self_loops)
    torch.manual_seed(0)
    args = dict(diag_lambda=diag_lambda, add_self_loops=add_self_loops, bias=bias)
    ref = pyg_nn.ClusterGCNConv(DIM, out_dim, **args)
    if bias:
        torch.nn.init.normal_(ref.lin_out.bias)
    layer = from_pyg(ref, **args)

    expected = ref(torch.tensor(x), torch.tensor(edges))
    out = layer(mx.array(edges), mx.array(x))

    assert_close(out, expected.detach().numpy(), PYG_TOL)


def test_cluster_gcn_conv_gradients():
    edges, x = make_graph()
    layer = ClusterGCNConv(DIM, 7, diag_lambda=0.5)

    def loss_fn(features):
        return layer(mx.array(edges), features).square().sum()

    loss, grads = nn.value_and_grad(layer, loss_fn)(mx.array(x))
    grads = dict(tree_flatten(grads))

    assert np.isfinite(loss.item())
    assert set(grads) == {"lin_out.weight", "lin_out.bias", "lin_root.weight"}
    for name, grad in grads.items():
        grad = np.array(grad)
        assert np.isfinite(grad).all(), name
        assert np.abs(grad).max() > 0, name


@pytest.mark.skipif(not mp.HAS_FUSED_KERNELS, reason="mlx_cluster SpMM not installed")
@pytest.mark.parametrize("out_dim", [6, 20])
def test_cluster_gcn_conv_fused_matches_gather_scatter(out_dim):
    edges, x = make_graph()
    layer = ClusterGCNConv(DIM, out_dim, diag_lambda=0.5)

    fused = layer(mx.array(edges), mx.array(x))
    with fused_message_passing_mode(False):
        reference = layer(mx.array(edges), mx.array(x))

    assert_close(fused, reference, FUSED_TOL)
