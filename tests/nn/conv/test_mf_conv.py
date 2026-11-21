import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch
import torch_geometric.nn as pyg_nn
from mlx.utils import tree_flatten

import mlx_graphs.nn.message_passing as mp
from mlx_graphs.nn import fused_message_passing_mode
from mlx_graphs.nn.conv.mf_conv import MFConv

NUM_NODES, DIM = 30, 12
PYG_TOL = 2e-2
FUSED_TOL = 1e-4


def assert_close(actual, expected, tol):
    actual, expected = np.array(actual), np.array(expected)
    scale = max(1.0, float(np.abs(expected).max()))
    np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol * scale)


def make_graph(seed=0):
    """Random directed graph with duplicate edges, self-loops and isolated nodes."""
    rng = np.random.default_rng(seed)
    edges = rng.integers(0, NUM_NODES - 5, (2, 120))
    edges = edges[:, edges[0] != edges[1]]
    loops = np.array([[0, 1, 2], [0, 1, 2]])
    edges = np.concatenate([edges, edges[:, :10], loops], axis=1)
    x = rng.normal(size=(NUM_NODES, DIM)).astype(np.float32)
    return edges, x


def to_mx(tensor):
    return mx.array(tensor.detach().numpy())


def from_pyg(ref, **kwargs):
    layer = MFConv(ref.in_channels, ref.out_channels, ref.max_degree, **kwargs)
    for mine, theirs in zip(layer.lins_l + layer.lins_r, [*ref.lins_l, *ref.lins_r]):
        mine.weight = to_mx(theirs.weight)
        if theirs.bias is not None:
            mine.bias = to_mx(theirs.bias)
    return layer


@pytest.mark.parametrize("layer_args", [dict(), dict(max_degree=3, bias=False)])
def test_mf_conv_shape(layer_args):
    layer = MFConv(8, 20, **layer_args)
    edge_index = mx.array([[0, 1, 2, 3, 50], [0, 0, 1, 1, 99]])

    out = layer(edge_index, mx.random.normal((100, 8)))

    assert out.shape == (100, 20)


# The test graph has in-degrees from 0 to ~10, so max_degree 3 clamps many.
@pytest.mark.parametrize("max_degree", [3, 10])
@pytest.mark.parametrize("aggr", ["add", "mean"])
@pytest.mark.parametrize("bias", [True, False])
def test_mf_conv_matches_pyg(max_degree, aggr, bias):
    edges, x = make_graph()
    torch.manual_seed(0)
    ref = pyg_nn.MFConv(DIM, 7, max_degree=max_degree, aggr=aggr, bias=bias)
    layer = from_pyg(ref, aggr=aggr, bias=bias)

    expected = ref(torch.tensor(x), torch.tensor(edges))
    out = layer(mx.array(edges), mx.array(x))

    assert_close(out, expected.detach().numpy(), PYG_TOL)


def test_mf_conv_bipartite_matches_pyg():
    rng = np.random.default_rng(0)
    edges = np.array([[0, 0, 1, 2, 3, 4, 5, 6, 7, 7], [0, 0, 0, 1, 1, 2, 2, 3, 3, 0]])
    x_src = rng.normal(size=(9, 16)).astype(np.float32)
    x_dst = rng.normal(size=(5, 8)).astype(np.float32)

    torch.manual_seed(0)
    ref = pyg_nn.MFConv((16, 8), 7, max_degree=2)
    layer = from_pyg(ref)

    expected = ref((torch.tensor(x_src), torch.tensor(x_dst)), torch.tensor(edges))
    out = layer(mx.array(edges), (mx.array(x_src), mx.array(x_dst)))

    assert out.shape == (5, 7)
    assert_close(out, expected.detach().numpy(), PYG_TOL)


@pytest.mark.parametrize("bias", [True, False])
def test_mf_conv_gradients_match_pyg(bias):
    # The per-degree weights are selected with gather_mm: their gradients
    # (and the input's) must match PyG's per-degree projections.
    edges, x = make_graph()
    torch.manual_seed(0)
    ref = pyg_nn.MFConv(DIM, 7, max_degree=3, bias=bias)
    layer = from_pyg(ref, bias=bias)

    x_torch = torch.tensor(x, requires_grad=True)
    ref(x_torch, torch.tensor(edges)).square().sum().backward()

    def loss_fn(features):
        return layer(mx.array(edges), features).square().sum()

    _, grads = nn.value_and_grad(layer, loss_fn)(mx.array(x))
    x_grad = mx.grad(loss_fn)(mx.array(x))
    assert_close(x_grad, x_torch.grad.numpy(), PYG_TOL)
    for name, lins in (("lins_l", ref.lins_l), ("lins_r", ref.lins_r)):
        for i, lin in enumerate(lins):
            assert_close(grads[name][i]["weight"], lin.weight.grad.numpy(), PYG_TOL)
            if lin.bias is not None:
                assert_close(grads[name][i]["bias"], lin.bias.grad.numpy(), PYG_TOL)


def test_mf_conv_gradients():
    edges, x = make_graph()
    layer = MFConv(DIM, 7, max_degree=3)

    def loss_fn(features):
        return layer(mx.array(edges), features).square().sum()

    loss, grads = nn.value_and_grad(layer, loss_fn)(mx.array(x))
    grads = dict(tree_flatten(grads))

    assert np.isfinite(loss.item())
    assert set(grads) == set(dict(tree_flatten(layer.trainable_parameters())))
    assert len(grads) == 3 * (3 + 1)  # lins_l (weight, bias), lins_r (weight)
    for name, grad in grads.items():
        assert np.isfinite(np.array(grad)).all(), name


@pytest.mark.skipif(not mp.HAS_FUSED_KERNELS, reason="mlx_cluster SpMM not installed")
def test_mf_conv_fused_matches_gather_scatter():
    edges, x = make_graph()
    layer = MFConv(DIM, 7, max_degree=4)

    fused = layer(mx.array(edges), mx.array(x))
    with fused_message_passing_mode(False):
        reference = layer(mx.array(edges), mx.array(x))

    assert_close(fused, reference, FUSED_TOL)
