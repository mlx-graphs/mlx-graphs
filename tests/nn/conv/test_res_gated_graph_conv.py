import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch
import torch_geometric.nn as pyg_nn
from mlx.utils import tree_flatten

from mlx_graphs.nn.conv.res_gated_graph_conv import ResGatedGraphConv

NUM_NODES, DIM, EDGE_DIM = 30, 12, 5
PYG_TOL = 2e-2


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
    edge_attr = rng.normal(size=(edges.shape[1], EDGE_DIM)).astype(np.float32)
    return edges, x, edge_attr


def to_mx(tensor):
    return mx.array(tensor.detach().numpy())


def copy_linear(dst, src):
    dst.weight = to_mx(src.weight)
    if src.bias is not None:
        dst.bias = to_mx(src.bias)


def from_pyg(ref, **kwargs):
    layer = ResGatedGraphConv(ref.in_channels, ref.out_channels, **kwargs)
    for name in ("lin_key", "lin_query", "lin_value"):
        copy_linear(layer[name], getattr(ref, name))
    if ref.lin_skip is not None:
        copy_linear(layer.lin_skip, ref.lin_skip)
    if ref.bias is not None:
        torch.nn.init.normal_(ref.bias)  # zeros in PyG, which hides bias bugs
        layer.bias = to_mx(ref.bias)
    return layer


@pytest.mark.parametrize(
    "layer_args",
    [dict(), dict(root_weight=False, bias=False), dict(edge_features_dim=EDGE_DIM)],
)
def test_res_gated_graph_conv_shape(layer_args):
    layer = ResGatedGraphConv(8, 20, **layer_args)
    edge_index = mx.array([[0, 1, 2, 3, 50], [0, 0, 1, 1, 99]])
    edge_features = (
        mx.random.normal((5, EDGE_DIM)) if "edge_features_dim" in layer_args else None
    )

    out = layer(edge_index, mx.random.normal((100, 8)), edge_features)

    assert out.shape == (100, 20)


@pytest.mark.parametrize("edge_dim", [None, EDGE_DIM])
@pytest.mark.parametrize("root_weight", [True, False])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("aggr", ["add", "mean"])
def test_res_gated_graph_conv_matches_pyg(edge_dim, root_weight, bias, aggr):
    edges, x, edge_attr = make_graph()
    args = dict(edge_dim=edge_dim, root_weight=root_weight, bias=bias, aggr=aggr)
    torch.manual_seed(0)
    ref = pyg_nn.ResGatedGraphConv(DIM, 7, **args)
    layer = from_pyg(
        ref, edge_features_dim=edge_dim, root_weight=root_weight, bias=bias, aggr=aggr
    )
    with_edges = edge_dim is not None

    expected = ref(
        torch.tensor(x),
        torch.tensor(edges),
        torch.tensor(edge_attr) if with_edges else None,
    )
    out = layer(
        mx.array(edges), mx.array(x), mx.array(edge_attr) if with_edges else None
    )

    assert_close(out, expected.detach().numpy(), PYG_TOL)


@pytest.mark.parametrize("edge_dim", [None, EDGE_DIM])
def test_res_gated_graph_conv_bipartite_matches_pyg(edge_dim):
    rng = np.random.default_rng(0)
    edges = np.array([[0, 0, 1, 2, 3, 4, 5, 6, 7, 7], [0, 0, 0, 1, 1, 2, 2, 3, 3, 0]])
    x_src = rng.normal(size=(9, 16)).astype(np.float32)
    x_dst = rng.normal(size=(5, 8)).astype(np.float32)
    edge_attr = rng.normal(size=(edges.shape[1], EDGE_DIM)).astype(np.float32)
    with_edges = edge_dim is not None

    torch.manual_seed(0)
    ref = pyg_nn.ResGatedGraphConv((16, 8), 7, edge_dim=edge_dim)
    layer = from_pyg(ref, edge_features_dim=edge_dim)

    expected = ref(
        (torch.tensor(x_src), torch.tensor(x_dst)),
        torch.tensor(edges),
        torch.tensor(edge_attr) if with_edges else None,
    )
    out = layer(
        mx.array(edges),
        (mx.array(x_src), mx.array(x_dst)),
        mx.array(edge_attr) if with_edges else None,
    )

    assert out.shape == (5, 7)
    assert_close(out, expected.detach().numpy(), PYG_TOL)


def test_res_gated_graph_conv_edge_features_mismatch():
    edge_index = mx.array([[0, 1], [1, 0]])
    x = mx.random.normal((2, DIM))
    with pytest.raises(ValueError):
        ResGatedGraphConv(DIM, 4)(edge_index, x, mx.zeros((2, EDGE_DIM)))
    with pytest.raises(ValueError):
        ResGatedGraphConv(DIM, 4, edge_features_dim=EDGE_DIM)(edge_index, x)


@pytest.mark.parametrize("edge_dim", [None, EDGE_DIM])
def test_res_gated_graph_conv_gradients(edge_dim):
    edges, x, edge_attr = make_graph()
    layer = ResGatedGraphConv(DIM, 7, edge_features_dim=edge_dim)
    e = mx.array(edge_attr) if edge_dim else None

    def loss_fn(features):
        return layer(mx.array(edges), features, e).square().sum()

    loss, grads = nn.value_and_grad(layer, loss_fn)(mx.array(x))
    grads = dict(tree_flatten(grads))

    assert np.isfinite(loss.item())
    assert set(grads) == set(dict(tree_flatten(layer.trainable_parameters())))
    for name, grad in grads.items():
        grad = np.array(grad)
        assert np.isfinite(grad).all(), name
        assert np.abs(grad).max() > 0, name
