import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch
import torch_geometric.nn as pyg_nn
from mlx.utils import tree_flatten

from mlx_graphs.nn.conv.nn_conv import ECConv, NNConv

NUM_NODES, DIM, EDGE_DIM, OUT = 30, 12, 5, 7
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


def make_mlps(in_dim=DIM, out_dim=OUT):
    torch_mlp = torch.nn.Sequential(
        torch.nn.Linear(EDGE_DIM, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, in_dim * out_dim),
    )
    mlx_mlp = nn.Sequential(
        nn.Linear(EDGE_DIM, 16), nn.ReLU(), nn.Linear(16, in_dim * out_dim)
    )
    return torch_mlp, mlx_mlp


def copy_weights(layer, ref, mlx_mlp, torch_mlp):
    """PyG resets `nn` in the constructor, so copy after building the layer."""
    for i in (0, 2):
        mlx_mlp.layers[i].weight = to_mx(torch_mlp[i].weight)
        mlx_mlp.layers[i].bias = to_mx(torch_mlp[i].bias)
    if ref.root_weight:
        layer.lin.weight = to_mx(ref.lin.weight)
    if ref.bias is not None:
        torch.nn.init.normal_(ref.bias)  # zeros in PyG, which hides bias bugs
        layer.bias = to_mx(ref.bias)


def test_ec_conv_alias():
    assert ECConv is NNConv


@pytest.mark.parametrize(
    "layer_args", [dict(), dict(aggr="mean"), dict(root_weight=False, bias=False)]
)
def test_nn_conv_shape(layer_args):
    _, mlp = make_mlps(8, 20)
    layer = NNConv(8, 20, mlp, **layer_args)
    edge_index = mx.array([[0, 1, 2, 3, 50], [0, 0, 1, 1, 99]])

    out = layer(edge_index, mx.random.normal((100, 8)), mx.random.normal((5, EDGE_DIM)))

    assert out.shape == (100, 20)


@pytest.mark.parametrize("aggr", ["add", "mean", "max"])
@pytest.mark.parametrize("root_weight", [True, False])
@pytest.mark.parametrize("bias", [True, False])
def test_nn_conv_matches_pyg(aggr, root_weight, bias):
    edges, x, edge_attr = make_graph()
    torch.manual_seed(0)
    torch_mlp, mlx_mlp = make_mlps()
    ref = pyg_nn.NNConv(
        DIM, OUT, torch_mlp, aggr=aggr, root_weight=root_weight, bias=bias
    )
    layer = NNConv(DIM, OUT, mlx_mlp, aggr=aggr, root_weight=root_weight, bias=bias)
    copy_weights(layer, ref, mlx_mlp, torch_mlp)

    expected = ref(torch.tensor(x), torch.tensor(edges), torch.tensor(edge_attr))
    out = layer(mx.array(edges), mx.array(x), mx.array(edge_attr))

    assert_close(out, expected.detach().numpy(), PYG_TOL)


def test_nn_conv_bipartite_matches_pyg():
    rng = np.random.default_rng(0)
    edges = np.array([[0, 0, 1, 2, 3, 4, 5, 6, 7, 7], [0, 0, 0, 1, 1, 2, 2, 3, 3, 0]])
    x_src = rng.normal(size=(9, 16)).astype(np.float32)
    x_dst = rng.normal(size=(5, 8)).astype(np.float32)
    edge_attr = rng.normal(size=(edges.shape[1], EDGE_DIM)).astype(np.float32)

    torch.manual_seed(0)
    torch_mlp, mlx_mlp = make_mlps(16, OUT)
    ref = pyg_nn.NNConv((16, 8), OUT, torch_mlp)
    layer = NNConv((16, 8), OUT, mlx_mlp)
    copy_weights(layer, ref, mlx_mlp, torch_mlp)

    expected = ref(
        (torch.tensor(x_src), torch.tensor(x_dst)),
        torch.tensor(edges),
        torch.tensor(edge_attr),
    )
    out = layer(
        mx.array(edges), (mx.array(x_src), mx.array(x_dst)), mx.array(edge_attr)
    )

    assert out.shape == (5, OUT)
    assert_close(out, expected.detach().numpy(), PYG_TOL)


def test_nn_conv_gradients():
    edges, x, edge_attr = make_graph()
    _, mlp = make_mlps()
    layer = NNConv(DIM, OUT, mlp)

    def loss_fn(features):
        out = layer(mx.array(edges), features, mx.array(edge_attr))
        return out.square().sum()

    loss, grads = nn.value_and_grad(layer, loss_fn)(mx.array(x))
    grads = dict(tree_flatten(grads))

    assert np.isfinite(loss.item())
    assert set(grads) == set(dict(tree_flatten(layer.trainable_parameters())))
    assert {"lin.weight", "bias", "mlp.layers.0.weight"} <= set(grads)
    for name, grad in grads.items():
        grad = np.array(grad)
        assert np.isfinite(grad).all(), name
        assert np.abs(grad).max() > 0, name
