import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch
import torch_geometric.nn as pyg_nn
from mlx.utils import tree_flatten

from mlx_graphs.nn.conv.gin_conv import GINConv
from mlx_graphs.nn.conv.gine_conv import GINEConv

NUM_NODES, DIM, EDGE_DIM = 30, 12, 5
# MLX's GPU matmul is only accurate to ~1e-3 relative, a wrong formula is O(1) off
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
    return edges, x, rng


def to_mx(tensor):
    return mx.array(tensor.detach().numpy())


def make_mlps(out_dim=8):
    torch_mlp = torch.nn.Sequential(
        torch.nn.Linear(DIM, 16), torch.nn.ReLU(), torch.nn.Linear(16, out_dim)
    )
    mlx_mlp = nn.Sequential(nn.Linear(DIM, 16), nn.ReLU(), nn.Linear(16, out_dim))
    return torch_mlp, mlx_mlp


def copy_mlp(mlx_mlp, torch_mlp):
    """PyG resets `nn` in the constructor, so copy after building the layer."""
    for i in (0, 2):
        mlx_mlp.layers[i].weight = to_mx(torch_mlp[i].weight)
        mlx_mlp.layers[i].bias = to_mx(torch_mlp[i].bias)


@pytest.mark.parametrize("edge_features_dim", [None, EDGE_DIM])
def test_gine_conv_shape(edge_features_dim):
    _, mlp = make_mlps(32)
    layer = GINEConv(mlp, edge_features_dim=edge_features_dim)
    edge_index = mx.array([[0, 1, 2, 3, 50], [0, 0, 1, 1, 99]])
    edge_dim = edge_features_dim or DIM

    out = layer(
        edge_index, mx.random.normal((100, DIM)), mx.random.normal((5, edge_dim))
    )

    assert out.shape == (100, 32)
    assert isinstance(layer, GINConv)


@pytest.mark.parametrize("edge_features_dim", [None, EDGE_DIM])
@pytest.mark.parametrize("eps, learn_eps", [(0.0, False), (0.3, True)])
def test_gine_conv_matches_pyg(edge_features_dim, eps, learn_eps):
    edges, x, rng = make_graph()
    edge_dim = edge_features_dim or DIM
    edge_attr = rng.normal(size=(edges.shape[1], edge_dim)).astype(np.float32)

    torch.manual_seed(0)
    torch_mlp, mlx_mlp = make_mlps()
    ref = pyg_nn.GINEConv(
        torch_mlp, eps=eps, train_eps=learn_eps, edge_dim=edge_features_dim
    )
    layer = GINEConv(
        mlx_mlp, eps=eps, learn_eps=learn_eps, edge_features_dim=edge_features_dim
    )
    copy_mlp(mlx_mlp, torch_mlp)
    if edge_features_dim is not None:
        layer.edge_projection.weight = to_mx(ref.lin.weight)
        layer.edge_projection.bias = to_mx(ref.lin.bias)

    expected = ref(torch.tensor(x), torch.tensor(edges), torch.tensor(edge_attr))
    out = layer(mx.array(edges), mx.array(x), mx.array(edge_attr))

    assert_close(out, expected.detach().numpy(), PYG_TOL)


def test_gine_conv_bipartite_matches_pyg():
    rng = np.random.default_rng(0)
    edges = np.array([[0, 0, 1, 2, 3, 4, 5, 6, 7, 7], [0, 0, 0, 1, 1, 2, 2, 3, 3, 0]])
    x_src = rng.normal(size=(9, DIM)).astype(np.float32)
    x_dst = rng.normal(size=(5, DIM)).astype(np.float32)
    edge_attr = rng.normal(size=(edges.shape[1], EDGE_DIM)).astype(np.float32)

    torch.manual_seed(0)
    torch_mlp, mlx_mlp = make_mlps()
    ref = pyg_nn.GINEConv(torch_mlp, edge_dim=EDGE_DIM)
    layer = GINEConv(mlx_mlp, edge_features_dim=EDGE_DIM)
    copy_mlp(mlx_mlp, torch_mlp)
    layer.edge_projection.weight = to_mx(ref.lin.weight)
    layer.edge_projection.bias = to_mx(ref.lin.bias)

    expected = ref(
        (torch.tensor(x_src), torch.tensor(x_dst)),
        torch.tensor(edges),
        torch.tensor(edge_attr),
    )
    out = layer(
        mx.array(edges), (mx.array(x_src), mx.array(x_dst)), mx.array(edge_attr)
    )

    assert out.shape == (5, 8)
    assert_close(out, expected.detach().numpy(), PYG_TOL)


def test_gine_conv_errors():
    _, mlp = make_mlps()
    edge_index = mx.array([[0, 1], [1, 0]])
    x = mx.random.normal((2, DIM))

    with pytest.raises(ValueError, match="edge_features"):
        GINEConv(mlp)(edge_index, x, None)
    with pytest.raises(ValueError, match="do not match"):
        GINEConv(mlp)(edge_index, x, mx.random.normal((2, DIM + 1)))
    with pytest.raises(ValueError, match="node_features_dim"):
        GINEConv(lambda h: h, edge_features_dim=3)

    # An explicit size works when it can't be inferred.
    layer = GINEConv(lambda h: h, edge_features_dim=3, node_features_dim=DIM)
    assert layer(edge_index, x, mx.random.normal((2, 3))).shape == (2, DIM)


def test_gine_conv_edge_projection_initialized_like_pyg():
    # PyG's GINEConv projects edge features with a Linear initialized like
    # torch.nn.Linear: weights and biases uniform in +-1/sqrt(edge_dim).
    mx.random.seed(0)
    layer = GINEConv(lambda h: h, edge_features_dim=4, node_features_dim=64)
    bound = 1 / 4**0.5
    for param in (layer.edge_projection.weight, layer.edge_projection.bias):
        assert mx.abs(param).max().item() <= bound
        assert mx.abs(param).max().item() > 0.9 * bound


def test_gine_conv_gradients():
    edges, x, rng = make_graph()
    edge_attr = mx.array(rng.normal(size=(edges.shape[1], EDGE_DIM)))
    _, mlp = make_mlps()
    layer = GINEConv(mlp, learn_eps=True, edge_features_dim=EDGE_DIM)

    def loss_fn(features):
        return layer(mx.array(edges), features, edge_attr).square().sum()

    loss, grads = nn.value_and_grad(layer, loss_fn)(mx.array(x))
    grads = dict(tree_flatten(grads))

    assert np.isfinite(loss.item())
    assert set(grads) == set(dict(tree_flatten(layer.trainable_parameters())))
    assert {"eps", "edge_projection.weight"} <= set(grads)
    for name, grad in grads.items():
        assert np.isfinite(np.array(grad)).all(), name


def test_gin_conv_edge_features_without_projection():
    """GINConv with edge features of the node size (no projection) is GINE."""
    edges, x, rng = make_graph()
    edge_attr = mx.array(rng.normal(size=(edges.shape[1], DIM)).astype(np.float32))
    _, mlp = make_mlps()

    gin = GINConv(mlp)(mx.array(edges), mx.array(x), edge_attr)
    gine = GINEConv(mlp)(mx.array(edges), mx.array(x), edge_attr)

    np.testing.assert_allclose(np.array(gin), np.array(gine), rtol=1e-5, atol=1e-5)
