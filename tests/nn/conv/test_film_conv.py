import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch
import torch_geometric.nn as pyg_nn
from mlx.utils import tree_flatten

from mlx_graphs.nn.conv.film_conv import FiLMConv

NUM_NODES, DIM, OUT = 30, 12, 7
PYG_TOL = 2e-2


def assert_close(actual, expected, tol):
    actual, expected = np.array(actual), np.array(expected)
    scale = max(1.0, float(np.abs(expected).max()))
    np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol * scale)


def make_graph(seed=0, num_relations=3):
    """Random directed graph with duplicate edges, self-loops and isolated nodes."""
    rng = np.random.default_rng(seed)
    edges = rng.integers(0, NUM_NODES - 5, (2, 120))
    edges = edges[:, edges[0] != edges[1]]
    loops = np.array([[0, 1, 2], [0, 1, 2]])
    edges = np.concatenate([edges, edges[:, :10], loops], axis=1)
    x = rng.normal(size=(NUM_NODES, DIM)).astype(np.float32)
    edge_type = rng.integers(0, num_relations, edges.shape[1])
    return edges, x, edge_type


def to_mx(tensor):
    return mx.array(tensor.detach().numpy())


def copy_linear(dst, src):
    dst.weight = to_mx(src.weight)
    if src.bias is not None:
        dst.bias = to_mx(src.bias)


def copy_module(dst, src):
    if not isinstance(src, torch.nn.Sequential):  # PyG Linear
        copy_linear(dst, src)
    else:  # Sequential(Linear, ReLU, Linear)
        copy_linear(dst.layers[0], src[0])
        copy_linear(dst.layers[2], src[2])


def copy_from_pyg(layer, ref):
    for mine, theirs in zip(layer.lins, ref.lins):
        copy_linear(mine, theirs)
    for mine, theirs in zip(layer.films, ref.films):
        copy_module(mine, theirs)
    copy_linear(layer.lin_skip, ref.lin_skip)
    copy_module(layer.film_skip, ref.film_skip)


@pytest.mark.parametrize("num_relations", [1, 3])
@pytest.mark.parametrize("act", [nn.relu, None])
def test_film_conv_shape(num_relations, act):
    layer = FiLMConv(8, 20, num_relations=num_relations, act=act)
    edge_index = mx.array([[0, 1, 2, 3, 50], [0, 0, 1, 1, 99]])
    edge_type = mx.array([0, 1, 2, 1, 0]) % num_relations

    out = layer(edge_index, mx.random.normal((100, 8)), edge_type)

    assert out.shape == (100, 20)


def test_film_conv_requires_edge_type():
    with pytest.raises(ValueError, match="edge_type"):
        FiLMConv(DIM, OUT, num_relations=2)(mx.array([[0], [1]]), mx.zeros((2, DIM)))


@pytest.mark.parametrize("num_relations", [1, 3])
@pytest.mark.parametrize("aggr", ["mean", "add", "max"])
@pytest.mark.parametrize("use_act", [True, False])
@pytest.mark.parametrize("custom_mlp", [False, True])
def test_film_conv_matches_pyg(num_relations, aggr, use_act, custom_mlp):
    edges, x, edge_type = make_graph(num_relations=num_relations)
    torch.manual_seed(0)
    torch_mlp, mlx_mlp = None, None
    if custom_mlp:
        torch_mlp = torch.nn.Sequential(
            torch.nn.Linear(DIM, 16), torch.nn.ReLU(), torch.nn.Linear(16, 2 * OUT)
        )
        mlx_mlp = nn.Sequential(nn.Linear(DIM, 16), nn.ReLU(), nn.Linear(16, 2 * OUT))
    ref = pyg_nn.FiLMConv(
        DIM,
        OUT,
        num_relations=num_relations,
        nn=torch_mlp,
        act=torch.nn.ReLU() if use_act else None,
        aggr=aggr,
    )
    layer = FiLMConv(
        DIM,
        OUT,
        num_relations=num_relations,
        mlp=mlx_mlp,
        act=nn.relu if use_act else None,
        aggr=aggr,
    )
    copy_from_pyg(layer, ref)

    expected = ref(torch.tensor(x), torch.tensor(edges), torch.tensor(edge_type))
    out = layer(mx.array(edges), mx.array(x), mx.array(edge_type))

    assert_close(out, expected.detach().numpy(), PYG_TOL)


@pytest.mark.parametrize("num_relations", [1, 2])
def test_film_conv_bipartite_matches_pyg(num_relations):
    rng = np.random.default_rng(0)
    edges = np.array([[0, 0, 1, 2, 3, 4, 5, 6, 7, 7], [0, 0, 0, 1, 1, 2, 2, 3, 3, 0]])
    edge_type = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0]) % num_relations
    x_src = rng.normal(size=(9, 16)).astype(np.float32)
    x_dst = rng.normal(size=(5, 8)).astype(np.float32)

    torch.manual_seed(0)
    ref = pyg_nn.FiLMConv((16, 8), OUT, num_relations=num_relations)
    layer = FiLMConv((16, 8), OUT, num_relations=num_relations)
    copy_from_pyg(layer, ref)

    expected = ref(
        (torch.tensor(x_src), torch.tensor(x_dst)),
        torch.tensor(edges),
        torch.tensor(edge_type),
    )
    out = layer(
        mx.array(edges), (mx.array(x_src), mx.array(x_dst)), mx.array(edge_type)
    )

    assert out.shape == (5, OUT)
    assert_close(out, expected.detach().numpy(), PYG_TOL)


def test_film_conv_gradients():
    edges, x, edge_type = make_graph(num_relations=2)
    layer = FiLMConv(DIM, OUT, num_relations=2)

    def loss_fn(features):
        out = layer(mx.array(edges), features, mx.array(edge_type))
        return out.square().sum()

    loss, grads = nn.value_and_grad(layer, loss_fn)(mx.array(x))
    grads = dict(tree_flatten(grads))

    assert np.isfinite(loss.item())
    assert set(grads) == set(dict(tree_flatten(layer.trainable_parameters())))
    assert {"lins.1.weight", "films.1.weight", "film_skip.weight"} <= set(grads)
    for name, grad in grads.items():
        grad = np.array(grad)
        assert np.isfinite(grad).all(), name
        assert np.abs(grad).max() > 0, name
