import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_graphs.nn.conv.sage_conv import SAGEConv


@pytest.mark.parametrize(
    "layer, edge_index, node_features, edge_weights, expected",
    [
        (
            SAGEConv(8, 20),
            mx.array([[0, 1, 2, 3], [0, 0, 1, 1]]),
            mx.random.uniform(0, 1, [6, 8]),
            None,
            (6, 20),
        ),
        (
            SAGEConv(16, 32),
            mx.array([[0, 1, 2, 3, 50], [0, 0, 1, 1, 99]]),
            mx.random.uniform(0, 1, [100, 16]),
            None,
            (100, 32),
        ),
        (
            SAGEConv(16, 32, bias=False),
            mx.array([[0, 1, 2, 3, 50], [0, 0, 1, 1, 99]]),
            mx.random.uniform(0, 1, [100, 16]),
            None,
            (100, 32),
        ),
        (
            SAGEConv(16, 32, bias=False),
            mx.array([[0, 1, 2, 3, 50], [0, 0, 1, 1, 99]]),
            mx.random.uniform(0, 1, [100, 16]),
            mx.random.normal(
                [
                    5,
                ]
            ),
            (100, 32),
        ),
    ],
)
def test_sage_conv(layer, edge_index, node_features, edge_weights, expected):
    assert (
        expected == layer(edge_index, node_features, edge_weights=edge_weights).shape
    ), "SAGEConv failed"


# ---------------------------------------------------------------------------
# Bipartite SAGEConv (e.g. one relation of a heterogeneous graph)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "in_dims, out_dim",
    [
        ((16, 8), 32),  # src dim < out: aggregate, then project
        ((64, 8), 4),  # src dim > out: project, then aggregate
        ((8, 8), 8),
    ],
)
def test_sage_conv_bipartite_shape(in_dims, out_dim):
    src_dim, dst_dim = in_dims
    conv = SAGEConv(in_dims, out_dim)
    edge_index = mx.array([[0, 1, 4, 2], [0, 0, 1, 2]])  # 5 src -> 3 dst
    x_src = mx.random.normal((5, src_dim))
    x_dst = mx.random.normal((3, dst_dim))

    out = conv(edge_index, (x_src, x_dst))

    assert out.shape == (3, out_dim)


def test_sage_conv_bipartite_matches_formula():
    """h_dst = W_self x_dst + W_neigh * mean_{src in N(dst)} x_src"""
    conv = SAGEConv((3, 2), 4)
    x_src = mx.random.normal((5, 3))
    x_dst = mx.random.normal((2, 2))
    edge_index = mx.array([[0, 1, 4], [0, 0, 1]])

    out = conv(edge_index, (x_src, x_dst))

    mean_neigh = mx.stack([(x_src[0] + x_src[1]) / 2, x_src[4]])
    expected = conv.self_proj(x_dst) + conv.neigh_proj(mean_neigh)
    assert mx.allclose(out, expected, atol=1e-5)


def test_sage_conv_dst_without_neighbors_gets_only_self_term():
    conv = SAGEConv((3, 2), 4)
    x_src = mx.random.normal((2, 3))
    x_dst = mx.random.normal((3, 2))
    edge_index = mx.array([[0, 1], [0, 0]])  # dst 1 and 2 have no neighbors

    out = conv(edge_index, (x_src, x_dst))

    assert mx.allclose(out[1:], conv.self_proj(x_dst[1:]), atol=1e-5)


@pytest.mark.parametrize("in_dim, out_dim", [(16, 32), (32, 8)])
def test_sage_conv_tuple_of_same_features_matches_single_array(in_dim, out_dim):
    conv = SAGEConv(in_dim, out_dim)
    x = mx.random.normal((6, in_dim))
    edge_index = mx.array([[0, 1, 2, 3, 5], [1, 1, 2, 4, 0]])

    assert mx.allclose(conv(edge_index, x), conv(edge_index, (x, x)), atol=1e-5)


@pytest.mark.parametrize("aggr", ["mean", "max", "min", "std", ["mean", "max"]])
@pytest.mark.parametrize("in_dim, out_dim", [(16, 4), (4, 16)])
def test_sage_conv_aggregations_match_pyg(aggr, in_dim, out_dim):
    """Regression: with more input than output features, the projection was
    applied before aggregating, which is only right for sum / mean (max(Wx) is
    not W max(x)); and a list of aggregations failed (the neighbor projection
    did not take the concatenated size)."""
    torch = pytest.importorskip("torch")
    pyg_nn = pytest.importorskip("torch_geometric.nn")

    rng = np.random.default_rng(0)
    edge_index = rng.integers(0, 10, size=(2, 40))
    x = rng.normal(size=(10, in_dim)).astype(np.float32)
    torch.manual_seed(0)
    pyg_conv = pyg_nn.SAGEConv(in_dim, out_dim, aggr=aggr)
    conv = SAGEConv(in_dim, out_dim, aggr=aggr)
    # PyG puts the bias on the neighbor projection, mlx-graphs on the root one
    conv.neigh_proj.weight = mx.array(pyg_conv.lin_l.weight.detach().numpy())
    conv.self_proj.weight = mx.array(pyg_conv.lin_r.weight.detach().numpy())
    conv.self_proj.bias = mx.array(pyg_conv.lin_l.bias.detach().numpy())

    with torch.no_grad():
        expected = pyg_conv(torch.from_numpy(x), torch.from_numpy(edge_index)).numpy()
    out = conv(mx.array(edge_index), mx.array(x))
    np.testing.assert_allclose(np.array(out), expected, rtol=1e-2, atol=1e-2)


def test_sage_conv_lstm_aggregation_matches_pyg():
    # As in PyG, aggr="lstm" builds an LSTM aggregation of the input size.
    torch = pytest.importorskip("torch")
    pyg_nn = pytest.importorskip("torch_geometric.nn")
    from tests.pyg_copy import load_torch, lstm_state

    rng = np.random.default_rng(0)
    edge_index = rng.integers(0, 10, size=(2, 40))
    edge_index = edge_index[:, np.argsort(edge_index[1], kind="stable")]
    x = rng.normal(size=(10, 6)).astype(np.float32)
    torch.manual_seed(0)
    pyg_conv = pyg_nn.SAGEConv(6, 4, aggr="lstm")
    conv = SAGEConv(6, 4, aggr="lstm")
    state = lstm_state("aggr_module.lstm", pyg_conv.aggr_module.lstm)
    state["neigh_proj.weight"] = pyg_conv.lin_l.weight.detach().numpy()
    state["self_proj.weight"] = pyg_conv.lin_r.weight.detach().numpy()
    state["self_proj.bias"] = pyg_conv.lin_l.bias.detach().numpy()
    load_torch(conv, state)

    with torch.no_grad():
        expected = pyg_conv(torch.from_numpy(x), torch.from_numpy(edge_index)).numpy()
    out = conv(mx.array(edge_index), mx.array(x))
    np.testing.assert_allclose(np.array(out), expected, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("aggr", ["max", "min"])
def test_non_linear_aggregation_aggregates_before_projecting(aggr):
    """max/min do not commute with the neighbor projection: aggregate first."""
    from mlx_graphs.utils import scatter

    conv = SAGEConv(8, 3, aggr=aggr)  # 8 > 3 would project first for "mean"
    x = mx.random.normal((5, 8))
    edge_index = mx.array([[0, 1, 2, 3, 4], [1, 1, 2, 2, 0]])
    aggregated = scatter(x[edge_index[0]], edge_index[1], 5, aggr)
    expected = conv.neigh_proj(aggregated) + conv.self_proj(x)
    assert mx.allclose(conv(edge_index, x), expected, atol=1e-4)


@pytest.mark.parametrize("bias", [True, False])
def test_sage_conv_without_root_weight_matches_pyg(bias):
    torch = pytest.importorskip("torch")
    pyg_nn = pytest.importorskip("torch_geometric.nn")

    rng = np.random.default_rng(0)
    edge_index = rng.integers(0, 10, size=(2, 40))
    x = rng.normal(size=(10, 6)).astype(np.float32)
    torch.manual_seed(0)
    pyg_conv = pyg_nn.SAGEConv(6, 4, root_weight=False, bias=bias)
    conv = SAGEConv(6, 4, root_weight=False, bias=bias)
    assert "self_proj" not in conv and ("bias" in conv) == bias
    conv.neigh_proj.weight = mx.array(pyg_conv.lin_l.weight.detach().numpy())
    if bias:
        conv.bias = mx.array(pyg_conv.lin_l.bias.detach().numpy())

    with torch.no_grad():
        expected = pyg_conv(torch.from_numpy(x), torch.from_numpy(edge_index)).numpy()
    out = conv(mx.array(edge_index), mx.array(x))
    np.testing.assert_allclose(np.array(out), expected, rtol=1e-2, atol=1e-2)


def test_sage_conv_root_weight_turned_off_after_construction_keeps_the_bias():
    conv = SAGEConv(6, 4)
    bias = conv.self_proj.bias
    conv.root_weight = False
    assert "self_proj" not in conv and conv.bias is bias
    params = dict(tree_flatten(conv.parameters()))
    assert set(params) == {"neigh_proj.weight", "bias"}
