import mlx.core as mx
import pytest

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
