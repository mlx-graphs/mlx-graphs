import mlx.core as mx

from mlx_graphs.nn.conv.simple_conv import SimpleConv


def test_simple_conv():
    # Aggr: add, combine_root_func=None, with edge weights.
    conv = SimpleConv(aggr="add", combine_root_func=None)
    node_features = mx.ones((5, 3))
    edge_index = mx.array([[0, 1, 2, 3, 4], [0, 0, 1, 1, 3]])
    edge_weights = mx.array([10, 20, 5, 2, 15])
    y_hat1 = conv(edge_index, node_features, edge_weights)
    y_true1 = mx.array([[30, 30, 30], [7, 7, 7], [0, 0, 0], [15, 15, 15], [0, 0, 0]])

    # Aggr: add, combine_root_func=None, without edge weights.
    conv = SimpleConv(aggr="add", combine_root_func=None)
    node_features = mx.ones((5, 3))
    edge_index = mx.array([[0, 1, 2, 3, 4], [0, 0, 1, 1, 3]])
    y_hat2 = conv(edge_index, node_features)
    y_true2 = mx.array([[2, 2, 2], [2, 2, 2], [0, 0, 0], [1, 1, 1], [0, 0, 0]])

    # Aggr: add, combine_root_func="self_loop", with edge weights.
    conv = SimpleConv(aggr="add", combine_root_func="self_loop")
    node_features = mx.ones((5, 3))
    edge_index = mx.array([[0, 1, 2, 3, 4], [0, 0, 1, 1, 3]])
    edge_weights = mx.array([10, 20, 5, 2, 15])
    y_hat3 = conv(edge_index, node_features, edge_weights)
    y_true3 = mx.array([[31, 31, 31], [8, 8, 8], [1, 1, 1], [16, 16, 16], [1, 1, 1]])

    # Aggr: add, combine_root_func="self_loop", without edge weights.
    conv = SimpleConv(aggr="add", combine_root_func="self_loop")
    node_features = mx.ones((5, 3))
    edge_index = mx.array([[0, 1, 2, 3, 4], [0, 0, 1, 1, 3]])
    y_hat4 = conv(edge_index, node_features)
    y_true4 = mx.array([[3, 3, 3], [3, 3, 3], [1, 1, 1], [2, 2, 2], [1, 1, 1]])

    # Aggr: add, combine_root_func="sum", without edge weights.
    conv = SimpleConv(aggr="add", combine_root_func="sum")
    node_features = mx.ones((5, 3))
    edge_index = mx.array([[0, 1, 2, 3, 4], [0, 0, 1, 1, 3]])
    y_hat5 = conv(edge_index, node_features)
    y_true5 = mx.array([[3, 3, 3], [3, 3, 3], [1, 1, 1], [2, 2, 2], [1, 1, 1]])

    # Aggr: add, combine_root_func="cat", without edge weights.
    conv = SimpleConv(aggr="add", combine_root_func="cat")
    node_features = mx.ones((5, 3))
    edge_index = mx.array([[0, 1, 2, 3, 4], [0, 0, 1, 1, 3]])
    y_hat6 = conv(edge_index, node_features)
    y_true6 = mx.array(
        [
            [1, 1, 1, 2, 2, 2],
            [1, 1, 1, 2, 2, 2],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0],
        ]
    )

    assert mx.array_equal(y_hat1, y_true1), "SimpleConv expected results y_true1 failed"
    assert mx.array_equal(y_hat2, y_true2), "SimpleConv expected results y_true2 failed"
    assert mx.array_equal(y_hat3, y_true3), "SimpleConv expected results y_true3 failed"
    assert mx.array_equal(y_hat4, y_true4), "SimpleConv expected results y_true4 failed"
    assert mx.array_equal(y_hat5, y_true5), "SimpleConv expected results y_true5 failed"
    assert mx.array_equal(y_hat6, y_true6), "SimpleConv expected results y_true6 failed"
