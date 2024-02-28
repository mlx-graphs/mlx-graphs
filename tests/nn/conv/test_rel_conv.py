import mlx.core as mx

from mlx_graphs.nn.conv.rel_conv import GeneralizedRelationalConv

mx.random.seed(42)


def test_rel_conv():
    conv = GeneralizedRelationalConv(8, 8, 4)

    # 2D inputs (one graph)
    node_features = mx.random.uniform(0, 1, [6, 8])
    edge_index = mx.array([[0, 1, 2, 3], [0, 0, 1, 1]])
    edge_type = mx.array([0, 0, 1, 1])
    y_hat1 = conv(node_features, edge_index, edge_type, node_features)

    # 3D inputs (bs separate graphs)
    node_features = mx.random.uniform(0, 1, [3, 6, 8])
    boundary = mx.zeros((3, 6, 8))
    boundary[mx.array([0, 1, 2]), mx.array([0, 2, 3])] = 1.0
    edge_index = mx.array([[0, 1, 2, 3], [0, 0, 1, 1]])
    edge_type = mx.array([0, 0, 1, 1])
    y_hat2 = conv(node_features, edge_index, edge_type, boundary)

    # disconnected nodes
    node_features = mx.random.uniform(0, 1, [100, 8])
    edge_index = mx.array([[0, 1, 2, 3, 50], [0, 0, 1, 1, 99]])
    edge_type = mx.array([0, 0, 1, 1, 2])
    y_hat3 = conv(node_features, edge_index, edge_type, node_features)

    # dependent = True
    conv = GeneralizedRelationalConv(8, 8, 4, dependent=True)
    node_features = mx.random.uniform(0, 1, [3, 6, 8])
    edge_index = mx.array([[0, 1, 2, 3], [0, 0, 1, 1]])
    edge_type = mx.array([0, 0, 1, 1])
    query = mx.random.uniform(0, 1, [3, 8])
    y_hat4 = conv(node_features, edge_index, edge_type, node_features, query)

    # test 2D PNA aggregation
    conv = GeneralizedRelationalConv(8, 8, 4, aggregate_func="pna")
    node_features = mx.random.uniform(0, 1, [6, 8])
    edge_index = mx.array([[0, 1, 2, 3], [0, 0, 1, 1]])
    edge_type = mx.array([0, 0, 1, 1])
    y_hat5 = conv(node_features, edge_index, edge_type, node_features)

    # test 3D PNA aggregation
    node_features = mx.random.uniform(0, 1, [3, 6, 8])
    boundary = mx.zeros((3, 6, 8))
    boundary[mx.array([0, 1, 2]), mx.array([0, 2, 3])] = 1.0
    edge_index = mx.array([[0, 1, 2, 3], [0, 0, 1, 1]])
    edge_type = mx.array([0, 0, 1, 1])
    y_hat6 = conv(node_features, edge_index, edge_type, boundary)

    assert y_hat1.shape == (6, 8), "Simple GeneralizedRelationalConv failed"
    assert y_hat2.shape == (3, 6, 8), "3D GeneralizedRelationalConv failed"
    assert y_hat3.shape == (
        100,
        8,
    ), "GeneralizedRelationalConv with different shapes failed"
    assert y_hat4.shape == (
        3,
        6,
        8,
    ), "GeneralizedRelationalConv with dependent=True failed"
    assert y_hat5.shape == (6, 8), "2D GeneralizedRelationalConv with PNA failed"
    assert y_hat6.shape == (3, 6, 8), "3D GeneralizedRelationalConv with PNA failed"
