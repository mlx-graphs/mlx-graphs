import mlx.core as mx

from mlx_graphs.nn.conv.gatv2_conf import GATv2Conv 

mx.random.seed(42)


def test_gat_conv():
    conv = GATv2Conv(8, 20, heads=1)

    node_features = mx.random.uniform(0, 1, [6, 8])
    edge_index = mx.array([[0, 1, 2, 3], [0, 0, 1, 1]])
    y_hat1 = conv(edge_index, node_features)

    node_features = mx.random.uniform(-1, 1, [6, 8])
    edge_index = mx.array([[0, 1, 2, 3], [0, 0, 1, 1]])
    y_hat2 = conv(edge_index, node_features)

    conv = GATv2Conv(16, 32, heads=1)
    node_features = mx.random.uniform(0, 1, [100, 16])
    edge_index = mx.array([[0, 1, 2, 3, 50], [0, 0, 1, 1, 99]])
    y_hat3 = conv(edge_index, node_features)

    conv = GATv2Conv(16, 32, heads=3, concat=True)
    node_features = mx.random.uniform(0, 1, [100, 16])
    edge_index = mx.array([[0, 1, 2, 3, 50], [0, 0, 1, 1, 99]])
    y_hat4 = conv(edge_index, node_features)

    conv = GATv2Conv(16, 32, heads=3, concat=False)
    node_features = mx.random.uniform(0, 1, [100, 16])
    edge_index = mx.array([[0, 1, 2, 3, 50], [0, 0, 1, 1, 99]])
    y_hat5 = conv(edge_index, node_features)

    conv = GATv2Conv(16, 32, heads=3, concat=False, edge_features_dim=10)
    node_features = mx.random.uniform(0, 1, [100, 16])
    edge_features = mx.random.uniform(0, 1, [5, 10])
    edge_index = mx.array([[0, 1, 2, 3, 50], [0, 0, 1, 1, 99]])
    y_hat6 = conv(edge_index, node_features, edge_features=edge_features)

    assert y_hat1.shape == (6, 20), "Simple GATv2Conv failed"
    assert y_hat2.shape == (6, 20), "GATv2Conv with negative values failed"
    assert y_hat3.shape == (100, 32), "GATv2Conv with different shapes failed"
    assert y_hat4.shape == (100, 32 * 3), "GATv2Conv with multiple heads concat failed"
    assert y_hat5.shape == (
        100,
        32,
    ), "GATv2Conv with multiple heads without concat failed"
    assert y_hat6.shape == (100, 32), "GATv2Conv with edge features failed"
