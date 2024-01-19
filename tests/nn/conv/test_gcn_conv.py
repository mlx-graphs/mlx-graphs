import mlx.core as mx

from mlx_graphs.nn.conv.gcn_conv import GCNConv

mx.random.seed(42)


def test_gcn_conv():
    conv = GCNConv(8, 20)

    x = mx.random.uniform(0, 1, (6, 8))
    edge_index = mx.array([[0, 1, 2, 3], [0, 0, 1, 1]])
    y_hat1 = conv(x, edge_index)

    x = mx.random.uniform(-1, 1, (6, 8))
    edge_index = mx.array([[0, 1, 2, 3], [0, 0, 1, 1]])
    y_hat2 = conv(x, edge_index)

    conv = GCNConv(16, 32)
    x = mx.random.uniform(0, 1, (100, 16))
    edge_index = mx.array([[0, 1, 2, 3, 50], [0, 0, 1, 1, 99]])
    y_hat3 = conv(x, edge_index)

    conv = GCNConv(16, 32, bias=False)
    x = mx.random.uniform(0, 1, (100, 16))
    edge_index = mx.array([[0, 1, 2, 3, 50], [0, 0, 1, 1, 99]])
    y_hat4 = conv(x, edge_index)

    assert y_hat1.shape == [6, 20], "Simple GCNConv failed"
    assert y_hat2.shape == [6, 20], "GCNConv with negative values failed"
    assert y_hat3.shape == [100, 32], "GCNConv with different shapes failed"
    assert y_hat4.shape == [100, 32], "GCNConv without bias failed"
