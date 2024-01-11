import mlx.core as mx
import pytest

from mlx_graphs.nn.conv.gat_conv import GATConv


def test_gat_conv():
    conv = GATConv(8, 20, heads=1)

    x = mx.random.uniform(0, 1, (4, 8))
    edge_index = mx.array([[0, 1, 2, 3], [0, 0, 1, 1]])

    y_hat = conv(x, edge_index)
    assert y_hat.shape == [4, 20], "Simple GATConv failed"

    # TODO: Add test with negative values in x
    # TODO: Add test with x of shape (n, d) with n != len(ei[0])
