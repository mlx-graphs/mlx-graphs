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
