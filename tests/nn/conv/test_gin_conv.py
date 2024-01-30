import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_graphs.nn.conv.gin_conv import GINConv


@pytest.mark.parametrize(
    "layer, edge_index, node_features, edge_weights, expected",
    [
        (
            GINConv(
                mlp=nn.Sequential(
                    nn.Linear(16, 32),
                    nn.ReLU(),
                    nn.Linear(32, 32),
                ),
            ),
            mx.array([[0, 1, 2, 3], [0, 0, 1, 1]]),
            mx.random.uniform(0, 1, (10, 16)),
            None,
            [10, 32],
        ),
        (
            GINConv(
                mlp=nn.Sequential(
                    nn.Linear(16, 32),
                    nn.ReLU(),
                    nn.Linear(32, 32),
                ),
            ),
            mx.array([[0, 1, 2, 3, 50], [0, 0, 1, 1, 99]]),
            mx.random.uniform(0, 1, (100, 16)),
            None,
            [100, 32],
        ),
        (
            GINConv(
                mlp=nn.Sequential(
                    nn.Linear(16, 32),
                    nn.ReLU(),
                    nn.Linear(32, 32),
                ),
            ),
            mx.array([[0, 1, 2, 3, 50], [0, 0, 1, 1, 99]]),
            mx.random.uniform(0, 1, (100, 16)),
            mx.random.normal((5,)),
            [100, 32],
        ),
    ],
)
def test_gin_conv(layer, edge_index, node_features, edge_weights, expected):
    assert (
        expected == layer(edge_index, node_features, edge_weights=edge_weights).shape
    ), "GINConv failed"
