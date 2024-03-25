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
            mx.random.uniform(0, 1, [10, 16]),
            None,
            (10, 32),
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
            mx.random.uniform(0, 1, [100, 16]),
            None,
            (100, 32),
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
def test_gin_conv(layer, edge_index, node_features, edge_weights, expected):
    assert (
        expected == layer(edge_index, node_features, edge_weights=edge_weights).shape
    ), "GINConv failed"


@pytest.mark.parametrize(
    "layer, edge_index, node_features, edge_weights, edge_features, expected",
    [
        (
            GINConv(
                mlp=nn.Sequential(
                    nn.Linear(16, 32),
                    nn.ReLU(),
                    nn.Linear(32, 32),
                ),
                node_features_dim=16,
                edge_features_dim=10,
            ),
            mx.array([[0, 1, 2, 3], [0, 0, 1, 1]]),
            mx.random.uniform(0, 1, [10, 16]),
            None,
            mx.random.normal((4, 10)),
            (10, 32),
        ),
        (
            GINConv(
                mlp=nn.Sequential(
                    nn.Linear(16, 32),
                    nn.ReLU(),
                    nn.Linear(32, 32),
                ),
                node_features_dim=16,
                edge_features_dim=10,
            ),
            mx.array([[0, 1, 2, 3, 50], [0, 0, 1, 1, 99]]),
            mx.random.uniform(0, 1, [100, 16]),
            mx.random.normal(
                [
                    5,
                ]
            ),
            mx.random.normal((5, 10)),
            (100, 32),
        ),
    ],
)
def test_gine_conv(
    layer, edge_index, node_features, edge_weights, edge_features, expected
):
    assert (
        expected
        == layer(
            edge_index,
            node_features,
            edge_weights=edge_weights,
            edge_features=edge_features,
        ).shape
    ), "GINEConv failed"
