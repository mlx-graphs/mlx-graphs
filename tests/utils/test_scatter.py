import mlx.core as mx
import pytest

from mlx_graphs.utils import scatter, degree


@pytest.mark.parametrize(
    "src, index, num_nodes, aggr, expected",
    [
        # Test case for test_scatter
        (
            mx.array([1.0, 1.0, 1.0, 1.0]),
            mx.array([0, 0, 1, 2]),
            None,
            "softmax",
            mx.array([0.5, 0.5, 1, 1]),
        ),
        (
            mx.array([1.0, 1.0, 1.0, 1.0]),
            mx.array([0, 0, 1, 2]),
            3,
            "add",
            mx.array([2, 1, 1]),
        ),
        (
            mx.array([1.0, 1.0, 1.0, 1.0]),
            mx.array([0, 0, 1, 2]),
            3,
            "max",
            mx.array([1, 1, 1]),
        ),
        (
            mx.array([1.0, 1.0, 1.0, 1.0]),
            mx.array([0, 0, 1, 2]),
            3,
            "mean",
            mx.array([1, 1, 1]),
        ),
        # Test case for test_scatter_2D
        (
            mx.array([[1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0]]),
            mx.array([0, 0, 1, 2]),
            None,
            "softmax",
            mx.array([[0.5, 0.269], [0.5, 0.731], [1, 1], [1, 1]]),
        ),
        (
            mx.array([[1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0]]),
            mx.array([0, 0, 1, 2]),
            3,
            "add",
            mx.array([[2, 5], [1, 4], [1, 5]]),
        ),
        (
            mx.array([[1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0]]),
            mx.array([0, 0, 1, 2]),
            3,
            "max",
            mx.array([[1, 3], [1, 4], [1, 5]]),
        ),
        (
            mx.array([[1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0]]),
            mx.array([0, 0, 1, 2]),
            3,
            "mean",
            mx.array([[1, 2.5], [1, 4], [1, 5]]),
        ),
    ],
)
def test_scatter(src, index, num_nodes, aggr, expected):
    if num_nodes is None:
        num_nodes = src.shape[0] if aggr == "softmax" else index.max().item() + 1
    y_hat = scatter(src, index, num_nodes, aggr=aggr)

    if aggr == "softmax":
        assert mx.array_equal(
            mx.round(y_hat, 3), mx.round(expected, 3)
        ), f"Scatter {aggr} failed"
    else:
        assert mx.array_equal(y_hat, expected), f"Scatter {aggr} failed"


def test_degree():
    index = mx.array([0, 0, 1, 2])

    assert mx.array_equal(degree(index, 3), mx.array([2, 1, 1])), "degree failed"
    assert mx.array_equal(
        degree(index), mx.array([2, 1, 1])
    ), "degree with automatic `num_nodes` failed"
    assert mx.array_equal(
        degree(index, 4), mx.array([2, 1, 1, 0])
    ), "degree with missing node failed"

    index = mx.array([[0, 0, 1, 2], [1, 2, 0, 1]])

    with pytest.raises(ValueError):
        degree(index, 3)
