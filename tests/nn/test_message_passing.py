import mlx.core as mx
import pytest

from mlx_graphs.nn.message_passing import MessagePassing


class MPNN(MessagePassing):

    def __init__(self, aggr: str="add"):
        super().__init__(aggr=aggr)

    def __call__(
        self, x: mx.array, edge_index: mx.array, edge_weight: mx.array, **kwargs
    ) -> mx.array:
        return self.propagate(x=x, edge_index=edge_index, edge_weight=edge_weight)

    def message(
        self, x_i: mx.array, x_j: mx.array, edge_weight: mx.array=None, **kwargs
    ) -> mx.array:
        return x_i if edge_weight is None else edge_weight.reshape(-1, 1) * x_i


def test_sum_aggregation():
    mpnn = MPNN(aggr="add")

    # shape (3,)
    x = mx.array([1, 2, 3])

    edge_index = mx.array([[0, 1, 2], [1, 2, 0]])
    y0 = mx.array([3, 1, 2])
    y_hat0 = mpnn(x, edge_index, edge_weight=None)
    
    # shape (3, 2)
    x = mx.array([[1, 2], [3, 4], [5, 6]])

    edge_index = mx.array([[0, 1, 2], [1, 2, 0]])
    y1 = mx.array([[5, 6], [1, 2], [3, 4]])
    y_hat1 = mpnn(x, edge_index, edge_weight=None)

    edge_index = mx.array([[0, 2, 0], [1, 1, 1]])
    y2 = mx.array([[0, 0], [7, 10], [0, 0]])
    y_hat2 = mpnn(x, edge_index, edge_weight=None)

    edge_index = mx.array([[0, 2, 0], [0, 1, 0]])
    y3 = mx.array([[2, 4], [5, 6], [0, 0]])
    y_hat3 = mpnn(x, edge_index, edge_weight=None)

    edge_index = mx.array([[0, 100, 0], [0, 1, 100]])
    y4 = mx.array([[1, 2], [0, 0], [0, 0]])
    y_hat4 = mpnn(x, edge_index, edge_weight=None)

    # shape (3, 2, 2)
    x = mx.array([
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]],
        [[9, 10], [11, 12]]
    ])
    edge_index = mx.array([[0, 1, 0], [1, 1, 1]])
    y_hat5 = mpnn(x, edge_index, edge_weight=None)
    y5 = mx.array([[[0, 0], [0, 0]], [[7, 10], [13, 16]], [[0, 0], [0, 0]]])

    # shape (3, 2, 2, 1)
    edge_index = edge_index.reshape(*edge_index.shape, 1)
    x = x.reshape(*x.shape, 1)
    y_hat6 = mpnn(x, edge_index, edge_weight=None)
    y6 = y5.reshape(*y5.shape, 1)
    
    assert mx.all(y0 == y_hat0), "Simple message passing failed"
    assert mx.all(y1 == y_hat1), "Simple message passing failed"
    assert mx.all(y2 == y_hat2), "Add message passing failed"
    assert mx.all(y3 == y_hat3), "Add message passing failed"
    assert mx.all(y4 == y_hat4), "Out of bound message passing failed"
    assert mx.all(y5 == y_hat5), "Add message passing with multiple dims failed"
    assert mx.all(y6 == y_hat6), "Add message passing with multiple dims failed"


def test_sum_aggregation_with_edge_weight():
    mpnn = MPNN(aggr="add")
    x = mx.array([[1, 2], [3, 4], [5, 6]])

    edge_index = mx.array([[0, 1, 2], [1, 2, 0]])
    edge_weight = mx.array([0.1, 0.5, 1.0])
    y1 = mx.array([[5.0, 6.0], [0.1, 0.2], [1.5, 2]])
    y_hat1 = mpnn(x, edge_index, edge_weight=edge_weight)

    edge_index = mx.array([[0, 2, 0], [1, 1, 1]])
    edge_weight = mx.array([0.1, 0.5, 1.0])
    y2 = mx.array([[0, 0], [3.6, 5.2], [0, 0]])
    y_hat2 = mpnn(x, edge_index, edge_weight=edge_weight)

    assert mx.all(y1 == y_hat1), "Simple message passing with weights failed"
    assert mx.all(y2 == y_hat2), "Add message passing with weights failed"


def test_sum_aggregation_raise():
    mpnn = MPNN(aggr="add")
    x = mx.array([[1, 2], [3, 4], [5, 6]])

    with pytest.raises(ValueError):
        edge_index = mx.array([[0, 1, 2], [1, 2, 0], [1, 2, 0]])
        mpnn(x, edge_index, edge_weight=None)

    with pytest.raises(ValueError):
        edge_index = [[0, 1, 2], [1, 2, 0], [1, 2, 0]]
        mpnn(x, edge_index, edge_weight=None)

test_sum_aggregation()