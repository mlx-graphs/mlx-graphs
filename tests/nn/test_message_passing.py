import mlx.core as mx
import pytest

from mlx_graphs.nn.message_passing import MessagePassing


class MPNN(MessagePassing):
    def __init__(self, aggr: str = "add"):
        super().__init__(aggr=aggr)

    def __call__(
        self,
        node_features: mx.array,
        edge_index: mx.array,
        edge_weights: mx.array,
        **kwargs,
    ) -> mx.array:
        return self.propagate(
            node_features=node_features,
            edge_index=edge_index,
            message_kwargs={"edge_weights": edge_weights},
        )

    def message(
        self,
        src_features: mx.array,
        dst_features: mx.array,
        edge_weights: mx.array = None,
        **kwargs,
    ) -> mx.array:
        return (
            src_features
            if edge_weights is None
            else edge_weights.reshape(-1, 1) * src_features
        )


def test_sum_aggregation():
    mpnn = MPNN(aggr="add")

    # shape (3,)
    x = mx.array([1, 2, 3])

    edge_index = mx.array([[0, 1, 2], [1, 2, 0]])
    y0 = mx.array([3, 1, 2])
    y_hat0 = mpnn(x, edge_index, edge_weights=None)

    # shape (3, 2)
    x = mx.array([[1, 2], [3, 4], [5, 6]])

    edge_index = mx.array([[0, 1, 2], [1, 2, 0]])
    y1 = mx.array([[5, 6], [1, 2], [3, 4]])
    y_hat1 = mpnn(x, edge_index, edge_weights=None)

    edge_index = mx.array([[0, 2, 0], [1, 1, 1]])
    y2 = mx.array([[0, 0], [7, 10], [0, 0]])
    y_hat2 = mpnn(x, edge_index, edge_weights=None)

    edge_index = mx.array([[0, 2, 0], [0, 1, 0]])
    y3 = mx.array([[2, 4], [5, 6], [0, 0]])
    y_hat3 = mpnn(x, edge_index, edge_weights=None)

    # edge_index = mx.array([[0, 100, 0], [0, 1, 100]])
    # y4 = mx.array([[1, 2], [0, 0], [0, 0]])
    # y_hat4 = mpnn(x, edge_index, edge_weights=None)

    # shape (3, 2, 2)
    x = mx.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
    edge_index = mx.array([[0, 1, 0], [1, 1, 1]])
    y_hat5 = mpnn(x, edge_index, edge_weights=None)
    y5 = mx.array([[[0, 0], [0, 0]], [[7, 10], [13, 16]], [[0, 0], [0, 0]]])

    # shape (3, 2, 2, 1)
    edge_index = edge_index
    x = x.reshape(*x.shape, 1)
    y_hat6 = mpnn(x, edge_index, edge_weights=None)
    y6 = y5.reshape(*y5.shape, 1)

    assert mx.array_equal(y0, y_hat0), "Simple message passing failed"
    assert mx.array_equal(y1, y_hat1), "Simple message passing failed"
    assert mx.array_equal(y2, y_hat2), "Add message passing failed"
    assert mx.array_equal(y3, y_hat3), "Add message passing failed"
    # assert mx.array_equal(y4, y_hat4), "Out of bound message passing failed"
    # mlx.core.array' object has no attribute 'typecode
    assert mx.array_equal(y5, y_hat5), "Add message passing with multiple dims failed"
    assert mx.array_equal(y6, y_hat6), "Add message passing with multiple dims failed"


def test_sum_aggregation_with_edge_weight():
    mpnn = MPNN(aggr="add")
    x = mx.array([[1, 2], [3, 4], [5, 6]])

    edge_index = mx.array([[0, 1, 2], [1, 2, 0]])
    edge_weights = mx.array([0.1, 0.5, 1.0])
    y1 = mx.array([[5.0, 6.0], [0.1, 0.2], [1.5, 2]])
    y_hat1 = mpnn(x, edge_index, edge_weights=edge_weights)

    edge_index = mx.array([[0, 2, 0], [1, 1, 1]])
    edge_weights = mx.array([0.1, 0.5, 1.0])
    y2 = mx.array([[0, 0], [3.6, 5.2], [0, 0]])
    y_hat2 = mpnn(x, edge_index, edge_weights=edge_weights)

    assert mx.array_equal(y1, y_hat1), "Simple message passing with weights failed"
    assert mx.array_equal(y2, y_hat2), "Add message passing with weights failed"


def test_sum_aggregation_raise():
    mpnn = MPNN(aggr="add")
    x = mx.array([[1, 2], [3, 4], [5, 6]])

    with pytest.raises(Exception):
        edge_index = mx.array([[0, 1, 2], [1, 2, 0], [1, 2, 0]])
        mpnn(x, edge_index, edge_weights=None)

    with pytest.raises(Exception):
        edge_index = [[0, 1, 2], [1, 2, 0], [1, 2, 0]]
        mpnn(x, edge_index, edge_weights=None)


# ---------------------------------------------------------------------------
# Bipartite message passing: (src_features, dst_features) tuples
# ---------------------------------------------------------------------------


def test_bipartite_output_has_one_row_per_dst_node():
    """5 source nodes (e.g. authors) send messages to 2 destination nodes
    (e.g. papers): the output must have 2 rows, not 5."""
    mpnn = MPNN(aggr="add")
    x_src = mx.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    x_dst = mx.zeros((2, 7))  # different feature size is fine
    edge_index = mx.array([[0, 1, 4], [0, 0, 1]])  # src -> dst

    out = mpnn((x_src, x_dst), edge_index, edge_weights=None)

    assert out.shape == (2, 1)
    assert mx.array_equal(out, mx.array([[3.0], [5.0]]))  # 1+2, 5


def test_bipartite_more_dst_than_src():
    mpnn = MPNN(aggr="add")
    x_src = mx.array([[1.0], [2.0]])
    x_dst = mx.zeros((4, 1))
    edge_index = mx.array([[0, 1, 1], [3, 3, 0]])

    out = mpnn((x_src, x_dst), edge_index, edge_weights=None)

    assert out.shape == (4, 1)
    assert mx.array_equal(out, mx.array([[2.0], [0.0], [0.0], [3.0]]))


@pytest.mark.parametrize("aggr", ["add", "mean", "max"])
def test_tuple_of_same_features_matches_single_array(aggr):
    mpnn = MPNN(aggr=aggr)
    x = mx.random.normal((6, 3))
    edge_index = mx.array([[0, 1, 2, 3, 5], [1, 1, 2, 4, 0]])

    out_single = mpnn(x, edge_index, edge_weights=None)
    out_tuple = mpnn((x, x), edge_index, edge_weights=None)

    assert mx.allclose(out_single, out_tuple)
