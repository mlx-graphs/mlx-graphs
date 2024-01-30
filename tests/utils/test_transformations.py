import mlx.core as mx
from mlx_graphs.utils.transformations import (
    to_edge_index,
    to_sparse_adjacency_matrix,
    to_adjacency_matrix,
    get_unique_edge_indices,
    add_self_loops,
    remove_self_loops,
)
import pytest


@pytest.mark.parametrize(
    "dtype",
    [
        (mx.uint8),
        (mx.uint16),
        (mx.uint32),
        (mx.uint64),
        (mx.int8),
        (mx.int16),
        (mx.int32),
        (mx.int64),
    ],
)
def test_to_edge_index_dtype(dtype):
    matrix = mx.array([[0, 1], [3, 0]])
    edge_index = to_edge_index(matrix, dtype=dtype)
    assert edge_index.dtype == dtype, "dtype of returned array incorrect"


def test_to_edge_index():
    matrix = mx.array([[0, 1, 2], [3, 0, 0], [5, 1, 2]])
    edge_index = to_edge_index(matrix)
    assert edge_index.dtype == mx.uint32, "Default dtype of returned array != uint32"

    expected_output = mx.array([[0, 0, 1, 2, 2, 2], [1, 2, 0, 0, 1, 2]])
    assert mx.array_equal(
        edge_index, expected_output
    ), "Incorrectly computed edge index"


def test_to_sparse_adjacency_matrix():
    matrix = mx.array([[0, 1, 2], [3, 0, 0], [5, 1, 2]])
    edge_index, edge_features = to_sparse_adjacency_matrix(matrix)

    expected_index = mx.array([[0, 0, 1, 2, 2, 2], [1, 2, 0, 0, 1, 2]])
    assert mx.array_equal(edge_index, expected_index), "Incorrect computed edge index"
    expected_features = mx.array([1, 2, 3, 5, 1, 2])
    assert mx.array_equal(edge_features, expected_features), "Incorrect edge features"


def test_to_adjacency_matrix():
    edge_index = mx.array([[0, 0, 1, 2, 2, 2], [1, 2, 0, 0, 1, 2]])
    edge_features = mx.array([1, 2, 3, 5, 1, 2])

    # 3 nodes
    num_nodes = 3
    expected_binary_matrix = mx.array([[0, 1, 1], [1, 0, 0], [1, 1, 1]])
    adj_matrix = to_adjacency_matrix(edge_index, num_nodes=num_nodes)
    assert mx.array_equal(
        expected_binary_matrix, adj_matrix
    ), "Incorrect conversion to adjacency matrix"
    expected_matrix = mx.array([[0, 1, 2], [3, 0, 0], [5, 1, 2]])
    weighted_adj_matrix = to_adjacency_matrix(
        edge_index, edge_features=edge_features, num_nodes=num_nodes
    )
    assert mx.array_equal(
        expected_matrix, weighted_adj_matrix
    ), "Incorrect conversion to weighted adjacency matrix"

    # 4 nodes (extra padding)
    num_nodes = 4
    expected_binary_matrix = mx.array(
        [[0, 1, 1, 0], [1, 0, 0, 0], [1, 1, 1, 0], [0, 0, 0, 0]]
    )
    adj_matrix = to_adjacency_matrix(edge_index, num_nodes=num_nodes)
    assert mx.array_equal(
        expected_binary_matrix, adj_matrix
    ), "Incorrect conversion to adjacency matrix"
    expected_matrix = mx.array([[0, 1, 2, 0], [3, 0, 0, 0], [5, 1, 2, 0], [0, 0, 0, 0]])
    weighted_adj_matrix = to_adjacency_matrix(
        edge_index, edge_features=edge_features, num_nodes=num_nodes
    )
    assert mx.array_equal(
        expected_matrix, weighted_adj_matrix
    ), "Incorrect conversion to weighted adjacency matrix"

    # 2 nodes (expect error as there are 3 in index)
    with pytest.raises(ValueError):
        to_adjacency_matrix(edge_index, num_nodes=2)

    # non 2D edge_index
    with pytest.raises(ValueError):
        to_adjacency_matrix(edge_index.reshape([1, 2, 6]), num_nodes=3)

    # column edge_index
    with pytest.raises(ValueError):
        to_adjacency_matrix(edge_index.transpose(), num_nodes=3)

    # more features than edges
    with pytest.raises(ValueError):
        edge_features = mx.array([1, 2, 3, 5, 1, 2, 5])
        to_adjacency_matrix(edge_index, edge_features=edge_features, num_nodes=3)

    # less features than edges
    with pytest.raises(ValueError):
        edge_features = mx.array([1, 2, 3, 5, 1])
        to_adjacency_matrix(edge_index, edge_features=edge_features, num_nodes=3)


def test_get_unique_edges():
    edge_index_1 = mx.array(
        [
            [0, 1, 1, 2],
            [1, 0, 2, 1],
        ]
    )
    edge_index_2 = mx.array(
        [
            [1, 2, 2],
            [2, 1, 2],
        ]
    )
    idx = get_unique_edge_indices(edge_index_1, edge_index_2)
    expected_idx = mx.array([0, 1])
    assert mx.array_equal(idx, expected_idx)


def test_add_self_loops():
    edge_index = mx.array([[0, 0, 1, 1], [0, 1, 0, 2]])
    edge_features = mx.ones([4, 2])

    # just index
    x, y = add_self_loops(edge_index)
    expected_x = mx.array([[0, 0, 1, 1, 0, 1, 2], [0, 1, 0, 2, 0, 1, 2]])
    assert y is None
    assert mx.array_equal(x, expected_x)

    # just index with extra nodes
    num_nodes = 5
    x, y = add_self_loops(edge_index, num_nodes=num_nodes)
    expected_x = mx.array([[0, 0, 1, 1, 0, 1, 2, 3, 4], [0, 1, 0, 2, 0, 1, 2, 3, 4]])
    assert y is None
    assert mx.array_equal(x, expected_x)

    # index and features
    x, y = add_self_loops(edge_index, edge_features)
    expected_x = mx.array([[0, 0, 1, 1, 0, 1, 2], [0, 1, 0, 2, 0, 1, 2]])
    expected_y = mx.ones([7, 2])
    assert mx.array_equal(y, expected_y)
    assert mx.array_equal(x, expected_x)

    # index and features with custom fill
    fill = 2
    x, y = add_self_loops(edge_index, edge_features, fill_value=fill)
    expected_x = mx.array([[0, 0, 1, 1, 0, 1, 2], [0, 1, 0, 2, 0, 1, 2]])
    expected_y = mx.concatenate([edge_features, mx.ones([3, 2]) * fill], 0)
    assert mx.array_equal(y, expected_y)
    assert mx.array_equal(x, expected_x)

    # don't allow repeated
    x, y = add_self_loops(edge_index, edge_features, allow_repeated=False)
    expected_x = mx.array([[0, 0, 1, 1, 1, 2], [0, 1, 0, 2, 1, 2]])
    expected_y = mx.ones([6, 2])
    assert mx.array_equal(y, expected_y)
    assert mx.array_equal(x, expected_x)


def test_remove_self_loops():
    edge_index = mx.array([[0, 0, 1, 1], [0, 1, 0, 2]])
    edge_features = mx.random.normal([4, 2])

    # just index
    x, y = remove_self_loops(edge_index)
    expected_x = mx.array([[0, 1, 1], [1, 0, 2]])
    assert y is None
    assert mx.array_equal(x, expected_x)

    # index and features
    x, y = remove_self_loops(edge_index, edge_features)
    expected_x = mx.array([[0, 1, 1], [1, 0, 2]])
    expected_y = edge_features[1:]
    assert mx.array_equal(y, expected_y)
    assert mx.array_equal(x, expected_x)

    # inactive on graph with no self loops
    edge_index = mx.array([[0, 1, 1], [1, 0, 2]])
    x, _ = remove_self_loops(edge_index)
    expected_x = mx.array([[0, 1, 1], [1, 0, 2]])
    assert mx.array_equal(x, expected_x)
