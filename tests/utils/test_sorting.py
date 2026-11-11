import mlx.core as mx
import pytest

from mlx_graphs.utils.sorting import index_to_ptr
from mlx_graphs.utils.topology import sort_edge_index, sort_edge_index_and_features


def test_index_to_ptr():
    # Every index appears at least once.
    ptr = index_to_ptr(mx.array([0, 0, 1, 2]), size=3)
    assert ptr.tolist() == [0, 2, 3, 4]
    assert ptr.dtype == mx.int64


def test_index_to_ptr_missing_indices():
    # Index 1 (middle) and index 4 (last) never appear.
    index = mx.array([0, 2, 2, 3])
    ptr = index_to_ptr(index, size=5).tolist()
    assert ptr == [0, 1, 1, 3, 4, 4]
    for i in range(5):
        assert index[ptr[i] : ptr[i + 1]].tolist() == [i] * (ptr[i + 1] - ptr[i])

    # Indices 0 and 1 (first) never appear.
    assert index_to_ptr(mx.array([2, 2]), size=3).tolist() == [0, 0, 0, 2]


def test_index_to_ptr_empty_index():
    index = mx.array([], dtype=mx.int32)
    assert index_to_ptr(index, size=3).tolist() == [0, 0, 0, 0]


def test_index_to_ptr_out_of_range():
    with pytest.raises(ValueError):
        index_to_ptr(mx.array([0, 3]), size=3)
    with pytest.raises(ValueError):
        index_to_ptr(mx.array([-1, 0]), size=3)


def test_sort_edge_index():
    edge_index = mx.array([[2, 1, 0, 2, 0, 2], [1, 0, 2, 0, 1, 2]])
    expected_edge_index = mx.array([[0, 0, 1, 2, 2, 2], [1, 2, 0, 0, 1, 2]])
    edge_features = mx.array([5, 3, 2, 4, 1, 6])
    expected_edge_features = mx.array([1, 2, 3, 4, 5, 6])
    s, idx = sort_edge_index(edge_index)
    assert mx.array_equal(expected_edge_index, s), "Index sorting failed"
    assert mx.array_equal(
        expected_edge_features, edge_features[idx]
    ), "Wrong sorted indices"


def test_sort_edge_index_and_features():
    edge_index = mx.array([[2, 1, 0, 2, 0, 2], [1, 0, 2, 0, 1, 2]])
    expected_edge_index = mx.array([[0, 0, 1, 2, 2, 2], [1, 2, 0, 0, 1, 2]])
    edge_features = mx.array([5, 3, 2, 4, 1, 6])
    expected_edge_features = mx.array([1, 2, 3, 4, 5, 6])
    s, f = sort_edge_index_and_features(edge_index, edge_features)
    assert mx.array_equal(expected_edge_index, s), "Index sorting failes"
    assert mx.array_equal(expected_edge_features, f), "Features sorting failed"
    # test 2d edge_features
    edge_features = mx.array([[5, 3, 2, 4, 1, 6], [2, 4, 5, 3, 6, 1]]).T
    expected_edge_features = mx.array([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]]).T
    s, f = sort_edge_index_and_features(edge_index, edge_features)
    assert mx.array_equal(expected_edge_index, s), "Index sorting failed"
    assert mx.array_equal(expected_edge_features, f), "Features sorting failed"
