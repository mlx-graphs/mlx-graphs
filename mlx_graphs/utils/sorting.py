import mlx.core as mx
import numpy as np

from mlx_graphs.utils.validators import (
    validate_edge_index,
    validate_edge_index_and_features,
)


def index_to_ptr(index: mx.array, size: int) -> mx.array:
    """Converts a vector of indices into a compressed pointer vector.

    Entry ``i`` of the result is the number of elements of ``index`` that are
    smaller than ``i``. When ``index`` is sorted, the elements equal to ``i``
    are located at ``ptr[i]:ptr[i + 1]``.

    With the sorted source row of an edge index, the result is the ``rowptr``
    of the CSR format. With the sorted target row, it is the ``colptr`` of the
    CSC format.

    Args:
        index: A 1D array of indices in ``[0, size)``.
        size: Number of possible indices, e.g. the number of nodes.

    Returns:
        An int64 array of length ``size + 1``.

    Example:

    .. code-block:: python

        >>> index = mx.array([0, 2, 2, 3])
        >>> index_to_ptr(index, size=5)
        array([0, 1, 1, 3, 4, 4], dtype=int64)
    """
    index_np = np.asarray(index).reshape(-1).astype(np.int64)
    if index_np.size > 0 and (index_np.min() < 0 or index_np.max() >= size):
        raise ValueError(f"index must only contain values in [0, {size})")
    ptr = np.zeros(size + 1, dtype=np.int64)
    np.cumsum(np.bincount(index_np, minlength=size), out=ptr[1:])
    return mx.array(ptr)


@validate_edge_index
def sort_edge_index(edge_index: mx.array) -> tuple[mx.array, mx.array]:
    """Sort the edge index.

    Args:
        edge_index: A [2, num_edges] array representing edge indices,
            where the first row contains source indices and the second row contains
            target indices.

    Returns:
        A tuple containing the sorted edge index and the
        corresponding sorting indices.
    """
    sorted_target_indices = mx.argsort(edge_index[1])
    target_sorted_index = edge_index[:, sorted_target_indices]
    sorted_source_indices = mx.argsort(target_sorted_index[0])
    sorted_edge_index = target_sorted_index[:, sorted_source_indices]
    sorting_indices = sorted_target_indices[sorted_source_indices]
    return sorted_edge_index, sorting_indices


@validate_edge_index_and_features
def sort_edge_index_and_features(
    edge_index: mx.array, edge_features: mx.array
) -> tuple[mx.array, mx.array]:
    """Sorts the given edge_index and their corresponding features.

    Args:
        edge_index: A [2, num_edges] array representing edge indices,
            where the first row contains source indices and the second row contains
            target indices.
        edge_features: An array representing edge features, where each row
            corresponds to an edge.

    Returns:
        A tuple containing the sorted edge index and the
        corresponding sorted edge features.
    """
    sorted_edge_index, sorting_indices = sort_edge_index(edge_index)
    sorted_edge_features = edge_features[sorting_indices]
    return sorted_edge_index, sorted_edge_features
