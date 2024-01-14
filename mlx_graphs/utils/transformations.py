import mlx.core as mx
import numpy as np


def check_adjacency_matrix(func):
    """Decorator function to check the validity of an adjacency matrix."""

    def wrapper(adjacency_matrix, *args, **kwargs):
        if adjacency_matrix.ndim != 2:
            raise ValueError(
                f"Adjacency matrix must be two-dimensional (got {adjacency_matrix.ndim} dimensions)"
            )
        if not mx.equal(*adjacency_matrix.shape):
            raise ValueError(
                f"Adjacency matrix must be a square matrix (got {adjacency_matrix.shape} shape)"
            )
        return func(adjacency_matrix, *args, **kwargs)

    return wrapper


@check_adjacency_matrix
def to_edge_index(adjacency_matrix: mx.array) -> mx.array:
    """
    Converts an adjacency matrix to an edge index representation.

    Args:
        adjacency_matrix (mlx.core.array): the input adjacency matrix

    Returns:
        mlx.core.array: a [2, num_edges] array representing the source and target nodes of each edge

    .. code-block:: python
        matrix = mx.array(
            [
                [0, 1, 2],
                [3, 0, 0],
                [5, 1, 2],
            ]
        )
        edge_index = to_edge_index(matrix)
        # mx.array([[0, 0, 1, 2, 2, 2], [1, 2, 0, 0, 1, 2]])
    """
    input_dtype = adjacency_matrix.dtype
    edge_index = mx.stack(
        [mx.array(x, dtype=input_dtype) for x in np.nonzero(adjacency_matrix)]
    )
    return edge_index


@check_adjacency_matrix
def to_sparse_adjacency_matrix(adjacency_matrix: mx.array) -> tuple[mx.array, mx.array]:
    """
    Converts a dense adjacency matrix to a sparse one, represented as an tuple of and
    edge index and edge features.

    Args:
        adjacency_matrix (mlx.core.array): the input adjacency matrix

    Returns:
        tuple[mlx.core.array, mlx.core.array]: A tuple representing the edge index and edge features

    .. code-block:: python
        matrix = mx.array(
            [
                [0, 1, 2],
                [3, 0, 0],
                [5, 1, 2],
            ]
        )
        edge_index, edge_features = to_sparse_adjacency_matrix(matrix)

        edge_index
        # mx.array([[0, 0, 1, 2, 2, 2], [1, 2, 0, 0, 1, 2]])

        edge_features
        # mx.array([1, 2, 3, 5, 1, 2])
    """
    edge_index = to_edge_index(adjacency_matrix)
    edge_features = adjacency_matrix[edge_index[0], edge_index[1]]
    return edge_index, edge_features
