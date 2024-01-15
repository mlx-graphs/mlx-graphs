from typing import Optional
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
def to_edge_index(adjacency_matrix: mx.array, dtype: mx.Dtype = mx.uint32) -> mx.array:
    """
    Converts an adjacency matrix to an edge index representation.

    Args:
        adjacency_matrix (mlx.core.array): the input adjacency matrix
        dtype (mlx.core.Dtype): type of the output edge_index. Default to uint32.

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
    edge_index = mx.stack(
        [mx.array(x, dtype=dtype) for x in np.nonzero(adjacency_matrix)]
    )
    return edge_index


@check_adjacency_matrix
def to_sparse_adjacency_matrix(
    adjacency_matrix: mx.array, dtype: mx.Dtype = mx.uint32
) -> tuple[mx.array, mx.array]:
    """
    Converts an adjacency matrix to a sparse representation as a tuple of edge index and edge features.

    Args:
        adjacency_matrix (mlx.core.array): the input adjacency matrix
        dtype (mlx.core.Dtype): type of the output edge_index. Default to uint32.

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


def to_adjacency_matrix(
    edge_index: mx.array, num_nodes: int, edge_features: Optional[mx.array] = None
) -> mx.array:
    """
    Converts an edge index representation to an adjacency matrix.

    Args:
        edge_index (mlx.core.array): a [2, num_edges] array representing the source and target nodes of each edge
        num_nodes (int): the number of nodes in the graph
        edge_features (mlx.core.array, optional): edge features corresponding to the edges in edge_index. Defaults to None.

    Returns:
        mlx.core.array: the resulting adjacency matrix
    """
    if mx.max(edge_index) > num_nodes - 1:
        raise ValueError(
            "num_nodes must be >= than the number of nodes in the edge_index ",
            f"(got num_nodes={num_nodes} and {mx.max(edge_index) + 1} nodes in index",
        )
    if edge_index.ndim != 2:
        raise ValueError(
            "edge_index must be 2-dimensional with shape [2, num_edges]",
            f"(got {edge_index.ndim} dimensions)",
        )
    if edge_index.shape[0] != 2:
        raise ValueError(
            "edge_index must be 2-dimensional with shape [2, num_edges]",
            f"(got {edge_index.shape} shape)",
        )

    if edge_features is not None:
        if edge_features.ndim != 1:
            raise ValueError(
                f"edge_features must be 1-dimensional (got {edge_features.ndim} dimensions)"
            )
        if edge_index.shape[1] != edge_features.shape[0]:
            raise ValueError(
                "edge_features must be 1 per edge ",
                f"(got {edge_index.shape[1]} edges and {edge_features.shape[0]} features)",
            )
    adjacency_matrix = mx.zeros((num_nodes, num_nodes), dtype=edge_index.dtype)
    if edge_features is None:
        adjacency_matrix[edge_index[0], edge_index[1]] = 1
    else:
        adjacency_matrix[edge_index[0], edge_index[1]] = edge_features
    return adjacency_matrix
