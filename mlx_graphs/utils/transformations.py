from typing import Optional

import mlx.core as mx
import numpy as np

from mlx_graphs.utils.validators import (
    validate_adjacency_matrix,
    validate_edge_index_and_features,
)


@validate_adjacency_matrix
def to_edge_index(
    adjacency_matrix: mx.array, *, dtype: mx.Dtype = mx.uint32
) -> mx.array:
    """
    Converts an adjacency matrix to an edge index representation.

    Args:
        adjacency_matrix (mlx.core.array): the input adjacency matrix
        dtype (mlx.core.Dtype): type of the output edge_index. Default to uint32.

    Returns:
        mlx.core.array: a [num_edges, 2] array representing the source and target nodes of each edge

    Example:

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
        [mx.array(x, dtype=dtype) for x in np.nonzero(adjacency_matrix)], axis=1
    )
    return edge_index


@validate_adjacency_matrix
def to_sparse_adjacency_matrix(
    adjacency_matrix: mx.array, *, dtype: mx.Dtype = mx.uint32
) -> tuple[mx.array, mx.array]:
    """
    Converts an adjacency matrix to a sparse representation as a tuple of edge index and edge features.

    Args:
        adjacency_matrix (mlx.core.array): the input adjacency matrix
        dtype (mlx.core.Dtype): type of the output edge_index. Default to uint32.

    Returns:
        tuple[mlx.core.array, mlx.core.array]: A tuple representing the edge index and edge features

    Example:

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
        # mx.array([[0, 1], [0, 2], [1, 2], [2, 0], [2, 1], [2, 2]])

        edge_features
        # mx.array([[1], [2], [3], [5], [1], [2]])
    """
    edge_index = to_edge_index(adjacency_matrix)
    edge_features = adjacency_matrix[edge_index[:, 0], edge_index[:, 1]]
    return edge_index, mx.expand_dims(edge_features, 1)


@validate_edge_index_and_features
def to_adjacency_matrix(
    edge_index: mx.array,
    edge_features: Optional[mx.array] = None,
    num_nodes: Optional[int] = None,
) -> mx.array:
    """
    Converts an edge index representation to an adjacency matrix.

    Args:
        edge_index (mlx.core.array): a [num_edges, 2] array representing the source and target nodes of each edge
        edge_features (mlx.core.array, optional): a [num_edges, 1] array representing the edge features corresponding to the edges in edge_index. Defaults to None.
        num_nodes (int, optional): the number of nodes in the graph. Defaults to the number of nodes in edge_index

    Returns:
        mlx.core.array: the resulting adjacency matrix
    """
    if num_nodes is not None:
        if mx.max(edge_index) > num_nodes - 1:
            raise ValueError(
                "num_nodes must be >= than the number of nodes in the edge_index ",
                f"(got num_nodes={num_nodes} and {mx.max(edge_index) + 1} nodes in index",
            )
    else:
        num_nodes = mx.max(edge_index) + 1

    if edge_features is not None:
        if edge_features.shape[1] != 1 or edge_features.shape[0] != edge_index.shape[0]:
            raise ValueError(
                f"edge features must be 1 per edge and scalars (got {edge_features.shape} edge_features shape and {edge_index.shape} edge_index shape)"
            )

    adjacency_matrix = mx.zeros((num_nodes, num_nodes), dtype=edge_index.dtype)
    if edge_features is None:
        adjacency_matrix[edge_index[:, 0], edge_index[:, 1]] = 1
    else:
        adjacency_matrix[edge_index[:, 0], edge_index[:, 1]] = edge_features.flatten()
    return adjacency_matrix
