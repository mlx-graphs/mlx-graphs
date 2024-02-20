from typing import Optional, Union, overload

import mlx.core as mx
import numpy as np

from mlx_graphs.utils.validators import (
    validate_adjacency_matrix,
    validate_edge_index,
    validate_edge_index_and_features,
)


@validate_adjacency_matrix
def to_edge_index(
    adjacency_matrix: mx.array, *, dtype: mx.Dtype = mx.uint32
) -> mx.array:
    """
    Converts an adjacency matrix to an edge index representation.

    Args:
        adjacency_matrix: the input adjacency matrix
        dtype: type of the output edge_index. Default to uint32.

    Returns:
        A [2, num_edges] array representing the source and target nodes of each edge

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
        [mx.array(x, dtype=dtype) for x in np.nonzero(adjacency_matrix)]
    )
    return edge_index


@validate_adjacency_matrix
def to_sparse_adjacency_matrix(
    adjacency_matrix: mx.array, *, dtype: mx.Dtype = mx.uint32
) -> tuple[mx.array, mx.array]:
    """
    Converts an adjacency matrix to a sparse representation as a tuple of edge index
    and edge features.

    Args:
        adjacency_matrix: the input adjacency matrix
        dtype: type of the output edge_index. Default to uint32.

    Returns:
        A tuple representing the edge index and edge features

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
        # mx.array([[0, 0, 1, 2, 2, 2], [1, 2, 0, 0, 1, 2]])

        edge_features
        # mx.array([1, 2, 3, 5, 1, 2])
    """
    edge_index = to_edge_index(adjacency_matrix)
    edge_features = adjacency_matrix[edge_index[0], edge_index[1]]
    return edge_index, edge_features


@validate_edge_index_and_features
def to_adjacency_matrix(
    edge_index: mx.array,
    edge_features: Optional[mx.array] = None,
    num_nodes: Optional[int] = None,
) -> mx.array:
    """
    Converts an edge index representation to an adjacency matrix.

    Args:
        edge_index: a [2, num_edges] array representing the source and target nodes
            of each edge
        edge_features: a 1-dimensional array representing the features corresponding
            to the edges in edge_index. Defaults to None.
        num_nodes: the number of nodes in the graph. Defaults to the number of nodes
            in edge_index

    Returns:
        The resulting adjacency matrix
    """
    if num_nodes is not None:
        if mx.max(edge_index) > num_nodes - 1:
            raise ValueError(
                "num_nodes must be >= than the number of nodes in the edge_index ",
                f"(got num_nodes={num_nodes} and {mx.max(edge_index) + 1}",
                "nodes in index",
            )
    else:
        num_nodes = (mx.max(edge_index) + 1).item()

    if edge_features is not None:
        if edge_features.ndim != 1:
            raise ValueError(
                "edge_features must be 1-dimensional",
                f"(got {edge_features.ndim} dimensions)",
            )

    adjacency_matrix = mx.zeros((num_nodes, num_nodes), dtype=edge_index.dtype)
    if edge_features is None:
        adjacency_matrix[edge_index[0], edge_index[1]] = 1
    else:
        adjacency_matrix[edge_index[0], edge_index[1]] = edge_features
    return adjacency_matrix


@validate_edge_index
def get_src_dst_features(
    edge_index: mx.array,
    node_features: Union[mx.array, tuple[mx.array, mx.array]],
) -> tuple[mx.array, mx.array]:
    """
    Extracts source and destination node features based on the given edge indices.

    Args:
        edge_index: a [2, num_edges] array representing the source and target nodes
            of each edge
        node_features: The input array of node features.

    Returns:
        A tuple containing source and destination features.
    """
    src_idx, dst_idx = edge_index

    if isinstance(node_features, tuple):
        src_val, dst_val = node_features
        src_val = src_val[src_idx]
        dst_val = dst_val[dst_idx]

    elif isinstance(node_features, mx.array):
        src_val = node_features[src_idx]
        dst_val = node_features[dst_idx]

    else:
        raise ValueError(
            "Invalid type for argument `array`, should be a `mx.array` or a `tuple`."
        )

    return src_val, dst_val


def get_unique_edge_indices(edge_index_1: mx.array, edge_index_2: mx.array) -> mx.array:
    """
    Computes the indices of the edges in edge_index_1 that are NOT present in
    edge_index_2

    Args:
        edge_index_1: The first edge index array.
        edge_index_2: The second edge index array.

    Returns:
        The indices of the edges in edge_index_1 that are not present in edge_index_2


    Example:

    .. code-block:: python

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
        x = get_unique_edge_indices(edge_index_1, edge_index_2)
        # [0, 1]
    """
    edge_2_unique = np.unique(edge_index_2.transpose(), axis=0).tolist()

    return mx.array(
        [
            i
            for i, edge in enumerate(np.array(edge_index_1.transpose(), copy=False))
            if edge.tolist() not in edge_2_unique
        ]
    )


@overload
def add_self_loops(
    edge_index: mx.array,
    edge_features: mx.array,
    num_nodes: Optional[int] = None,
    fill_value: Optional[Union[float, mx.array]] = 1,
    allow_repeated: Optional[bool] = True,
) -> tuple[mx.array, mx.array]:
    ...


@overload
def add_self_loops(
    edge_index: mx.array,
    edge_features=None,
    num_nodes: Optional[int] = None,
    fill_value: Optional[Union[float, mx.array]] = 1,
    allow_repeated: Optional[bool] = True,
) -> mx.array:
    ...


@validate_edge_index_and_features
def add_self_loops(
    edge_index: mx.array,
    edge_features: Optional[mx.array] = None,
    num_nodes: Optional[int] = None,
    fill_value: Optional[Union[float, mx.array]] = 1,
    allow_repeated: Optional[bool] = True,
) -> Union[mx.array, tuple[mx.array, mx.array]]:
    """
    Adds self-loops to the given graph represented by edge_index and edge_features.

    Args:
        edge_index: a [2, num_edges] array representing the source and target nodes
            of each edge
        edge_features: Optional tensor representing features associated with each edge,
            with shape [num_edges, num_edge_features]
        num_nodes: Optional number of nodes in the graph. If not provided, it is
            inferred from edge_index.
        fill_value: Value used for filling the self-loop features. Default is 1.
        allow_repeated: Specify whether to add self-loops for all nodes, even if they
            are already in the edge_index. Defaults to True.

    Returns:
        A tuple containing the updated edge_index and edge_features with self-loops.

    """
    if num_nodes is not None:
        if mx.max(edge_index) > num_nodes - 1:
            raise ValueError(
                "num_nodes must be >= than the number of nodes in the edge_index ",
                f"(got num_nodes={num_nodes} and {mx.max(edge_index) + 1}",
                "nodes in index",
            )
    else:
        num_nodes = (mx.max(edge_index) + 1).item()

    # add self loops to index
    self_loop_index = mx.repeat(mx.expand_dims(mx.arange(num_nodes), 0), 2, 0)
    if not allow_repeated:
        self_loop_index = self_loop_index[
            :, get_unique_edge_indices(self_loop_index, edge_index)
        ]
    full_edge_index = mx.concatenate([edge_index, self_loop_index], 1)

    if edge_features is not None:
        # add self loops to features
        self_loop_features = (
            mx.ones([self_loop_index.shape[1], edge_features.shape[1]]) * fill_value
        )
        full_edge_features = mx.concatenate([edge_features, self_loop_features], 0)
        return full_edge_index, full_edge_features
    return full_edge_index


@overload
def remove_self_loops(
    edge_index: mx.array,
    edge_features=None,
) -> mx.array:
    ...


@overload
def remove_self_loops(
    edge_index: mx.array,
    edge_features: mx.array,
) -> tuple[mx.array, mx.array]:
    ...


@validate_edge_index_and_features
def remove_self_loops(
    edge_index: mx.array,
    edge_features: Optional[mx.array] = None,
) -> Union[mx.array, tuple[mx.array, mx.array]]:
    """
    Removes self-loops from the given graph represented by edge_index and edge_features.

    Args:
        edge_index: a [2, num_edges] array representing the source and target nodes
            of each edge
        edge_features: Optional tensor representing features associated with each edge,
            with shape [num_edges, num_edge_features]

    Returns:
        A tuple containing the updated edge_index and edge_features without self-loops.

    """
    num_nodes = (mx.max(edge_index) + 1).item()

    # add self loops to index
    self_loop_index = mx.repeat(mx.expand_dims(mx.arange(num_nodes), 0), 2, 0)
    preserved_idx = get_unique_edge_indices(edge_index, self_loop_index)
    no_self_loop_index = mx.array([[]])
    if len(preserved_idx) != 0:
        no_self_loop_index = edge_index[:, preserved_idx]

    if edge_features is not None:
        no_self_loop_features = mx.array([[]])
        if len(preserved_idx) != 0:
            no_self_loop_features = edge_features[preserved_idx]
        return no_self_loop_index, no_self_loop_features
    return no_self_loop_index


@overload
def to_undirected(edge_index: mx.array, edge_features=None) -> mx.array:
    ...


@overload
def to_undirected(
    edge_index: mx.array, edge_features: mx.array
) -> tuple[mx.array, mx.array]:
    ...


@validate_edge_index_and_features
def to_undirected(
    edge_index: mx.array, edge_features: Optional[mx.array] = None
) -> Union[mx.array, tuple[mx.array, mx.array]]:
    """
    Converts a graph given as `edge_index` and, optionally, `edge_features`
    to an undirected one.

    Args:
        edge_index: The edge index of the graph.
        edge_features: Edge features associated with each edge.

    Returns:
        The undirected `edge_index` (and `edge_features`)
    """

    src, dst = edge_index
    undirected_edge_index = mx.concatenate(
        [mx.stack([src, dst]), mx.stack([dst, src])], 1
    )
    if edge_features is not None:
        undirected_edge_features = mx.concatenate([edge_features, edge_features], 0)
        return undirected_edge_index, undirected_edge_features
    return undirected_edge_index


@validate_edge_index
def remove_duplicate_directed_edges(edge_index: mx.array) -> mx.array:
    """
    Removes the duplicate directed edges in `edge_index`.

    Args:
        edge_index: The `edge_index` on which remove duplicate edges.

    Returns:
        An `edge_index` without duplicates.
    """
    base_dtype = edge_index.dtype
    if base_dtype == mx.int64:
        raise ValueError("Does not support yet `edge_index` with int64 dtype.")

    # TODO: this method shall be updated to full mlx once unique() and structured
    # arrays are implemented
    edge_index = np.array(edge_index, np.int32)

    # Convert edge_index to a structured array to leverage np.unique for
    # multi-column uniqueness
    dtype = [("node1", edge_index.dtype), ("node2", edge_index.dtype)]
    structured_edges = np.array(list(zip(edge_index[0], edge_index[1])), dtype=dtype)

    unique_edges = np.unique(structured_edges)
    unique_edge_index = np.array([unique_edges["node1"], unique_edges["node2"]])

    return mx.array(unique_edge_index, mx.int32)
