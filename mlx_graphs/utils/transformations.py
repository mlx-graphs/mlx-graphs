from typing import Optional, Union, overload

import mlx.core as mx
import numpy as np

from mlx_graphs.utils.scatter import scatter
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
        [
            mx.array(x, dtype=dtype)
            for x in np.nonzero(np.array(adjacency_matrix, copy=False))
        ]
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


def _to_numpy(array: mx.array) -> np.ndarray:
    """Host view of an ``mx.array`` (evaluates it; no copy when possible).

    Unsigned indices are widened to int64, so that arithmetic and sentinels
    such as ``max(initial=-1)`` cannot overflow.
    """
    out = np.asarray(array)
    if out.dtype.kind == "u":
        out = out.astype(np.int64)
    return out


def _edge_keys(edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
    """Encodes each ``(src, dst)`` pair as one int64, ordered by ``(src, dst)``."""
    return edge_index[0].astype(np.int64) * num_nodes + edge_index[1].astype(np.int64)


def _infer_num_nodes(edge_index: mx.array, num_nodes: Optional[int] = None) -> int:
    if num_nodes is not None:
        return int(num_nodes)
    if edge_index.size == 0:
        return 0
    return int(mx.max(edge_index).item()) + 1


def _empty_like(edge_index: mx.array) -> mx.array:
    return mx.zeros((2, 0), dtype=edge_index.dtype)


def _take_edges(edge_index: mx.array, keep: np.ndarray) -> mx.array:
    """Columns ``keep`` (host indices) of ``edge_index``; ``[2, 0]`` if empty."""
    if keep.size == 0:
        return _empty_like(edge_index)
    return edge_index[:, mx.array(keep)]


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
    edges_1, edges_2 = _to_numpy(edge_index_1), _to_numpy(edge_index_2)
    num_nodes = int(max(edges_1.max(initial=-1), edges_2.max(initial=-1))) + 1
    present = np.isin(_edge_keys(edges_1, num_nodes), _edge_keys(edges_2, num_nodes))
    return mx.array(np.nonzero(~present)[0])


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
        if edge_index.size > 0 and mx.max(edge_index).item() > num_nodes - 1:
            raise ValueError(
                "num_nodes must be >= than the number of nodes in the edge_index ",
                f"(got num_nodes={num_nodes} and {mx.max(edge_index).item() + 1}",
                "nodes in index",
            )
    else:
        num_nodes = _infer_num_nodes(edge_index)

    # add self loops to index
    loop_nodes = mx.arange(num_nodes, dtype=edge_index.dtype)
    if not allow_repeated:
        edges = _to_numpy(edge_index)
        has_loop = np.zeros(num_nodes, dtype=bool)
        has_loop[edges[0][edges[0] == edges[1]]] = True
        loop_nodes = mx.array(np.nonzero(~has_loop)[0], dtype=edge_index.dtype)
    self_loop_index = mx.stack([loop_nodes, loop_nodes])
    full_edge_index = mx.concatenate([edge_index, self_loop_index], 1)

    if edge_features is not None:
        # add self loops to features
        self_loop_features = mx.full(
            (self_loop_index.shape[1], *edge_features.shape[1:]),
            fill_value,
            dtype=edge_features.dtype,
        ).astype(edge_features.dtype)
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
    edges = _to_numpy(edge_index)
    keep = np.nonzero(edges[0] != edges[1])[0]
    if keep.size == edges.shape[1]:  # nothing to remove
        if edge_features is not None:
            return edge_index, edge_features
        return edge_index

    no_self_loop_index = _take_edges(edge_index, keep)
    if edge_features is not None:
        if keep.size == 0:
            return no_self_loop_index, mx.zeros(
                (0, *edge_features.shape[1:]), dtype=edge_features.dtype
            )
        return no_self_loop_index, edge_features[mx.array(keep)]
    return no_self_loop_index


@overload
def to_undirected(
    edge_index: mx.array,
    edge_features=None,
    coalesce: bool = False,
    num_nodes: Optional[int] = None,
    reduce: str = "add",
) -> mx.array:
    ...


@overload
def to_undirected(
    edge_index: mx.array,
    edge_features: mx.array,
    coalesce: bool = False,
    num_nodes: Optional[int] = None,
    reduce: str = "add",
) -> tuple[mx.array, mx.array]:
    ...


@validate_edge_index_and_features
def to_undirected(
    edge_index: mx.array,
    edge_features: Optional[mx.array] = None,
    coalesce: bool = False,
    num_nodes: Optional[int] = None,
    reduce: str = "add",
) -> Union[mx.array, tuple[mx.array, mx.array]]:
    """
    Converts a graph given as `edge_index` and, optionally, `edge_features`
    to an undirected one, by adding the reverse of every edge.

    Args:
        edge_index: The edge index of the graph.
        edge_features: Edge features associated with each edge.
        coalesce: If ``True``, duplicate edges are merged afterwards (see
            :func:`coalesce`), so an edge that already had its reverse is kept
            once per direction. This is the behavior of PyG's
            ``to_undirected``. Default: ``False`` (plain concatenation).
        num_nodes: Number of nodes, only used when ``coalesce=True``.
        reduce: How the features of merged edges are combined when
            ``coalesce=True``. Default: ``"add"``.

    Returns:
        The undirected `edge_index` (and `edge_features`)
    """

    src, dst = edge_index
    undirected_edge_index = mx.concatenate(
        [mx.stack([src, dst]), mx.stack([dst, src])], 1
    )
    undirected_edge_features = None
    if edge_features is not None:
        undirected_edge_features = mx.concatenate([edge_features, edge_features], 0)

    if coalesce:
        return _coalesce(
            undirected_edge_index, undirected_edge_features, num_nodes, reduce
        )
    if undirected_edge_features is not None:
        return undirected_edge_index, undirected_edge_features
    return undirected_edge_index


@validate_edge_index
def remove_duplicate_directed_edges(edge_index: mx.array) -> mx.array:
    """
    Removes the duplicate directed edges in `edge_index`.

    Args:
        edge_index: The `edge_index` on which remove duplicate edges.

    Returns:
        An `edge_index` without duplicates, sorted by source then destination.
    """
    return coalesce(edge_index)


@overload
def coalesce(
    edge_index: mx.array,
    edge_features=None,
    num_nodes: Optional[int] = None,
    reduce: str = "add",
) -> mx.array:
    ...


@overload
def coalesce(
    edge_index: mx.array,
    edge_features: mx.array,
    num_nodes: Optional[int] = None,
    reduce: str = "add",
) -> tuple[mx.array, mx.array]:
    ...


@validate_edge_index_and_features
def coalesce(
    edge_index: mx.array,
    edge_features: Optional[mx.array] = None,
    num_nodes: Optional[int] = None,
    reduce: str = "add",
) -> Union[mx.array, tuple[mx.array, mx.array]]:
    """
    Sorts ``edge_index`` by source then destination and merges duplicate edges.

    The features of merged edges are combined with ``reduce``.

    Args:
        edge_index: Edges to coalesce.
        edge_features: Optional features of each edge, of shape
            ``[num_edges, ...]``.
        num_nodes: Number of nodes. Inferred from ``edge_index`` if not given.
        reduce: How the features of duplicate edges are combined: ``"add"``,
            ``"mean"``, ``"max"`` or ``"min"``. Default: ``"add"``.

    Returns:
        The sorted edges without duplicates, and their features if
        ``edge_features`` was given.

    Example:

    .. code-block:: python

        edge_index = mx.array([[1, 0, 1, 0], [0, 1, 0, 2]])
        edge_features = mx.array([1.0, 2.0, 3.0, 4.0])

        coalesce(edge_index)
        # mx.array([[0, 0, 1], [1, 2, 0]])

        coalesce(edge_index, edge_features)
        # (mx.array([[0, 0, 1], [1, 2, 0]]), mx.array([2.0, 4.0, 4.0]))
    """
    return _coalesce(edge_index, edge_features, num_nodes, reduce)


def _coalesce(
    edge_index: mx.array,
    edge_features: Optional[mx.array],
    num_nodes: Optional[int],
    reduce: str,
) -> Union[mx.array, tuple[mx.array, mx.array]]:
    if edge_index.shape[1] == 0:
        return edge_index if edge_features is None else (edge_index, edge_features)

    edges = _to_numpy(edge_index)
    keys = _edge_keys(edges, _infer_num_nodes(edge_index, num_nodes))

    if edge_features is None:
        if keys.size < 2 or bool(np.all(keys[1:] > keys[:-1])):
            return edge_index  # already sorted, no duplicates
        _, first = np.unique(keys, return_index=True)
        return _take_edges(edge_index, first)

    unique_keys, first, inverse = np.unique(
        keys, return_index=True, return_inverse=True
    )
    coalesced_index = _take_edges(edge_index, first)
    if unique_keys.size == keys.size:  # no duplicates: this is only a sort
        return coalesced_index, edge_features[mx.array(first)]
    return coalesced_index, scatter(
        edge_features, mx.array(inverse), unique_keys.size, aggr=reduce
    )


@validate_edge_index
def get_isolated_nodes_mask(
    edge_index: mx.array, num_nodes: int, complement: bool = True
) -> mx.array:
    """Returns the indices of non-isolated nodes, or of isolated nodes.

    A node is isolated if it has no edge to another node (self-loops are
    ignored).

    Args:
        edge_index: Edge index from which to remove isolated nodes.
        num_nodes: Number of nodes of the graph.
        complement: If ``True`` (default), returns the indices of the nodes
            that are not isolated. If ``False``, returns the indices of the
            isolated nodes.

    Returns:
        A sorted array of node indices.

    Example:

    .. code-block:: python

        edge_index = mx.array([[0, 2, 0], [2, 0, 0]])
        mask = get_isolated_nodes_mask(edge_index, 3)
        >>> mx.array([0,2])
        mask = get_isolated_nodes_mask(edge_index, 3, complement=False)
        >>> mx.array([1])
    """
    connected = _connected_nodes(edge_index, num_nodes)
    return mx.array(np.nonzero(connected if complement else ~connected)[0])


def _connected_nodes(edge_index: mx.array, num_nodes: int) -> np.ndarray:
    """Boolean host mask of the nodes with at least one non-self-loop edge."""
    edges = _to_numpy(edge_index)
    edges = edges[:, edges[0] != edges[1]]
    size = max(int(num_nodes), int(edges.max(initial=-1)) + 1)
    connected = np.zeros(size, dtype=bool)
    connected[edges.reshape(-1)] = True
    return connected


@validate_edge_index
def has_isolated_nodes(edge_index: mx.array, num_nodes: int) -> bool:
    """Function to check for isolated nodes.
    (i.e. nodes that don't have a link to any other nodes )

    Args:
        edge_index: Edge index on which to check for isolated nodes.
        num_nodes: Number of nodes of the graph.

    Returns:
        A boolean of whether the graph has isolated nodes.

    Example:

    .. code-block:: python

        edge_index = mx.array([[0, 2, 0], [2, 0, 0]])
        has_isolated_nodes(edge_index, 3)
        >>> True

    """
    return bool(_connected_nodes(edge_index, num_nodes).sum() < num_nodes)


@validate_edge_index
def has_self_loops(edge_index: mx.array) -> bool:
    """Function to check for self loops.

    Args:
        edge_index: Edge index on which to check for self loops.

    Returns:
        A boolean of whether the graph has self loops.

    Example:

    .. code-block:: python

        edge_index = mx.array([[0, 2, 0], [2, 0, 0]])
        has_self_loops(edge_index)
        >>> True
    """

    return ((edge_index[0] == edge_index[1]).sum() > 0).item()
