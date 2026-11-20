from typing import Optional

import mlx.core as mx
import numpy as np

from mlx_graphs.utils.transformations import _to_numpy
from mlx_graphs.utils.validators import validate_edge_index


def _batch_layout(
    batch_indices: Optional[mx.array], num_nodes: int
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Host layout of a batch of graphs.

    Returns the graph of each node, the position of each node within its
    graph, the number of graphs and the size of the largest graph.
    """
    if batch_indices is None:
        batch = np.zeros(num_nodes, dtype=np.int64)
    else:
        batch = _to_numpy(batch_indices).reshape(-1).astype(np.int64)
        if batch.size != num_nodes:
            raise ValueError(
                f"batch_indices must have one entry per node (got {batch.size} "
                f"entries for {num_nodes} nodes)"
            )
    batch_size = int(batch.max()) + 1 if batch.size > 0 else 1
    counts = np.bincount(batch, minlength=batch_size)
    starts = np.cumsum(counts) - counts

    # Position of each node among the nodes of its graph. The stable sort makes
    # this work when the nodes of a graph are not contiguous.
    order = np.argsort(batch, kind="stable")
    local = np.empty(batch.size, dtype=np.int64)
    local[order] = np.arange(batch.size, dtype=np.int64) - starts[batch[order]]
    return batch, local, batch_size, int(counts.max(initial=0))


def to_dense_batch(
    node_features: mx.array,
    batch_indices: Optional[mx.array] = None,
    fill_value: float = 0.0,
    max_num_nodes: Optional[int] = None,
) -> tuple[mx.array, mx.array]:
    """
    Converts the node features of a batch of graphs to a dense array of shape
    ``[batch_size, max_num_nodes, num_features]``, padded with ``fill_value``.

    Args:
        node_features: Array of shape ``[num_nodes, ...]`` with the features of
            the nodes of all the graphs.
        batch_indices: Array of shape ``[num_nodes]`` with the graph of each
            node (e.g. ``GraphDataBatch.batch_indices``). Default: ``None``
            (all the nodes belong to a single graph).
        fill_value: Value of the padded entries. Default: ``0.0``
        max_num_nodes: Number of nodes per graph in the output. Graphs with
            more nodes are truncated. Default: ``None`` (the size of the
            largest graph).

    Returns:
        A tuple ``(dense, mask)`` with the dense features of shape
        ``[batch_size, max_num_nodes, ...]`` and a boolean mask of shape
        ``[batch_size, max_num_nodes]`` which is ``True`` for real nodes and
        ``False`` for padding.

    Example:

    .. code-block:: python

        node_features = mx.array([[1.0], [2.0], [3.0]])
        batch_indices = mx.array([0, 0, 1])

        dense, mask = to_dense_batch(node_features, batch_indices)
        # dense: mx.array([[[1.0], [2.0]], [[3.0], [0.0]]])
        # mask: mx.array([[True, True], [True, False]])
    """
    num_nodes = node_features.shape[0]
    batch, local, batch_size, largest = _batch_layout(batch_indices, num_nodes)
    if max_num_nodes is None:
        max_num_nodes = largest

    feature_shape = node_features.shape[1:]
    # `mx.full` ignores an integer dtype for a float fill value.
    dense = mx.full(
        (batch_size * max_num_nodes, *feature_shape),
        fill_value,
        dtype=node_features.dtype,
    ).astype(node_features.dtype)
    mask = np.zeros(batch_size * max_num_nodes, dtype=bool)

    keep = np.nonzero(local < max_num_nodes)[0]
    if keep.size > 0:
        flat = batch[keep] * max_num_nodes + local[keep]
        if keep.size < num_nodes:
            node_features = node_features[mx.array(keep)]
        dense[mx.array(flat)] = node_features
        mask[flat] = True

    return (
        dense.reshape(batch_size, max_num_nodes, *feature_shape),
        mx.array(mask.reshape(batch_size, max_num_nodes)),
    )


@validate_edge_index
def to_dense_adj(
    edge_index: mx.array,
    batch_indices: Optional[mx.array] = None,
    edge_features: Optional[mx.array] = None,
    max_num_nodes: Optional[int] = None,
) -> mx.array:
    """
    Converts the edges of a batch of graphs to dense adjacency matrices of
    shape ``[batch_size, max_num_nodes, max_num_nodes]``.

    Entry ``[b, i, j]`` holds the edge from the ``i``-th to the ``j``-th node
    of graph ``b``. Duplicate edges are summed.

    Args:
        edge_index: a [2, num_edges] array representing the source and target
            nodes of each edge
        batch_indices: Array of shape ``[num_nodes]`` with the graph of each
            node (e.g. ``GraphDataBatch.batch_indices``). Default: ``None``
            (a single graph, whose size is inferred from ``edge_index``).
        edge_features: Optional features of each edge, of shape
            ``[num_edges, ...]``, used as the values of the adjacency
            matrices. Default: ``None`` (ones).
        max_num_nodes: Number of nodes per graph in the output. Edges of nodes
            beyond that size are dropped. Default: ``None`` (the size of the
            largest graph).

    Returns:
        The dense adjacency matrices, of shape
        ``[batch_size, max_num_nodes, max_num_nodes, ...]`` (the trailing
        dimensions are those of ``edge_features``). The dtype is the one of
        ``edge_features``, or float32 if they are not given.

    Example:

    .. code-block:: python

        edge_index = mx.array([[0, 0, 1, 2, 3], [0, 1, 0, 3, 2]])
        batch_indices = mx.array([0, 0, 1, 1])

        to_dense_adj(edge_index, batch_indices)
        # mx.array([[[1, 1], [1, 0]], [[0, 1], [1, 0]]])
    """
    num_edges = edge_index.shape[1]
    if edge_features is not None and edge_features.shape[0] != num_edges:
        raise ValueError(
            "edge_features must be 1 per edge "
            f"(got {num_edges} edges and {edge_features.shape[0]} features)"
        )
    edges = _to_numpy(edge_index).astype(np.int64)
    max_node = int(edges.max()) if edges.size > 0 else -1
    if batch_indices is None:
        num_nodes = max_node + 1
    else:
        num_nodes = batch_indices.size
        if max_node >= num_nodes:
            raise ValueError(
                f"batch_indices has {num_nodes} entries but `edge_index` "
                f"contains node {max_node}"
            )
    batch, local, batch_size, largest = _batch_layout(batch_indices, num_nodes)
    if max_num_nodes is None:
        max_num_nodes = largest

    graph, row, col = batch[edges[0]], local[edges[0]], local[edges[1]]
    keep = np.nonzero((row < max_num_nodes) & (col < max_num_nodes))[0]

    if edge_features is None:
        feature_shape: tuple = ()
        dtype = mx.float32
    else:
        feature_shape = tuple(edge_features.shape[1:])
        dtype = edge_features.dtype
    adj = mx.zeros((batch_size * max_num_nodes * max_num_nodes, *feature_shape), dtype)

    if keep.size > 0:
        flat = (graph[keep] * max_num_nodes + row[keep]) * max_num_nodes + col[keep]
        if edge_features is None:
            values = mx.ones((keep.size,), dtype=dtype)
        elif keep.size < num_edges:
            values = edge_features[mx.array(keep)]
        else:
            values = edge_features
        adj = adj.at[mx.array(flat)].add(values)

    return adj.reshape(batch_size, max_num_nodes, max_num_nodes, *feature_shape)
