from typing import Optional

import mlx.core as mx
import numpy as np

from mlx_graphs.utils.transformations import _to_numpy
from mlx_graphs.utils.validators import validate_edge_index


def _graph_sizes(batch_indices: mx.array, batch_size: Optional[int]) -> np.ndarray:
    batch = _to_numpy(batch_indices).astype(np.int64).reshape(-1)
    minlength = 0 if batch_size is None else int(batch_size)
    return np.bincount(batch, minlength=minlength)


def _split(array: mx.array, sizes: np.ndarray, axis: int) -> list[mx.array]:
    if sizes.size == 0:
        return []
    return mx.split(array, np.cumsum(sizes)[:-1].tolist(), axis=axis)


def unbatch(
    src: mx.array,
    batch_indices: mx.array,
    axis: int = 0,
    batch_size: Optional[int] = None,
) -> list[mx.array]:
    """
    Splits ``src`` into one array per graph, according to ``batch_indices``.

    This is the array-level counterpart of
    :func:`mlx_graphs.data.batch.unbatch`, which splits a whole
    :class:`~mlx_graphs.data.GraphDataBatch`.

    Args:
        src: The array to split, e.g. node features of shape
            ``[num_nodes, ...]``.
        batch_indices: The graph of each entry of ``src`` along ``axis``, of
            shape ``[src.shape[axis]]``, with the entries of each graph stored
            contiguously and in graph order.
        axis: The axis along which to split. Default: ``0``
        batch_size: Number of graphs. Inferred from ``batch_indices`` if not
            given (graphs after the last one in ``batch_indices`` are then
            missing).

    Returns:
        A list with one array per graph.

    Example:

    .. code-block:: python

        src = mx.arange(7)
        batch_indices = mx.array([0, 0, 0, 1, 1, 2, 2])
        unbatch(src, batch_indices)
        # [mx.array([0, 1, 2]), mx.array([3, 4]), mx.array([5, 6])]
    """
    return _split(src, _graph_sizes(batch_indices, batch_size), axis)


@validate_edge_index
def unbatch_edge_index(
    edge_index: mx.array,
    batch_indices: mx.array,
    batch_size: Optional[int] = None,
) -> list[mx.array]:
    """
    Splits the ``edge_index`` of a batch of graphs into one ``edge_index`` per
    graph, with the node ids of each graph starting at 0.

    Args:
        edge_index: a [2, num_edges] array representing the source and target
            nodes of each edge of the batch, with the edges of each graph
            stored contiguously and in graph order.
        batch_indices: The graph of each node, of shape ``[num_nodes]``, with
            the nodes of each graph stored contiguously and in graph order.
        batch_size: Number of graphs. Inferred from ``batch_indices`` if not
            given.

    Returns:
        A list with the ``edge_index`` of each graph.

    Example:

    .. code-block:: python

        edge_index = mx.array([[0, 1, 1, 2, 4, 5, 5, 6], [1, 0, 2, 1, 5, 4, 6, 5]])
        batch_indices = mx.array([0, 0, 0, 0, 1, 1, 1])
        unbatch_edge_index(edge_index, batch_indices)
        # [mx.array([[0, 1, 1, 2], [1, 0, 2, 1]]),
        #  mx.array([[0, 1, 1, 2], [1, 0, 2, 1]])]
    """
    num_nodes = _graph_sizes(batch_indices, batch_size)
    ptr = np.concatenate([[0], np.cumsum(num_nodes)[:-1]])
    batch = _to_numpy(batch_indices).astype(np.int64).reshape(-1)
    edges = _to_numpy(edge_index).astype(np.int64)
    edge_batch = batch[edges[0]]
    local = mx.array(edges - ptr[edge_batch], dtype=edge_index.dtype)
    sizes = np.bincount(edge_batch, minlength=num_nodes.size)
    return _split(local, sizes, axis=1)
