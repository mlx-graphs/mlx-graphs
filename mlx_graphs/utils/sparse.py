from typing import Any, Optional

import mlx.core as mx
import numpy as np

from mlx_graphs.utils.transformations import _infer_num_nodes, _to_numpy
from mlx_graphs.utils.validators import validate_edge_index_and_features


def _to_mlx(array: np.ndarray) -> mx.array:
    """``mx.array`` of a host array, with float64 narrowed to float32."""
    if array.dtype == np.float64:
        array = array.astype(np.float32)
    return mx.array(array)


@validate_edge_index_and_features
def to_scipy_sparse_matrix(
    edge_index: mx.array,
    edge_features: Optional[mx.array] = None,
    num_nodes: Optional[int] = None,
) -> Any:
    """
    Converts a graph given by ``edge_index`` (and optional scalar edge
    features) to a ``scipy.sparse.coo_matrix``. Requires ``scipy``.

    Args:
        edge_index: a [2, num_edges] array representing the source and target
            nodes of each edge
        edge_features: Optional scalar weight of each edge, of shape
            ``[num_edges]`` or ``[num_edges, 1]``. Default: ``None`` (all
            ones)
        num_nodes: Number of nodes in the graph. Inferred from ``edge_index``
            if not given.

    Returns:
        A ``scipy.sparse.coo_matrix`` of shape ``[num_nodes, num_nodes]``.

    Example:

    .. code-block:: python

        edge_index = mx.array([[0, 1, 1], [1, 0, 2]])
        to_scipy_sparse_matrix(edge_index).toarray()
        # array([[0., 1., 0.],
        #        [1., 0., 1.],
        #        [0., 0., 0.]], dtype=float32)
    """
    import scipy.sparse as sp

    edges = _to_numpy(edge_index)
    if edge_features is None:
        values = np.ones(edges.shape[1], dtype=np.float32)
    else:
        values = np.asarray(edge_features).reshape(-1)
        if values.size != edges.shape[1]:
            raise ValueError(
                "`edge_features` must hold one scalar per edge "
                f"(got shape {edge_features.shape})"
            )
    n = _infer_num_nodes(edge_index, num_nodes)
    return sp.coo_matrix((values, (edges[0], edges[1])), shape=(n, n))


def from_scipy_sparse_matrix(matrix: Any) -> tuple[mx.array, mx.array]:
    """
    Converts a ``scipy.sparse`` matrix (or array) to an ``edge_index`` and
    edge weights.

    Args:
        matrix: A ``scipy.sparse`` matrix, in any format.

    Returns:
        A tuple ``(edge_index, edge_weights)`` with an int32 ``[2, nnz]``
        ``edge_index`` and the stored values (float64 values are converted to
        float32). Explicitly stored zeros are kept.

    Example:

    .. code-block:: python

        import scipy.sparse as sp

        matrix = sp.coo_matrix(([1.0, 2.0], ([0, 1], [1, 0])), shape=(2, 2))
        from_scipy_sparse_matrix(matrix)
        # (mx.array([[0, 1], [1, 0]]), mx.array([1.0, 2.0]))
    """
    coo = matrix.tocoo()
    edge_index = mx.array(np.stack([coo.row, coo.col]).astype(np.int32))
    return edge_index, _to_mlx(np.asarray(coo.data))


def dense_to_sparse(
    adj: mx.array, mask: Optional[mx.array] = None
) -> tuple[mx.array, mx.array]:
    """
    Converts a dense adjacency matrix, or a batch of them, to an
    ``edge_index`` and edge weights (the non-zero entries).

    For a batch of adjacency matrices of shape ``[batch_size, num_nodes,
    num_nodes]``, the graphs are stacked as one block-diagonal graph: the
    nodes of graph ``b`` are offset by ``b * num_nodes``, or, if ``mask`` is
    given, by the number of valid nodes of the previous graphs (only the rows
    of the valid nodes are read).

    Args:
        adj: Dense adjacency of shape ``[num_nodes, num_nodes]`` or
            ``[batch_size, num_nodes, num_nodes]``.
        mask: Optional boolean mask of the valid nodes of each graph, of shape
            ``[batch_size, num_nodes]`` (e.g. from
            :func:`~mlx_graphs.utils.to_dense_batch`). Only used for batched
            adjacencies. Default: ``None``

    Returns:
        A tuple ``(edge_index, edge_weights)`` with an int32 ``edge_index``.

    Example:

    .. code-block:: python

        adj = mx.array([[0, 1, 0], [2, 0, 0], [0, 0, 3]])
        dense_to_sparse(adj)
        # (mx.array([[0, 1, 2], [1, 0, 2]]), mx.array([1, 2, 3]))
    """
    if adj.ndim < 2 or adj.ndim > 3:
        raise ValueError(
            "Dense adjacency matrix `adj` must be two- or three-dimensional "
            f"(got {adj.ndim} dimensions)"
        )
    if mask is not None and adj.ndim == 2:
        mask = None  # ignored, as in PyG
    if mask is not None and mask.ndim != 2:
        raise ValueError(f"Mask must be two-dimensional (got {mask.ndim} dimensions)")
    if mask is not None and adj.shape[-2] != adj.shape[-1]:
        raise ValueError(
            "Mask is only supported on square adjacency matrices "
            f"(got {list(adj.shape)})"
        )

    adj_np = np.asarray(adj)
    if adj.ndim == 2:
        row, col = np.nonzero(adj_np)
        edge_weights = adj[mx.array(row), mx.array(col)] if row.size else adj[0, :0]
        return mx.array(np.stack([row, col]).astype(np.int32)), edge_weights

    batch_size, num_rows, num_cols = adj.shape
    flat = adj.reshape(-1, num_cols)
    if mask is None:
        offset = np.repeat(np.arange(batch_size) * num_cols, num_rows)
        rows = np.arange(batch_size * num_rows)
    else:
        mask_np = np.asarray(mask).astype(bool).reshape(-1)
        rows = np.nonzero(mask_np)[0]
        count = np.asarray(mask).astype(bool).sum(axis=-1)
        offset = np.repeat(np.concatenate([[0], np.cumsum(count)[:-1]]), count)
    row, col = np.nonzero(adj_np.reshape(-1, num_cols)[rows])
    if row.size == 0:
        return mx.zeros((2, 0), dtype=mx.int32), flat[0, :0]
    edge_weights = flat[mx.array(rows[row]), mx.array(col)]
    col = col + offset[row]
    return mx.array(np.stack([row, col]).astype(np.int32)), edge_weights
