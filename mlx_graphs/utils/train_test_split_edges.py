import copy
import math
from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np

from mlx_graphs.utils.transformations import _to_numpy, to_undirected

if TYPE_CHECKING:
    from mlx_graphs.data import GraphData


def _permutation(size: int) -> np.ndarray:
    if size == 0:
        return np.zeros(0, dtype=np.int64)
    return np.asarray(mx.random.permutation(size)).astype(np.int64)


def train_test_split_edges(
    data: "GraphData",
    val_ratio: float = 0.05,
    test_ratio: float = 0.1,
) -> "GraphData":
    """
    Splits the edges of an undirected graph into positive and negative
    training, validation and test edges.

    Only the edges ``(i, j)`` with ``i < j`` are considered (one per
    undirected edge). ``floor(val_ratio * num)`` of them become validation
    edges, ``floor(test_ratio * num)`` test edges and the others training
    edges, which are returned in both directions. As many negative edges
    (pairs ``i < j`` that are not edges) as positive ones are sampled for the
    validation and test sets.

    The returned graph has the attributes ``train_pos_edge_index``,
    ``val_pos_edge_index``, ``test_pos_edge_index``, ``val_neg_edge_index``,
    ``test_neg_edge_index`` and ``train_neg_adj_mask`` (a boolean
    ``[num_nodes, num_nodes]`` upper-triangular mask of the pairs that can
    still be used as training negatives), plus ``*_pos_edge_features`` if the
    graph has ``edge_features``. Its ``edge_index`` and ``edge_features`` are
    set to ``None``. The input graph is not modified.

    Randomness comes from ``mx.random``, so results are reproducible with
    ``mx.random.seed``.

    .. note::

        Prefer :class:`~mlx_graphs.transforms.RandomLinkSplit`, which this
        function predates (it is deprecated in PyG).

    Args:
        data: The graph to split.
        val_ratio: Ratio of validation edges. Default: ``0.05``
        test_ratio: Ratio of test edges. Default: ``0.1``

    Returns:
        A copy of ``data`` with the splits.

    Example:

    .. code-block:: python

        graph = GraphData(
            edge_index=mx.array([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
            num_nodes=4,
        )
        graph = train_test_split_edges(graph, val_ratio=0.34, test_ratio=0.34)
        graph.val_pos_edge_index
        # e.g. mx.array([[1], [2]])
    """
    num_nodes = data.num_nodes
    data = copy.copy(data)
    edge_index = data.edge_index
    edge_features = data.edge_features
    data.num_nodes = num_nodes

    edges = _to_numpy(edge_index).astype(np.int64)
    upper = np.nonzero(edges[0] < edges[1])[0]
    row, col = edges[0][upper], edges[1][upper]

    n_v = int(math.floor(val_ratio * row.size))
    n_t = int(math.floor(test_ratio * row.size))

    perm = _permutation(row.size)
    row, col = row[perm], col[perm]
    dtype = edge_index.dtype
    features = None
    if edge_features is not None:
        features = edge_features[mx.array(upper[perm])] if upper.size else None

    def pos_edges(start: int, end: int) -> mx.array:
        return mx.array(np.stack([row[start:end], col[start:end]]), dtype=dtype)

    data.val_pos_edge_index = pos_edges(0, n_v)
    data.test_pos_edge_index = pos_edges(n_v, n_v + n_t)
    train_pos = pos_edges(n_v + n_t, row.size)
    if features is not None:
        data.val_pos_edge_features = features[:n_v]
        data.test_pos_edge_features = features[n_v : n_v + n_t]
        data.train_pos_edge_index, data.train_pos_edge_features = to_undirected(
            train_pos,
            features[n_v + n_t :],
            coalesce=True,
            num_nodes=num_nodes,
        )
    else:
        data.train_pos_edge_index = to_undirected(
            train_pos, coalesce=True, num_nodes=num_nodes
        )

    # Negative edges.
    neg_adj_mask = np.triu(np.ones((num_nodes, num_nodes), dtype=bool), k=1)
    neg_adj_mask[row, col] = False
    neg_row, neg_col = np.nonzero(neg_adj_mask)
    neg_perm = _permutation(neg_row.size)[: n_v + n_t]
    neg_row, neg_col = neg_row[neg_perm], neg_col[neg_perm]
    neg_adj_mask[neg_row, neg_col] = False
    data.train_neg_adj_mask = mx.array(neg_adj_mask)

    data.val_neg_edge_index = mx.array(
        np.stack([neg_row[:n_v], neg_col[:n_v]]), dtype=dtype
    )
    data.test_neg_edge_index = mx.array(
        np.stack([neg_row[n_v : n_v + n_t], neg_col[n_v : n_v + n_t]]), dtype=dtype
    )

    data.edge_index = None
    data.edge_features = None
    return data
