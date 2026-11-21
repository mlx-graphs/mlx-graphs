from typing import Literal, Optional, Union

import mlx.core as mx
import numpy as np

from mlx_graphs.utils.negative_sampling import negative_sampling
from mlx_graphs.utils.transformations import _empty_like, _to_numpy
from mlx_graphs.utils.validators import validate_edge_index


def shuffle_node(
    node_features: mx.array,
    batch_indices: Optional[mx.array] = None,
    training: bool = True,
) -> tuple[mx.array, mx.array]:
    """
    Randomly shuffles the rows of ``node_features``, within each graph if
    ``batch_indices`` is given.

    Randomness comes from ``mx.random``, so results are reproducible with
    ``mx.random.seed``.

    Args:
        node_features: The node features, of shape ``[num_nodes, ...]``.
        batch_indices: Optional graph of each node, of shape ``[num_nodes]``,
            with the nodes of each graph stored contiguously (as in
            :class:`~mlx_graphs.data.GraphDataBatch`). Default: ``None``
        training: If ``False``, nothing is shuffled. Default: ``True``

    Returns:
        A tuple ``(node_features[perm], perm)`` with the shuffled features and
        the permutation.

    Example:

    .. code-block:: python

        node_features = mx.arange(6).reshape(6, 1)
        batch_indices = mx.array([0, 0, 0, 1, 1, 1])

        shuffle_node(node_features, batch_indices)
        # e.g. (mx.array([[1], [2], [0], [5], [3], [4]]),
        #       mx.array([1, 2, 0, 5, 3, 4]))
    """
    num_nodes = node_features.shape[0]
    if not training:
        return node_features, mx.arange(num_nodes)
    if batch_indices is None:
        perm = mx.random.permutation(num_nodes)
        return node_features[perm], perm

    batch = _to_numpy(batch_indices).reshape(-1)
    keys = _to_numpy(mx.random.uniform(shape=(num_nodes,)))
    # Sorting by (graph, random key) shuffles the nodes within each graph.
    perm = mx.array(np.lexsort((keys, batch)).astype(np.int32))
    return node_features[perm], perm


def mask_feature(
    node_features: mx.array,
    p: float = 0.5,
    mode: Literal["row", "col", "all"] = "col",
    fill_value: float = 0.0,
    training: bool = True,
) -> tuple[mx.array, mx.array]:
    """
    Randomly masks the features of ``node_features`` with probability ``p``,
    replacing them with ``fill_value``.

    Randomness comes from ``mx.random``, so results are reproducible with
    ``mx.random.seed``.

    Args:
        node_features: The features, of shape ``[num_nodes, num_features]``.
        p: Masking probability. Default: ``0.5``
        mode: ``"col"`` masks whole feature columns (the same features for
            all nodes), ``"row"`` masks all the features of whole nodes and
            ``"all"`` masks each entry independently. Default: ``"col"``
        fill_value: Value of the masked entries. Default: ``0.0``
        training: If ``False``, nothing is masked. Default: ``True``

    Returns:
        A tuple ``(node_features, feature_mask)`` with the masked features and
        the boolean mask of the kept entries, of shape ``[1, num_features]``
        (``"col"``), ``[num_nodes, 1]`` (``"row"``) or the shape of
        ``node_features`` (``"all"``).

    Example:

    .. code-block:: python

        node_features = mx.ones((3, 4))
        node_features, feature_mask = mask_feature(node_features, p=0.5)
        # node_features: e.g. mx.array([[1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0]])
        # feature_mask: e.g. mx.array([[True, False, True, False]])
    """
    if p < 0.0 or p > 1.0:
        raise ValueError(f"Masking ratio has to be between 0 and 1 (got {p})")
    if not training or p == 0.0:
        return node_features, mx.ones(node_features.shape, dtype=mx.bool_)
    if mode == "row":
        mask = mx.random.uniform(shape=(node_features.shape[0], 1)) >= p
    elif mode == "col":
        mask = mx.random.uniform(shape=(1, node_features.shape[1])) >= p
    elif mode == "all":
        mask = mx.random.uniform(shape=node_features.shape) >= p
    else:
        raise ValueError(f"Unknown mode '{mode}', expected 'row', 'col' or 'all'")
    fill = mx.array(fill_value, dtype=node_features.dtype)
    return mx.where(mask, node_features, fill), mask


@validate_edge_index
def add_random_edge(
    edge_index: mx.array,
    p: float = 0.5,
    force_undirected: bool = False,
    num_nodes: Optional[Union[int, tuple[int, int]]] = None,
    training: bool = True,
) -> tuple[mx.array, mx.array]:
    """
    Randomly adds ``round(p * num_edges)`` new edges to ``edge_index``. The
    new edges are sampled with :func:`~mlx_graphs.utils.negative_sampling`:
    they are all different, are not self-loops and are not already in the
    graph.

    Args:
        edge_index: a [2, num_edges] array representing the source and target
            nodes of each edge
        p: Ratio of edges to add, relative to the number of edges.
            Default: ``0.5``
        force_undirected: If ``True``, the added edges are undirected (both
            directions are added and counted). Default: ``False``
        num_nodes: Number of nodes in the graph, or a tuple ``(num_src_nodes,
            num_dst_nodes)`` for a bipartite graph. Inferred from
            ``edge_index`` if not given.
        training: If ``False``, nothing is added. Default: ``True``

    Returns:
        A tuple ``(edge_index, added_edges)`` with the edges of the graph
        followed by the added edges, and the added edges alone.

    Example:

    .. code-block:: python

        edge_index = mx.array([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
        edge_index, added_edges = add_random_edge(edge_index, p=0.5)
        # edge_index: e.g. mx.array([[0, 1, 1, 2, 2, 3, 2, 1, 3],
        #                            [1, 0, 2, 1, 3, 2, 0, 3, 0]])
        # added_edges: e.g. mx.array([[2, 1, 3], [0, 3, 0]])
    """
    if p < 0.0 or p > 1.0:
        raise ValueError(f"Ratio of added edges has to be between 0 and 1 (got {p})")
    if force_undirected and isinstance(num_nodes, (tuple, list)):
        raise ValueError("`force_undirected` is not supported for bipartite graphs")
    if not training or p == 0.0:
        return edge_index, _empty_like(edge_index)

    added = negative_sampling(
        edge_index,
        num_nodes=num_nodes,
        num_neg_samples=round(edge_index.shape[1] * p),
        force_undirected=force_undirected,
    )
    return mx.concatenate([edge_index, added], axis=1), added
