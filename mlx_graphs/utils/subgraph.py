from typing import Optional, Sequence, Union

import mlx.core as mx
import numpy as np

from mlx_graphs.utils.transformations import (
    _infer_num_nodes,
    _take_edges,
    _to_numpy,
)
from mlx_graphs.utils.validators import validate_edge_index

NodeSubset = Union[mx.array, np.ndarray, Sequence[int], Sequence[bool]]


def _take_rows(array: mx.array, keep: np.ndarray) -> mx.array:
    """Rows ``keep`` (host indices) of ``array``; ``[0, ...]`` if empty."""
    if keep.size == 0:
        return mx.zeros((0, *array.shape[1:]), dtype=array.dtype)
    return array[mx.array(keep)]


def _subset_to_mask(
    subset: NodeSubset, edge_index: mx.array, num_nodes: Optional[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Returns the boolean host mask of ``subset`` and the new id of each node.

    The new id of a node is its position in ``subset`` (its rank among the
    selected nodes for a boolean mask), and -1 for nodes outside ``subset``.
    """
    subset_np = _to_numpy(subset) if isinstance(subset, mx.array) else subset
    subset_np = np.asarray(subset_np).reshape(-1)
    max_node = int(mx.max(edge_index).item()) if edge_index.size > 0 else -1

    if subset_np.dtype == np.bool_:
        mask = subset_np
        if num_nodes is not None and num_nodes != mask.size:
            raise ValueError(
                f"The boolean `subset` has {mask.size} entries but "
                f"num_nodes={num_nodes}"
            )
        if max_node >= mask.size:
            raise ValueError(
                f"The boolean `subset` has {mask.size} entries but `edge_index` "
                f"contains node {max_node}"
            )
        mapping = np.full(mask.size, -1, dtype=np.int64)
        mapping[mask] = np.arange(int(mask.sum()))
        return mask, mapping

    index = subset_np.astype(np.int64)
    size = max(_infer_num_nodes(edge_index, num_nodes), max_node + 1)
    if index.size > 0:
        if index.min() < 0:
            raise ValueError("`subset` must contain non-negative node indices")
        size = max(size, int(index.max()) + 1)
    mask = np.zeros(size, dtype=bool)
    mask[index] = True
    mapping = np.full(size, -1, dtype=np.int64)
    # Reversed so that the first occurrence of a repeated node wins.
    mapping[index[::-1]] = np.arange(index.size)[::-1]
    return mask, mapping


@validate_edge_index
def _subgraph(
    edge_index: mx.array,
    subset: NodeSubset,
    edge_features: Optional[mx.array],
    relabel_nodes: bool,
    num_nodes: Optional[int],
) -> tuple[mx.array, Optional[mx.array], np.ndarray]:
    """Implementation of :func:`subgraph`, returning the edge mask on the host."""
    if edge_features is not None and edge_features.shape[0] != edge_index.shape[1]:
        raise ValueError(
            "edge_features must be 1 per edge "
            f"(got {edge_index.shape[1]} edges and {edge_features.shape[0]} features)"
        )
    mask, mapping = _subset_to_mask(subset, edge_index, num_nodes)
    edges = _to_numpy(edge_index)
    edge_mask = mask[edges[0]] & mask[edges[1]]
    keep = np.nonzero(edge_mask)[0]

    if relabel_nodes:
        sub_edge_index = mx.array(mapping[edges[:, keep]], dtype=edge_index.dtype)
    else:
        sub_edge_index = _take_edges(edge_index, keep)

    if edge_features is not None:
        edge_features = _take_rows(edge_features, keep)
    return sub_edge_index, edge_features, edge_mask


def subgraph(
    subset: NodeSubset,
    edge_index: mx.array,
    edge_features: Optional[mx.array] = None,
    relabel_nodes: bool = False,
    num_nodes: Optional[int] = None,
    return_edge_mask: bool = False,
) -> Union[
    tuple[mx.array, Optional[mx.array]],
    tuple[mx.array, Optional[mx.array], mx.array],
]:
    """
    Returns the subgraph induced by the nodes in ``subset``, i.e. the edges of
    ``edge_index`` whose source and destination are both in ``subset``.

    Args:
        subset: The nodes to keep, either as an array (or list) of node
            indices or as a boolean mask of shape ``[num_nodes]``.
        edge_index: a [2, num_edges] array representing the source and target
            nodes of each edge
        edge_features: Optional features of each edge, of shape
            ``[num_edges, ...]``. Default: ``None``
        relabel_nodes: If ``True``, the returned ``edge_index`` uses new node
            ids in ``[0, len(subset))``: the position of each node in
            ``subset`` (its rank among the selected nodes for a boolean mask).
            Default: ``False``
        num_nodes: Number of nodes in the graph. Inferred from ``edge_index``
            (or from the length of a boolean ``subset``) if not given.
        return_edge_mask: If ``True``, also returns the boolean mask of shape
            ``[num_edges]`` of the edges that were kept. Default: ``False``

    Returns:
        A tuple ``(edge_index, edge_features)`` with the edges of the subgraph
        and their features (``None`` if ``edge_features`` was not given),
        followed by the edge mask if ``return_edge_mask`` is ``True``.

    Example:

    .. code-block:: python

        edge_index = mx.array([[0, 1, 2, 3], [1, 2, 3, 0]])

        subgraph(mx.array([1, 2, 3]), edge_index)
        # (mx.array([[1, 2], [2, 3]]), None)

        subset = mx.array([False, True, True, True])
        subgraph(subset, edge_index, relabel_nodes=True)
        # (mx.array([[0, 1], [1, 2]]), None)
    """
    sub_edge_index, sub_edge_features, edge_mask = _subgraph(
        edge_index, subset, edge_features, relabel_nodes, num_nodes
    )
    if return_edge_mask:
        return sub_edge_index, sub_edge_features, mx.array(edge_mask)
    return sub_edge_index, sub_edge_features


@validate_edge_index
def _k_hop_subgraph(
    edge_index: mx.array,
    node_idx: Union[int, Sequence[int], mx.array],
    num_hops: int,
    relabel_nodes: bool,
    num_nodes: Optional[int],
    directed: bool,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    edges = _to_numpy(edge_index)
    src, dst = edges[0], edges[1]

    targets = _to_numpy(node_idx) if isinstance(node_idx, mx.array) else node_idx
    targets = np.asarray(targets).reshape(-1).astype(np.int64)
    size = _infer_num_nodes(edge_index, num_nodes)
    if edges.size > 0:
        size = max(size, int(edges.max()) + 1)
    if targets.size > 0:
        size = max(size, int(targets.max()) + 1)

    node_mask = np.zeros(size, dtype=bool)
    preserved_edge_mask = np.zeros(src.size, dtype=bool)
    subsets = [targets]
    for _ in range(num_hops):
        node_mask[:] = False
        node_mask[subsets[-1]] = True
        hop_edge_mask = node_mask[dst]
        preserved_edge_mask |= hop_edge_mask
        subsets.append(src[hop_edge_mask].astype(np.int64))

    subset, inverse = np.unique(np.concatenate(subsets), return_inverse=True)
    mapping = inverse.reshape(-1)[: targets.size]

    node_mask[:] = False
    node_mask[subset] = True
    if directed:
        edge_mask = preserved_edge_mask
    else:
        edge_mask = node_mask[src] & node_mask[dst]
    keep = np.nonzero(edge_mask)[0]

    dtype = edge_index.dtype
    if relabel_nodes:
        new_ids = np.full(size, -1, dtype=np.int64)
        new_ids[subset] = np.arange(subset.size)
        sub_edge_index = mx.array(new_ids[edges[:, keep]], dtype=dtype)
    else:
        sub_edge_index = _take_edges(edge_index, keep)

    return (
        mx.array(subset, dtype=dtype),
        sub_edge_index,
        mx.array(mapping, dtype=dtype),
        mx.array(edge_mask),
    )


def k_hop_subgraph(
    node_idx: Union[int, Sequence[int], mx.array],
    num_hops: int,
    edge_index: mx.array,
    relabel_nodes: bool = False,
    num_nodes: Optional[int] = None,
    directed: bool = False,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """
    Computes the ``num_hops``-hop subgraph around the nodes ``node_idx``.

    Edges are followed backwards from the target nodes: the subgraph contains
    every node whose messages reach ``node_idx`` within ``num_hops`` message
    passing steps (i.e. the receptive field of ``node_idx`` for a GNN with
    ``num_hops`` layers).

    Args:
        node_idx: The central node(s), as an int, a list or an array.
        num_hops: Number of hops to follow.
        edge_index: a [2, num_edges] array representing the source and target
            nodes of each edge
        relabel_nodes: If ``True``, the returned ``edge_index`` uses new node
            ids in ``[0, len(subset))``, following the order of the returned
            ``subset``. Default: ``False``
        num_nodes: Number of nodes in the graph. Inferred from ``edge_index``
            if not given.
        directed: If ``True``, only the edges that were followed to reach the
            nodes are kept. If ``False`` (default), all the edges between the
            nodes of the subgraph are kept.

    Returns:
        A tuple ``(subset, edge_index, mapping, edge_mask)`` with

        - ``subset``: the sorted ids of the nodes of the subgraph;
        - ``edge_index``: the edges of the subgraph;
        - ``mapping``: the position of each node of ``node_idx`` in ``subset``
          (its new id when ``relabel_nodes=True``);
        - ``edge_mask``: boolean mask of shape ``[num_edges]`` of the edges
          that were kept.

    Example:

    .. code-block:: python

        edge_index = mx.array([[0, 1, 2, 3], [1, 2, 3, 4]])  # path 0 -> ... -> 4

        subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(3, 2, edge_index)
        # subset: mx.array([1, 2, 3])
        # sub_edge_index: mx.array([[1, 2], [2, 3]])
        # mapping: mx.array([2])
        # edge_mask: mx.array([False, True, True, False])
    """
    return _k_hop_subgraph(
        edge_index, node_idx, num_hops, relabel_nodes, num_nodes, directed
    )
