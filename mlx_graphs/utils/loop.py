from typing import Optional, Union

import mlx.core as mx
import numpy as np

from mlx_graphs.utils.scatter import scatter
from mlx_graphs.utils.subgraph import _take_rows
from mlx_graphs.utils.transformations import (
    _connected_nodes,
    _infer_num_nodes,
    _take_edges,
    _to_numpy,
    has_isolated_nodes,
    has_self_loops,
)
from mlx_graphs.utils.validators import (
    validate_edge_index,
    validate_edge_index_and_features,
)

FillValue = Optional[Union[float, mx.array, str]]


@validate_edge_index
def contains_self_loops(edge_index: mx.array) -> bool:
    """
    Returns ``True`` if the graph given by ``edge_index`` contains self-loops.

    PyG-named alias of :func:`~mlx_graphs.utils.transformations.has_self_loops`.

    Args:
        edge_index: a [2, num_edges] array representing the source and target
            nodes of each edge

    Returns:
        Whether the graph has at least one edge ``(i, i)``.

    Example:

    .. code-block:: python

        contains_self_loops(mx.array([[0, 1, 1], [1, 0, 1]]))
        # True
    """
    if edge_index.shape[1] == 0:
        return False
    return has_self_loops(edge_index)


@validate_edge_index
def contains_isolated_nodes(
    edge_index: mx.array, num_nodes: Optional[int] = None
) -> bool:
    """
    Returns ``True`` if the graph given by ``edge_index`` contains isolated
    nodes, i.e. nodes without any edge to another node (self-loops are
    ignored).

    PyG-named alias of
    :func:`~mlx_graphs.utils.transformations.has_isolated_nodes`, where
    ``num_nodes`` is optional.

    Args:
        edge_index: a [2, num_edges] array representing the source and target
            nodes of each edge
        num_nodes: Number of nodes in the graph. Inferred from ``edge_index``
            if not given.

    Returns:
        Whether the graph has isolated nodes.

    Example:

    .. code-block:: python

        contains_isolated_nodes(mx.array([[0, 1, 2], [1, 0, 2]]))
        # True (node 2 only has a self-loop)
    """
    return has_isolated_nodes(edge_index, _infer_num_nodes(edge_index, num_nodes))


def _check_num_nodes(edge_index: mx.array, num_nodes: Optional[int]) -> int:
    inferred = _infer_num_nodes(edge_index)
    if num_nodes is None:
        return inferred
    if inferred > num_nodes:
        raise ValueError(
            f"num_nodes={num_nodes} but `edge_index` contains node {inferred - 1}"
        )
    return int(num_nodes)


def _loop_features(
    edge_index: mx.array,
    edge_features: mx.array,
    num_nodes: int,
    fill_value: FillValue,
) -> mx.array:
    """Features of the self-loop of every node, computed from ``fill_value``."""
    shape = (num_nodes, *edge_features.shape[1:])
    if fill_value is None:
        return mx.ones(shape, dtype=edge_features.dtype)
    if isinstance(fill_value, (int, float)):
        # `mx.full` ignores an integer dtype for a float fill value.
        return mx.full(shape, fill_value, dtype=edge_features.dtype).astype(
            edge_features.dtype
        )
    if isinstance(fill_value, mx.array):
        fill = fill_value.astype(edge_features.dtype)
        return mx.broadcast_to(fill, shape)
    if isinstance(fill_value, str):
        reduce = "add" if fill_value == "sum" else fill_value
        if edge_index.shape[1] == 0:
            return mx.zeros(shape, dtype=edge_features.dtype)
        return scatter(edge_features, edge_index[1], num_nodes, aggr=reduce)
    raise ValueError(f"Invalid `fill_value` {fill_value!r}")


@validate_edge_index_and_features
def add_remaining_self_loops(
    edge_index: mx.array,
    edge_features: Optional[mx.array] = None,
    fill_value: FillValue = None,
    num_nodes: Optional[int] = None,
) -> tuple[mx.array, Optional[mx.array]]:
    """
    Adds a self-loop ``(i, i)`` to every node of the graph that does not
    already have one.

    The existing self-loops are kept with their features (if a node has
    several self-loops, only the last one is kept), while the new ones get
    features computed from ``fill_value``. All the self-loops are placed after
    the other edges, in node order.

    Args:
        edge_index: a [2, num_edges] array representing the source and target
            nodes of each edge
        edge_features: Optional features of each edge, of shape
            ``[num_edges, ...]``. Default: ``None``
        fill_value: Features of the added self-loops. Either ``None`` (ones),
            a number, an array broadcastable to the shape of one edge's
            features, or a reduction (``"add"``/``"sum"``, ``"mean"``,
            ``"max"``, ``"min"``) of the features of the incoming edges of
            each node. Default: ``None``
        num_nodes: Number of nodes in the graph. Inferred from ``edge_index``
            if not given.

    Returns:
        A tuple ``(edge_index, edge_features)`` (``edge_features`` is ``None``
        if it was not given).

    Example:

    .. code-block:: python

        edge_index = mx.array([[0, 1, 1], [1, 0, 1]])
        edge_features = mx.array([0.5, 0.5, 2.0])

        add_remaining_self_loops(edge_index, edge_features, num_nodes=3)
        # (mx.array([[0, 1, 0, 1, 2], [1, 0, 0, 1, 2]]),
        #  mx.array([0.5, 0.5, 1.0, 2.0, 1.0]))
    """
    num_nodes = _check_num_nodes(edge_index, num_nodes)
    edges = _to_numpy(edge_index)
    is_loop = edges[0] == edges[1]
    keep = np.nonzero(~is_loop)[0]

    loop_nodes = mx.arange(num_nodes, dtype=edge_index.dtype)
    loop_index = mx.stack([loop_nodes, loop_nodes])
    out_index = mx.concatenate([_take_edges(edge_index, keep), loop_index], axis=1)

    if edge_features is None:
        return out_index, None

    loop_features = _loop_features(edge_index, edge_features, num_nodes, fill_value)
    loop_positions = np.nonzero(is_loop)[0]
    if loop_positions.size > 0:
        loop_owner = edges[0][loop_positions]
        # Keep the last self-loop of each node, as in PyG.
        _, last = np.unique(loop_owner[::-1], return_index=True)
        last = loop_positions.size - 1 - last
        loop_features = mx.array(loop_features)
        loop_features[mx.array(loop_owner[last])] = edge_features[
            mx.array(loop_positions[last])
        ]
    out_features = mx.concatenate(
        [_take_rows(edge_features, keep), loop_features], axis=0
    )
    return out_index, out_features


@validate_edge_index_and_features
def segregate_self_loops(
    edge_index: mx.array,
    edge_features: Optional[mx.array] = None,
) -> tuple[mx.array, Optional[mx.array], mx.array, Optional[mx.array]]:
    """
    Splits the edges of a graph into the self-loops and the other edges.

    Args:
        edge_index: a [2, num_edges] array representing the source and target
            nodes of each edge
        edge_features: Optional features of each edge, of shape
            ``[num_edges, ...]``. Default: ``None``

    Returns:
        A tuple ``(edge_index, edge_features, loop_edge_index,
        loop_edge_features)`` with the edges that are not self-loops, their
        features, the self-loops and their features (features are ``None`` if
        ``edge_features`` was not given).

    Example:

    .. code-block:: python

        edge_index = mx.array([[0, 0, 1], [0, 1, 0]])
        segregate_self_loops(edge_index)
        # (mx.array([[0, 1], [1, 0]]), None, mx.array([[0], [0]]), None)
    """
    edges = _to_numpy(edge_index)
    is_loop = edges[0] == edges[1]
    keep, loops = np.nonzero(~is_loop)[0], np.nonzero(is_loop)[0]

    out_features = loop_features = None
    if edge_features is not None:
        out_features = _take_rows(edge_features, keep)
        loop_features = _take_rows(edge_features, loops)
    return (
        _take_edges(edge_index, keep),
        out_features,
        _take_edges(edge_index, loops),
        loop_features,
    )


@validate_edge_index_and_features
def remove_isolated_nodes(
    edge_index: mx.array,
    edge_features: Optional[mx.array] = None,
    num_nodes: Optional[int] = None,
) -> tuple[mx.array, Optional[mx.array], mx.array]:
    """
    Removes the isolated nodes of a graph and relabels the remaining ones to
    ``[0, num_remaining_nodes)`` in their original order.

    A node is isolated if it has no edge to another node: a node whose only
    edges are self-loops is isolated, and its self-loops are removed. Each
    remaining node keeps at most one self-loop (the last one), placed after
    the other edges, in node order.

    Args:
        edge_index: a [2, num_edges] array representing the source and target
            nodes of each edge
        edge_features: Optional features of each edge, of shape
            ``[num_edges, ...]``. Default: ``None``
        num_nodes: Number of nodes in the graph. Inferred from ``edge_index``
            if not given.

    Returns:
        A tuple ``(edge_index, edge_features, node_mask)`` with the relabeled
        edges, their features (``None`` if not given) and a boolean mask of
        shape ``[num_nodes]`` of the nodes that were kept.

    Example:

    .. code-block:: python

        edge_index = mx.array([[0, 1, 2], [1, 0, 2]])
        remove_isolated_nodes(edge_index, num_nodes=4)
        # (mx.array([[0, 1], [1, 0]]), None,
        #  mx.array([True, True, False, False]))
    """
    num_nodes = _check_num_nodes(edge_index, num_nodes)
    edges = _to_numpy(edge_index)
    node_mask = _connected_nodes(edge_index, num_nodes)

    mapping = np.full(num_nodes, -1, dtype=np.int64)
    mapping[node_mask] = np.arange(int(node_mask.sum()))

    is_loop = edges[0] == edges[1]
    keep = np.nonzero(~is_loop)[0]
    loop_positions = np.nonzero(is_loop)[0]
    # Last self-loop of each connected node, ordered by node.
    last_loop = np.full(num_nodes, -1, dtype=np.int64)
    last_loop[edges[0][loop_positions]] = loop_positions
    kept_loops = last_loop[node_mask & (last_loop >= 0)]

    positions = np.concatenate([keep, kept_loops])
    out_index = mx.array(mapping[edges[:, positions]], dtype=edge_index.dtype)
    if positions.size == 0:
        out_index = mx.zeros((2, 0), dtype=edge_index.dtype)

    out_features = None
    if edge_features is not None:
        out_features = _take_rows(edge_features, positions)
    return out_index, out_features, mx.array(node_mask)
