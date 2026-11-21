"""Edge normalizations shared by the spectral convolution layers.

Both helpers keep the number of edges fixed by the input shapes (no boolean
masking): an edge that has to be dropped gets a weight of 0 instead, which
leaves a weighted sum over neighbours unchanged.
"""

from collections import OrderedDict
from typing import Any, Callable, Hashable, Literal, Optional, Tuple, Union

import mlx.core as mx

from mlx_graphs.utils import degree, invert_sqrt_degree, scatter

LaplacianNormalization = Optional[Literal["sym", "rw"]]


def _append_self_loops(edge_index: mx.array, num_nodes: int) -> mx.array:
    loop = mx.arange(num_nodes, dtype=edge_index.dtype)
    return mx.concatenate([edge_index, mx.stack([loop, loop])], axis=1)


def gcn_norm(
    edge_index: mx.array,
    num_nodes: int,
    edge_weights: Optional[mx.array] = None,
    add_self_loops: bool = True,
) -> Tuple[mx.array, mx.array]:
    r"""Symmetric GCN normalization of a graph.

    .. math::
        \hat{e}_{j,i} = \frac{e_{j,i}}{\sqrt{\hat{d}_j \hat{d}_i}}, \quad
        \hat{d}_i = \sum_{j \in \mathcal{N}(i)} e_{j,i}

    Degrees are the (weighted) in-degrees, computed after the self-loops are
    added. Nodes with a degree of 0 get a normalization of 0.

    With ``add_self_loops``, a self-loop of weight 1 is added to every node
    that does not have one. Existing self-loops keep their weight, as in PyG.
    The returned ``edge_index`` always has ``num_edges + num_nodes`` columns in
    that case: the loop appended to a node that already has one has weight 0.

    Args:
        edge_index: Graph connectivity of shape `[2, num_edges]`
        num_nodes: Number of nodes in the graph
        edge_weights: Optional scalar weight per edge, of shape `[num_edges]`
            or `[num_edges, 1]`. Default: ``None`` (all ones)
        add_self_loops: Whether to add self-loops before normalizing

    Returns:
        The edge index (with self-loops if requested) and one normalized weight
        per edge, of shape `[num_edges_out]`.
    """
    row, col = edge_index[0], edge_index[1]
    weights = None if edge_weights is None else edge_weights.reshape(-1)

    # In-degree; the unweighted case uses the exact integer count.
    deg = degree(col, num_nodes, edge_weights=weights)
    if weights is None:
        weights = mx.ones((edge_index.shape[1],), dtype=mx.float32)

    if add_self_loops:
        is_loop = (row == col).astype(mx.int32)
        num_loops = mx.zeros((num_nodes,), dtype=mx.int32).at[row].add(is_loop)
        loop_weights = (num_loops == 0).astype(weights.dtype)

        deg = deg + loop_weights
        weights = mx.concatenate([weights, loop_weights])
        edge_index = _append_self_loops(edge_index, num_nodes)
        row, col = edge_index[0], edge_index[1]

    deg_inv_sqrt = invert_sqrt_degree(deg)
    return edge_index, deg_inv_sqrt[row] * weights * deg_inv_sqrt[col]


def laplacian(
    edge_index: mx.array,
    num_nodes: int,
    edge_weights: Optional[mx.array] = None,
    normalization: LaplacianNormalization = None,
) -> Tuple[mx.array, mx.array]:
    r"""Graph Laplacian in edge list form.

    * ``None``: :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`
    * ``"sym"``: :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
      \mathbf{D}^{-1/2}`
    * ``"rw"``: :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`

    Self-loops of the input graph are ignored. As in PyG, :math:`\mathbf{D}`
    holds the (weighted) out-degrees, i.e. the degrees of ``edge_index[0]``.

    Args:
        edge_index: Graph connectivity of shape `[2, num_edges]`
        num_nodes: Number of nodes in the graph
        edge_weights: Optional scalar weight per edge. Default: ``None`` (all ones)
        normalization: ``None``, ``"sym"`` or ``"rw"``

    Returns:
        An edge index of shape `[2, num_edges + num_nodes]` and the matching
        Laplacian entries. The first ``num_edges`` entries are the off-diagonal
        ones, in the order of the input edges (0 for input self-loops); the
        last ``num_nodes`` entries are the diagonal, in node order.
    """
    if normalization not in (None, "sym", "rw"):
        raise ValueError(
            f"normalization must be None, 'sym' or 'rw', got {normalization!r}"
        )

    row, col = edge_index[0], edge_index[1]
    if edge_weights is None:
        weights = (row != col).astype(mx.float32)
    else:
        weights = edge_weights.reshape(-1)
        weights = mx.where(row != col, weights, mx.zeros_like(weights))

    deg = scatter(weights, row, num_nodes, "add")

    if normalization is None:
        diagonal = deg
    elif normalization == "sym":
        deg_inv_sqrt = invert_sqrt_degree(deg)
        weights = deg_inv_sqrt[row] * weights * deg_inv_sqrt[col]
        diagonal = mx.ones_like(deg)
    else:
        deg_inv = 1.0 / deg
        deg_inv = mx.where(deg_inv == float("inf"), mx.zeros_like(deg_inv), deg_inv)
        weights = deg_inv[row] * weights
        diagonal = mx.ones_like(deg)

    edge_index = _append_self_loops(edge_index, num_nodes)
    return edge_index, mx.concatenate([-weights, diagonal])


class _NormCache:
    """Small LRU cache of normalized graphs, keyed on the edge_index array.

    Recomputing the normalization at every layer and every epoch is wasted
    work in full-batch training, and returning the same arrays each time also
    lets fused message passing reuse its CSR conversion. Only graphs without
    edge weights are cached. The cache keeps a reference to each edge_index,
    so its ``id`` cannot be reused by another array while the entry is alive.
    """

    def __init__(self, maxsize: int = 8):
        self.maxsize = maxsize
        self._entries: "OrderedDict[Tuple[Hashable, ...], Tuple[Any, ...]]" = (
            OrderedDict()
        )

    def get(
        self,
        edge_index: mx.array,
        key: Tuple[Hashable, ...],
        compute: Callable[[], Tuple[mx.array, mx.array]],
    ) -> Tuple[mx.array, mx.array]:
        key = (id(edge_index), *key)
        hit = self._entries.get(key)
        if hit is not None and hit[0] is edge_index:
            self._entries.move_to_end(key)
            return hit[1], hit[2]
        normalized = compute()
        self._entries[key] = (edge_index, *normalized)
        if len(self._entries) > self.maxsize:
            self._entries.popitem(last=False)
        return normalized

    def clear(self):
        self._entries.clear()


_norm_cache = _NormCache()


def clear_norm_cache() -> None:
    """Drops the cached graph normalizations.

    Layers such as :class:`~mlx_graphs.nn.GCNConv` cache the normalization of
    a graph without edge weights, and reuse it as long as the same
    ``edge_index`` array is passed. Call this after modifying an
    ``edge_index`` in place.
    """
    _norm_cache.clear()


def cached_gcn_norm(
    edge_index: mx.array,
    num_nodes: int,
    edge_weights: Optional[mx.array] = None,
    add_self_loops: bool = True,
) -> Tuple[mx.array, mx.array]:
    """:func:`gcn_norm`, cached per ``edge_index`` array when the graph has no
    edge weights."""
    if edge_weights is not None:
        return gcn_norm(edge_index, num_nodes, edge_weights, add_self_loops)
    return _norm_cache.get(
        edge_index,
        ("gcn", num_nodes, add_self_loops),
        lambda: gcn_norm(edge_index, num_nodes, None, add_self_loops),
    )


SelfLoopFillValue = Union[
    float, mx.array, Literal["mean", "add", "sum", "max", "min", "mul"]
]


def attention_self_loops(
    edge_index: mx.array,
    num_nodes: int,
    edge_features: Optional[mx.array] = None,
    fill_value: SelfLoopFillValue = "mean",
) -> Tuple[mx.array, Optional[mx.array], mx.array]:
    """Self-loops of the attention layers, as PyG's ``remove_self_loops``
    followed by ``add_self_loops``, but with shapes fixed by the inputs.

    One self-loop per node is appended to ``edge_index``. Existing self-loops
    are kept in place (the number of edges cannot depend on the data), and
    the returned mask flags them: the caller excludes them from the attention
    softmax, which gives the same result as removing them.

    Args:
        edge_index: Graph connectivity of shape `[2, num_edges]`.
        num_nodes: Number of nodes.
        edge_features: Optional features of each edge, of shape
            `[num_edges, ...]`.
        fill_value: Features of the added self-loops: a constant (float or
            array), or the reduction (``"mean"``, ``"add"``/``"sum"``,
            ``"max"``, ``"min"`` or ``"mul"``) of the features of the incoming
            edges of each node, existing self-loops excluded (0 for nodes
            without incoming edges, as in PyG). Default: ``"mean"``

    Returns:
        The edge index with ``num_edges + num_nodes`` columns, the edge
        features of the same length (or ``None``), and a boolean mask of
        shape `[num_edges + num_nodes]`, ``True`` for the input self-loops.
    """
    row, col = edge_index[0], edge_index[1]
    is_loop = row == col
    loop = mx.arange(num_nodes, dtype=edge_index.dtype)
    new_edge_index = mx.concatenate([edge_index, mx.stack([loop, loop])], axis=1)
    removed = mx.concatenate([is_loop, mx.zeros((num_nodes,), dtype=mx.bool_)])

    if edge_features is None:
        return new_edge_index, None, removed

    loop_shape = (num_nodes, *edge_features.shape[1:])
    if not isinstance(fill_value, str):
        loop_features = mx.broadcast_to(
            mx.array(fill_value, dtype=edge_features.dtype), loop_shape
        )
    else:
        per_edge = (-1,) + (1,) * (edge_features.ndim - 1)
        keep = (~is_loop).reshape(per_edge)
        count = scatter((~is_loop).astype(mx.int32), col, num_nodes, "add")
        has_edges = (count > 0).reshape(per_edge)
        if fill_value == "mean":
            total = scatter(edge_features * keep, col, num_nodes, "add")
            denominator = mx.maximum(count, 1).astype(edge_features.dtype)
            loop_features = total / denominator.reshape(per_edge)
        elif fill_value in ("add", "sum"):
            loop_features = scatter(edge_features * keep, col, num_nodes, "add")
        elif fill_value in ("max", "min", "mul"):
            neutral = {"max": -float("inf"), "min": float("inf"), "mul": 1.0}
            values = mx.where(keep, edge_features, neutral[fill_value])
            loop_features = scatter(values, col, num_nodes, fill_value)
            empty = 1.0 if fill_value == "mul" else 0.0
            loop_features = mx.where(has_edges, loop_features, empty)
        else:
            raise ValueError(
                f"Unknown fill_value {fill_value!r}: use a constant, 'mean', "
                "'add', 'sum', 'max', 'min' or 'mul'"
            )
    new_edge_features = mx.concatenate(
        [edge_features, loop_features.astype(edge_features.dtype)], axis=0
    )
    return new_edge_index, new_edge_features, removed
