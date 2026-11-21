from typing import Callable, Optional, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_graphs.utils.scatter import scatter_softmax
from mlx_graphs.utils.transformations import _to_numpy


def topk(
    scores: mx.array,
    ratio: Optional[Union[float, int]],
    batch_indices: mx.array,
    min_score: Optional[float] = None,
    tol: float = 1e-7,
) -> mx.array:
    r"""Selects the nodes with the highest scores in each graph.

    If ``min_score`` is ``None``, keeps the :math:`k` best nodes of each graph,
    with :math:`k = \lceil \mathrm{ratio} \cdot N \rceil` for a ratio smaller
    than 1 and :math:`k = \mathrm{ratio}` otherwise (all the nodes of graphs
    with fewer than :math:`k` nodes). Otherwise, keeps the nodes whose score is
    larger than ``min_score``, and the best node of each graph in all cases.

    Args:
        scores: Score of each node, of shape ``[num_nodes]``.
        ratio: The pooling ratio, or the number of nodes to keep per graph.
            Ignored if ``min_score`` is given.
        batch_indices: Graph of each node, of shape ``[num_nodes]``.
        min_score: Minimal score of the kept nodes. Default: ``None``
        tol: Tolerance on the best score of each graph, used with
            ``min_score``. Default: ``1e-7``

    Returns:
        The indices of the kept nodes. Without ``min_score``, they are grouped
        by graph, by decreasing score; with ``min_score``, in increasing order.

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import topk

        scores = mx.array([0.1, 0.9, 0.5, 0.3, 0.7])
        batch_indices = mx.array([0, 0, 0, 1, 1])
        >>> topk(scores, 0.5, batch_indices)
        array([1, 2, 4], dtype=int32)
    """
    if scores.size == 0:
        # before reading the scores back: MLX can crash when converting an
        # empty matmul output to numpy
        return mx.zeros((0,), dtype=mx.int32)
    score = _to_numpy(scores.reshape(-1)).astype(np.float64)
    batch = _to_numpy(batch_indices).reshape(-1).astype(np.int64)
    num_graphs = int(batch.max()) + 1

    if min_score is not None:
        scores_max = np.full(num_graphs, -np.inf)
        np.maximum.at(scores_max, batch, score)
        scores_min = np.minimum(scores_max[batch] - tol, min_score)
        return mx.array(np.nonzero(score > scores_min)[0].astype(np.int32))

    if ratio is None:
        raise ValueError(
            "At least one of the 'ratio' and 'min_score' parameters must be "
            "specified"
        )
    num_nodes = np.bincount(batch, minlength=num_graphs)
    if ratio >= 1:
        k = np.full(num_graphs, int(ratio))
    else:
        k = np.ceil(np.float32(ratio) * num_nodes.astype(np.float32)).astype(np.int64)

    # Sorted by graph, then by decreasing score.
    order = np.lexsort((-score, batch))
    starts = np.cumsum(num_nodes) - num_nodes
    rank = np.arange(order.size) - starts[batch[order]]
    return mx.array(order[rank < k[batch[order]]].astype(np.int32))


def filter_adj(
    edge_index: mx.array,
    edge_features: Optional[mx.array],
    perm: mx.array,
    num_nodes: Optional[int] = None,
) -> tuple[mx.array, Optional[mx.array]]:
    """Keeps the edges between the nodes ``perm``, relabeled by their position
    in ``perm``.

    Args:
        edge_index: Edges of shape ``[2, num_edges]``.
        edge_features: Optional features of each edge, of shape
            ``[num_edges, ...]``.
        perm: Indices of the kept nodes. Node ``perm[i]`` becomes node ``i``.
        num_nodes: Number of nodes. Default: ``None`` (inferred from
            ``edge_index`` and ``perm``).

    Returns:
        The filtered ``edge_index`` and ``edge_features``.

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import filter_adj

        edge_index = mx.array([[0, 1, 2], [1, 2, 0]])
        edge_index, _ = filter_adj(edge_index, None, mx.array([2, 0]))
        >>> edge_index
        array([[0],
               [1]], dtype=int32)
    """
    edges = _to_numpy(edge_index).astype(np.int64).reshape(2, -1)
    kept = _to_numpy(perm).reshape(-1).astype(np.int64)
    if num_nodes is None:
        num_nodes = int(max(edges.max(initial=-1), kept.max(initial=-1))) + 1

    mapping = np.full(num_nodes, -1, dtype=np.int64)
    mapping[kept] = np.arange(kept.size)
    row, col = mapping[edges[0]], mapping[edges[1]]
    keep = np.nonzero((row >= 0) & (col >= 0))[0]

    new_edge_index = mx.array(np.stack([row[keep], col[keep]]).astype(np.int32))
    if edge_features is not None:
        edge_features = edge_features[mx.array(keep)]
    return new_edge_index, edge_features


def _resolve_activation(act: Union[str, Callable]) -> Callable:
    if callable(act):
        return act
    fn = getattr(mx, act, None) or getattr(nn, act, None)
    if fn is None or not callable(fn):
        raise ValueError(f"Unknown activation function {act!r}")
    return fn


class _SelectTopK(nn.Module):
    """Learnable top-k node selection of :class:`TopKPooling` and
    :class:`SAGPooling` (PyG's ``SelectTopK``)."""

    def __init__(
        self,
        node_features_dim: int,
        ratio: Optional[Union[int, float]] = 0.5,
        min_score: Optional[float] = None,
        act: Union[str, Callable] = "tanh",
    ):
        super().__init__()
        if ratio is None and min_score is None:
            raise ValueError(
                "At least one of the 'ratio' and 'min_score' parameters must be "
                "specified"
            )
        self.node_features_dim = node_features_dim
        self.ratio = ratio
        self.min_score = min_score
        self._act = _resolve_activation(act)
        bound = 1.0 / np.sqrt(node_features_dim)
        self.weight = mx.random.uniform(-bound, bound, (1, node_features_dim))

    def __call__(
        self, node_features: mx.array, batch_indices: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Returns the indices of the kept nodes and their scores."""
        if node_features.ndim == 1:
            node_features = node_features.reshape(-1, 1)
        score = (node_features * self.weight).sum(axis=-1)
        if self.min_score is None:
            weight_norm = mx.sqrt((self.weight * self.weight).sum(axis=-1))
            score = self._act(score / weight_norm)
        else:
            num_graphs = batch_indices.max().item() + 1 if batch_indices.size > 0 else 1
            score = scatter_softmax(score, batch_indices, num_graphs)
        perm = topk(score, self.ratio, batch_indices, self.min_score)
        return perm, score[perm]


def _pool(
    node_features: mx.array,
    edge_index: mx.array,
    edge_features: Optional[mx.array],
    batch_indices: mx.array,
    perm: mx.array,
    score: mx.array,
    multiplier: float,
) -> tuple[mx.array, mx.array, Optional[mx.array], mx.array, mx.array, mx.array]:
    """Keeps the nodes ``perm``, scaled by ``score``, and the edges between
    them."""
    out = node_features[perm] * score.reshape(-1, 1)
    if multiplier != 1:
        out = multiplier * out
    edge_index, edge_features = filter_adj(
        edge_index, edge_features, perm, num_nodes=node_features.shape[0]
    )
    return out, edge_index, edge_features, batch_indices[perm], perm, score
