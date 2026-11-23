from typing import Callable, NamedTuple, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_graphs.utils.scatter import scatter, scatter_softmax
from mlx_graphs.utils.transformations import _to_numpy


class UnpoolInfo(NamedTuple):
    """What :meth:`EdgePooling.unpool` needs to undo a pooling step."""

    edge_index: mx.array
    cluster: mx.array
    batch_indices: mx.array
    new_edge_score: mx.array


def _merge_greedily(
    edges: np.ndarray, order: np.ndarray, num_nodes: int
) -> Tuple[np.ndarray, list, int]:
    """Contracts the edges in ``order`` that touch no already contracted node.

    Returns the cluster of each node, the contracted edges and the number of
    clusters (contracted edges first, then the remaining nodes in order).
    """
    cluster = np.empty(num_nodes, dtype=np.int64)
    free = np.ones(num_nodes, dtype=bool)
    chosen = []
    sources, targets = edges[0].tolist(), edges[1].tolist()
    for e in order.tolist():
        source, target = sources[e], targets[e]
        if not free[source] or not free[target]:
            continue
        cluster[source] = cluster[target] = len(chosen)
        free[source] = free[target] = False
        chosen.append(e)
    remaining = np.nonzero(free)[0]
    cluster[remaining] = len(chosen) + np.arange(remaining.size)
    return cluster, chosen, len(chosen) + remaining.size


class EdgePooling(nn.Module):
    r"""The edge pooling operator from the `"Towards Graph Pooling by Edge
    Contraction" <https://graphreason.github.io/papers/17.pdf>`_ and
    `"Edge Contraction Pooling for Graph Neural Networks"
    <https://arxiv.org/abs/1905.10990>`_ papers.

    Each edge gets a score from the concatenated features of its two nodes
    (a linear layer, followed by ``edge_score_method``, plus
    ``add_to_edge_score``). Edges are then contracted greedily by decreasing
    score, skipping the edges that touch an already contracted node. The
    features of a new node are the sum of the features of its merged nodes,
    multiplied by the score of the contracted edge (1 for the nodes that were
    not merged).

    As in PyG, the greedy contraction runs on the host (numpy), and equal
    scores are ordered by edge index.

    Args:
        node_features_dim: Size of the input node features.
        edge_score_method: Function ``(raw_scores, edge_index, num_nodes)``
            computing the edge scores, e.g.
            :meth:`compute_edge_score_softmax` (the default, a softmax over the
            incoming edges of each node), :meth:`compute_edge_score_tanh` or
            :meth:`compute_edge_score_sigmoid`.
        dropout: Dropout probability of the raw edge scores. Default: ``0.0``
        add_to_edge_score: Value added to every edge score, so that the
            features of merged nodes are not scaled down too much.
            Default: ``0.5``

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import EdgePooling

        pool = EdgePooling(16)
        edge_index = mx.array([[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]])
        node_features = mx.random.normal((5, 16))
        batch_indices = mx.array([0, 0, 0, 1, 1])

        x, edge_index, batch_indices, unpool_info = pool(
            edge_index, node_features, batch_indices
        )
        >>> x.shape
        (3, 16)
        x, edge_index, batch_indices = pool.unpool(x, unpool_info)
    """

    def __init__(
        self,
        node_features_dim: int,
        edge_score_method: Optional[Callable] = None,
        dropout: float = 0.0,
        add_to_edge_score: float = 0.5,
    ):
        super().__init__()
        self.node_features_dim = node_features_dim
        if edge_score_method is None:
            edge_score_method = self.compute_edge_score_softmax
        self.compute_edge_score = edge_score_method
        self.add_to_edge_score = add_to_edge_score
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        # PyTorch's default initialization, as in PyG
        self.lin = nn.Linear(2 * node_features_dim, 1)

    @staticmethod
    def compute_edge_score_softmax(
        raw_edge_score: mx.array, edge_index: mx.array, num_nodes: int
    ) -> mx.array:
        """Normalizes the scores with a softmax over the incoming edges of
        each node."""
        return scatter_softmax(raw_edge_score, edge_index[1], num_nodes)

    @staticmethod
    def compute_edge_score_tanh(
        raw_edge_score: mx.array,
        edge_index: Optional[mx.array] = None,
        num_nodes: Optional[int] = None,
    ) -> mx.array:
        """Normalizes the scores with a tanh."""
        return mx.tanh(raw_edge_score)

    @staticmethod
    def compute_edge_score_sigmoid(
        raw_edge_score: mx.array,
        edge_index: Optional[mx.array] = None,
        num_nodes: Optional[int] = None,
    ) -> mx.array:
        """Normalizes the scores with a sigmoid."""
        return mx.sigmoid(raw_edge_score)

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        batch_indices: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array, mx.array, UnpoolInfo]:
        """Pools the graphs by contracting edges.

        Args:
            edge_index: Edges of shape ``[2, num_edges]``.
            node_features: Node features of shape ``[num_nodes,
                node_features_dim]``.
            batch_indices: Graph of each node, of shape ``[num_nodes]``.
                Default: ``None`` (a single graph).

        Returns:
            A tuple ``(node_features, edge_index, batch_indices, unpool_info)``
            with the pooled graph (whose edges are coalesced, and include the
            self-loops of the contracted edges) and the information needed by
            :meth:`unpool`.
        """
        num_nodes = node_features.shape[0]
        if batch_indices is None:
            batch_indices = mx.zeros((num_nodes,), dtype=mx.int32)
        e = mx.concatenate(
            [node_features[edge_index[0]], node_features[edge_index[1]]], axis=-1
        )
        e = self.lin(e).reshape(-1)
        if self.dropout is not None:
            e = self.dropout(e)
        e = self.compute_edge_score(e, edge_index, num_nodes)
        e = e + self.add_to_edge_score
        return self._merge_edges(edge_index, node_features, batch_indices, e)

    def _merge_edges(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        batch_indices: mx.array,
        edge_score: mx.array,
    ) -> Tuple[mx.array, mx.array, mx.array, UnpoolInfo]:
        num_nodes = node_features.shape[0]
        edges = _to_numpy(edge_index).astype(np.int64).reshape(2, -1)
        scores = _to_numpy(edge_score).reshape(-1)
        # decreasing score, ties by edge index
        order = np.argsort(-scores, kind="stable")
        cluster_np, chosen, num_clusters = _merge_greedily(edges, order, num_nodes)
        cluster = mx.array(cluster_np.astype(np.int32))

        new_x = scatter(node_features, cluster, num_clusters, "add")
        num_remaining = num_clusters - len(chosen)
        new_edge_score = edge_score[mx.array(np.array(chosen, dtype=np.int32))]
        if num_remaining > 0:
            new_edge_score = mx.concatenate(
                [new_edge_score, mx.ones((num_remaining,), dtype=new_x.dtype)]
            )
        new_x = new_x * new_edge_score.reshape(-1, 1)

        # coalesced edges between the clusters
        new_edges = cluster_np[edges]
        keys = np.unique(new_edges[0] * num_clusters + new_edges[1])
        new_edge_index = mx.array(
            np.stack([keys // num_clusters, keys % num_clusters]).astype(np.int32)
        )
        batch_np = _to_numpy(batch_indices).reshape(-1)
        new_batch = np.empty(num_clusters, dtype=batch_np.dtype)
        new_batch[cluster_np] = batch_np
        new_batch_indices = mx.array(new_batch).astype(batch_indices.dtype)

        unpool_info = UnpoolInfo(
            edge_index=edge_index,
            cluster=cluster,
            batch_indices=batch_indices,
            new_edge_score=new_edge_score,
        )
        return new_x, new_edge_index, new_batch_indices, unpool_info

    def unpool(
        self, node_features: mx.array, unpool_info: UnpoolInfo
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Unpools a pooled graph: every node gets the features of its
        cluster, divided by the score of the contracted edge.

        Args:
            node_features: Features of the pooled nodes.
            unpool_info: The information returned by the pooling step.

        Returns:
            A tuple ``(node_features, edge_index, batch_indices)`` of the
            unpooled graph.
        """
        new_x = node_features / unpool_info.new_edge_score.reshape(-1, 1)
        new_x = new_x[unpool_info.cluster]
        return new_x, unpool_info.edge_index, unpool_info.batch_indices

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.node_features_dim})"
