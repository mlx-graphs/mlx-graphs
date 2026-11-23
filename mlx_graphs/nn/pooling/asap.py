from typing import Callable, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_graphs.nn.conv.le_conv import LEConv
from mlx_graphs.nn.pooling._sparse import spspmm, transpose
from mlx_graphs.nn.pooling.select import _SelectTopK
from mlx_graphs.utils.loop import add_remaining_self_loops
from mlx_graphs.utils.scatter import scatter, scatter_softmax
from mlx_graphs.utils.transformations import _to_numpy


class ASAPooling(nn.Module):
    r"""The Adaptive Structure Aware Pooling operator from the `"ASAP: Adaptive
    Structure Aware Pooling for Learning Hierarchical Graph Representations"
    <https://arxiv.org/abs/1911.07979>`_ paper.

    Every node :math:`i` defines a cluster made of its 1-hop neighbourhood
    (and itself). The cluster representation is an attention-weighted sum of
    the features of its members, where the attention (a softmax over the
    members) compares each member with a master query (the feature-wise
    maximum of the members). The clusters with the best fitness (a sigmoid of
    an :class:`~mlx_graphs.nn.LEConv` layer) are kept, scaled by their
    fitness, and the pooled adjacency is :math:`\mathbf{S}^{\top} \mathbf{A}
    \mathbf{S}`, where :math:`\mathbf{S}` holds the attention scores of the
    kept clusters.

    Args:
        node_features_dim: Size of the input node features.
        ratio: The graph pooling ratio, used to compute
            :math:`k = \lceil \mathrm{ratio} \cdot N \rceil` for a float, or
            :math:`k` itself for an integer. Default: ``0.5``
        GNN: Optional class of a graph neural network layer computing the
            features used by the attention, created as
            ``GNN(node_features_dim, node_features_dim, **kwargs)`` and called
            as ``gnn(edge_index, node_features, edge_weights)`` (e.g.
            :class:`~mlx_graphs.nn.GraphConv`). Default: ``None``
        dropout: Dropout probability of the attention scores.
            Default: ``0.0``
        negative_slope: Negative slope of the leaky ReLU of the attention.
            Default: ``0.2``
        add_self_loops: Whether to add a self-loop to every pooled node (the
            pooled self-loops are removed otherwise). Default: ``False``
        **kwargs: Additional arguments of ``GNN``.

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import ASAPooling

        pool = ASAPooling(16, ratio=0.5)
        edge_index = mx.array([[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]])
        node_features = mx.random.normal((5, 16))
        batch_indices = mx.array([0, 0, 0, 1, 1])

        x, edge_index, edge_weights, batch_indices, perm = pool(
            edge_index, node_features, batch_indices=batch_indices
        )
        >>> x.shape
        (3, 16)
    """

    def __init__(
        self,
        node_features_dim: int,
        ratio: Union[float, int] = 0.5,
        GNN: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.0,
        negative_slope: float = 0.2,
        add_self_loops: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.node_features_dim = node_features_dim
        self.ratio = ratio
        self.negative_slope = negative_slope
        self.add_self_loops = add_self_loops
        # PyTorch's default initialization, as in PyG
        self.lin = nn.Linear(node_features_dim, node_features_dim)
        self.att = nn.Linear(2 * node_features_dim, 1)
        self.gnn_score = LEConv(node_features_dim, 1)
        if GNN is not None:
            self.gnn_intra_cluster = GNN(node_features_dim, node_features_dim, **kwargs)
        self.select = _SelectTopK(1, ratio)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_weights: Optional[mx.array] = None,
        batch_indices: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array, Optional[mx.array], mx.array, mx.array]:
        """Pools the graphs.

        Args:
            edge_index: Edges of shape ``[2, num_edges]``.
            node_features: Node features of shape ``[num_nodes,
                node_features_dim]``.
            edge_weights: Optional weight of each edge, of shape
                ``[num_edges]``. Default: ``None``
            batch_indices: Graph of each node, of shape ``[num_nodes]``.
                Default: ``None`` (a single graph).

        Returns:
            A tuple ``(node_features, edge_index, edge_weights, batch_indices,
            perm)`` with the pooled graph (``edge_weights`` is ``None`` if it
            was not given) and the indices ``perm`` of the kept clusters.
        """
        x = node_features
        num_nodes = x.shape[0]
        if edge_weights is not None:
            edge_weights = edge_weights.reshape(-1)
        edge_index, edge_weights = add_remaining_self_loops(
            edge_index, edge_weights, fill_value=1.0, num_nodes=num_nodes
        )
        if batch_indices is None:
            batch_indices = mx.zeros((num_nodes,), dtype=mx.int32)
        if x.ndim == 1:
            x = x[:, None]
        src, dst = edge_index[0], edge_index[1]

        x_pool = x
        if "gnn_intra_cluster" in self:
            x_pool = self.gnn_intra_cluster(edge_index, x, edge_weights)

        x_pool_j = x_pool[src]
        x_q = scatter(x_pool_j, dst, num_nodes, "max")
        x_q = self.lin(x_q)[dst]

        score = self.att(mx.concatenate([x_q, x_pool_j], axis=-1)).reshape(-1)
        score = nn.leaky_relu(score, self.negative_slope)
        score = scatter_softmax(score, dst, num_nodes)
        if self.dropout is not None:
            score = self.dropout(score)

        x = scatter(x[src] * score[:, None], dst, num_nodes, "add")

        # Cluster selection.
        fitness = mx.sigmoid(self.gnn_score(edge_index, x)).reshape(-1)
        perm, _ = self.select(fitness, batch_indices)
        x = x[perm] * fitness[perm][:, None]
        batch_indices = batch_indices[perm]

        # Graph coarsening: S^T A S, with S [num_nodes, num_kept] the
        # attention scores of the kept clusters.
        edges = _to_numpy(edge_index).astype(np.int64)
        kept = _to_numpy(perm).astype(np.int64)
        position = np.full(num_nodes, -1, dtype=np.int64)
        position[kept] = np.arange(kept.size)
        in_s = np.nonzero(position[edges[1]] >= 0)[0]
        s_index = mx.array(
            np.stack([edges[0][in_s], position[edges[1][in_s]]]).astype(np.int32)
        )
        s_value = score[mx.array(in_s.astype(np.int32))]

        a_s_index, a_s_value = spspmm(
            edge_index, edge_weights, s_index, s_value, kept.size
        )
        new_index, new_value = spspmm(
            transpose(s_index), s_value, a_s_index, a_s_value, kept.size
        )

        new_weights = new_value if edge_weights is not None else None
        if self.add_self_loops:
            # the pooled self-loops keep their weight
            new_index, new_weights = add_remaining_self_loops(
                new_index, new_weights, num_nodes=kept.size
            )
        else:
            new_edges = _to_numpy(new_index)
            not_loop = np.nonzero(new_edges[0] != new_edges[1])[0]
            new_index = mx.array(new_edges[:, not_loop].astype(np.int32))
            if new_weights is not None:
                new_weights = new_weights[mx.array(not_loop.astype(np.int32))]

        return x, new_index, new_weights, batch_indices, perm

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.node_features_dim}, ratio={self.ratio})"
        )
