from typing import Callable, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_graphs.nn.conv.gcn_conv import GCNConv
from mlx_graphs.nn.models.basic_gnn import _resolve_activation
from mlx_graphs.nn.pooling._sparse import spspmm
from mlx_graphs.nn.pooling.topk_pool import TopKPooling
from mlx_graphs.utils.transformations import _to_numpy


def _remove_self_loops(
    edge_index: mx.array, edge_weights: mx.array
) -> Tuple[mx.array, mx.array]:
    edges = _to_numpy(edge_index)
    keep = np.nonzero(edges[0] != edges[1])[0]
    if keep.size == edges.shape[1]:
        return edge_index, edge_weights
    keep_mx = mx.array(keep.astype(np.int32))
    return mx.array(edges[:, keep].astype(np.int32)), edge_weights[keep_mx]


class GraphUNet(nn.Module):
    r"""The Graph U-Net model from the `"Graph U-Nets"
    <https://arxiv.org/abs/1905.05178>`_ paper, which implements a U-Net like
    architecture with graph pooling and unpooling operations.

    The encoder alternates :class:`~mlx_graphs.nn.GCNConv` layers
    (``improved=True``) and :class:`~mlx_graphs.nn.TopKPooling`. Before each
    pooling step, the graph is augmented with its 2-hop edges (the edges of
    :math:`(\mathbf{A} + \mathbf{I})^2` without self-loops, weighted), so that
    the pooled graph stays connected. The decoder puts the pooled features
    back at their positions and adds (``sum_res``) or concatenates them to
    the features of the encoder at the same depth.

    Args:
        node_features_dim: Size of the input node features.
        hidden_features_dim: Size of the hidden node features.
        out_features_dim: Size of the output node features.
        depth: The depth of the U-Net (number of pooling steps).
        pool_ratios: Pooling ratio of each depth, or one ratio for all.
            Default: ``0.5``
        sum_res: Whether the skip connections are summed (``True``) or
            concatenated. Default: ``True``
        act: The non-linearity, a name (e.g. ``"relu"``) or a callable.
            Default: ``"relu"``

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import GraphUNet

        model = GraphUNet(16, 32, 4, depth=2)
        edge_index = mx.array([[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]])
        node_features = mx.random.normal((5, 16))
        >>> model(edge_index, node_features).shape
        (5, 4)
    """

    def __init__(
        self,
        node_features_dim: int,
        hidden_features_dim: int,
        out_features_dim: int,
        depth: int,
        pool_ratios: Union[float, List[float]] = 0.5,
        sum_res: bool = True,
        act: Union[str, Callable] = "relu",
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("`depth` must be at least 1")
        self.node_features_dim = node_features_dim
        self.hidden_features_dim = hidden_features_dim
        self.out_features_dim = out_features_dim
        self.depth = depth
        if isinstance(pool_ratios, (list, tuple)):
            ratios = list(pool_ratios)
            ratios += [ratios[-1]] * (depth - len(ratios))
            self.pool_ratios = ratios[:depth]
        else:
            self.pool_ratios = [pool_ratios] * depth
        self.act = _resolve_activation(act)
        self.sum_res = sum_res

        channels = hidden_features_dim
        self.down_convs = [GCNConv(node_features_dim, channels, improved=True)]
        self.pools = []
        for i in range(depth):
            self.pools.append(TopKPooling(channels, self.pool_ratios[i]))
            self.down_convs.append(GCNConv(channels, channels, improved=True))

        in_channels = channels if sum_res else 2 * channels
        self.up_convs = [
            GCNConv(in_channels, channels, improved=True) for _ in range(depth - 1)
        ]
        self.up_convs.append(GCNConv(in_channels, out_features_dim, improved=True))

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        batch_indices: Optional[mx.array] = None,
        edge_weights: Optional[mx.array] = None,
    ) -> mx.array:
        """Computes the node embeddings.

        Args:
            edge_index: Edges of shape ``[2, num_edges]``.
            node_features: Node features of shape ``[num_nodes,
                node_features_dim]``.
            batch_indices: Graph of each node, of shape ``[num_nodes]``.
                Default: ``None`` (a single graph).
            edge_weights: Optional weight of each edge, of shape
                ``[num_edges]``. Default: ``None`` (ones).

        Returns:
            The node embeddings, of shape ``[num_nodes, out_features_dim]``.
        """
        x = node_features
        if batch_indices is None:
            batch_indices = mx.zeros((x.shape[0],), dtype=mx.int32)
        if edge_weights is None:
            edge_weights = mx.ones((edge_index.shape[1],), dtype=x.dtype)
        if edge_weights.ndim != 1 or edge_weights.shape[0] != edge_index.shape[1]:
            raise ValueError("edge_weights must have shape [num_edges]")

        x = self.act(self.down_convs[0](edge_index, x, edge_weights))

        xs = [x]
        edge_indices = [edge_index]
        all_edge_weights = [edge_weights]
        perms = []

        for i in range(1, self.depth + 1):
            edge_index, edge_weights = self.augment_adj(
                edge_index, edge_weights, x.shape[0]
            )
            x, edge_index, edge_weights, batch_indices, perm, _ = self.pools[i - 1](
                edge_index, x, edge_weights, batch_indices
            )
            x = self.act(self.down_convs[i](edge_index, x, edge_weights))

            if i < self.depth:
                xs.append(x)
                edge_indices.append(edge_index)
                all_edge_weights.append(edge_weights)
            perms.append(perm)

        for i in range(self.depth):
            j = self.depth - 1 - i
            res = xs[j]
            perm = perms[j]

            up = mx.zeros_like(res).at[perm].add(x)
            x = res + up if self.sum_res else mx.concatenate([res, up], axis=-1)

            x = self.up_convs[i](edge_indices[j], x, all_edge_weights[j])
            if i < self.depth - 1:
                x = self.act(x)

        return x

    def augment_adj(
        self, edge_index: mx.array, edge_weights: mx.array, num_nodes: int
    ) -> Tuple[mx.array, mx.array]:
        r"""Edges of :math:`(\mathbf{A} + \mathbf{I})^2` without self-loops
        (where the self-loops of :math:`\mathbf{A}` are replaced by
        :math:`\mathbf{I}`), sorted by source then target node."""
        edge_index, edge_weights = _remove_self_loops(edge_index, edge_weights)
        loops = mx.arange(num_nodes, dtype=edge_index.dtype)
        edge_index = mx.concatenate([edge_index, mx.stack([loops, loops])], axis=1)
        edge_weights = mx.concatenate(
            [edge_weights, mx.ones((num_nodes,), dtype=edge_weights.dtype)]
        )
        edge_index, edge_weights = spspmm(
            edge_index, edge_weights, edge_index, edge_weights, num_nodes
        )
        return _remove_self_loops(edge_index, edge_weights)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.node_features_dim}, "
            f"{self.hidden_features_dim}, {self.out_features_dim}, "
            f"depth={self.depth}, pool_ratios={self.pool_ratios})"
        )
