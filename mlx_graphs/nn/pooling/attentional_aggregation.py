from typing import Optional

import mlx.core as mx
from mlx.nn import Module

from mlx_graphs.utils import scatter


class AttentionalAggregation(Module):
    r"""Soft attention pooling (global attention) from `"Gated Graph Sequence Neural
    Networks" <https://arxiv.org/abs/1511.05493>`_ paper.

    .. math::
        \mathbf{r}_g = \sum_{i \in \mathcal{V}_g}
        \textrm{softmax}_{i \in \mathcal{V}_g} \left(
        h_{\textrm{gate}} ( \mathbf{x}_i ) \right) \cdot
        h_{\mathbf{\Theta}} ( \mathbf{x}_i ),

    where :math:`\mathcal{V}_g` are the nodes of graph :math:`g`,
    :math:`h_{\textrm{gate}}` (``gate_nn``) computes the attention scores and
    :math:`h_{\mathbf{\Theta}}` (``nn``) optionally transforms the node features
    before pooling.

    Args:
        gate_nn: Callable :class:`mlx.nn.Module` computing the attention scores,
            mapping node features of shape `[num_nodes, node_features_dim]` to
            shape `[num_nodes, 1]` (one score per node) or
            `[num_nodes, out_features_dim]` (one score per feature)
        nn: Optional callable :class:`mlx.nn.Module` mapping node features of shape
            `[num_nodes, node_features_dim]` to shape
            `[num_nodes, out_features_dim]` before pooling. Default: ``None``

    Example:

    .. code-block:: python

        import mlx.core as mx
        import mlx.nn as nn
        from mlx_graphs.nn.pooling.attentional_aggregation import (
            AttentionalAggregation,
        )

        pool = AttentionalAggregation(gate_nn=nn.Linear(16, 1), nn=nn.Linear(16, 32))

        node_features = mx.random.uniform(shape=(5, 16))
        batch_indices = mx.array([0, 0, 0, 1, 1])

        >>> pool(node_features, batch_indices).shape
        (2, 32)
    """

    def __init__(self, gate_nn: Module, nn: Optional[Module] = None):
        super().__init__()

        self.gate_nn = gate_nn
        if nn is not None:
            self.nn = nn

    def __call__(
        self,
        node_features: mx.array,
        batch_indices: Optional[mx.array] = None,
        num_graphs: Optional[int] = None,
    ) -> mx.array:
        """Pools the node features of each graph with attention.

        Args:
            node_features: Node features array.
            batch_indices: Batch array of shape (node_features.shape[0]),
                indicating for each node its batch index. If not provided, all
                nodes belong to a single graph.
            num_graphs: Number of graphs in the batch. By default, inferred as
                ``batch_indices.max() + 1``.

        Returns:
            An array of shape (num_graphs, out_features_dim) with the pooled
            features of each graph, (1, out_features_dim) if `batch_indices` is
            None. Graphs without nodes get zeros.
        """
        if batch_indices is None:
            batch_indices = mx.zeros((node_features.shape[0],), dtype=mx.int32)
            num_graphs = 1
        elif num_graphs is None:
            num_graphs = batch_indices.max().item() + 1

        gate = self.gate_nn(node_features)
        if "nn" in self:
            node_features = self.nn(node_features)

        # softmax over the nodes of each graph
        gate = scatter(gate, batch_indices, num_graphs, aggr="softmax")

        return scatter(gate * node_features, batch_indices, num_graphs, aggr="add")
