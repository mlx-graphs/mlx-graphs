from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.utils import scatter


class Set2Set(nn.Module):
    r"""Set2Set global pooling from `"Order Matters: Sequence to Sequence for Sets"
    <https://arxiv.org/abs/1511.06391>`_ paper.

    Starting from :math:`\mathbf{q}^{*}_0 = \mathbf{0}`, each processing step
    :math:`t` reads the nodes of a graph :math:`\mathcal{V}_g` with attention:

    .. math::
        \mathbf{q}_t &= \textrm{LSTM}(\mathbf{q}^{*}_{t-1})

        \alpha_{i,t} &= \textrm{softmax}_{i \in \mathcal{V}_g} \left(
        \mathbf{x}_i \cdot \mathbf{q}_t \right)

        \mathbf{r}_t &= \sum_{i \in \mathcal{V}_g} \alpha_{i,t} \mathbf{x}_i

        \mathbf{q}^{*}_t &= \mathbf{q}_t \, \Vert \, \mathbf{r}_t,

    where :math:`\mathbf{q}^{*}_T` is the output of the layer, with twice the
    size of the input features.

    Args:
        node_features_dim: Size of input node features
        processing_steps: Number of processing steps :math:`T`

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn.pooling.set2set import Set2Set

        pool = Set2Set(16, processing_steps=3)

        node_features = mx.random.uniform(shape=(5, 16))
        batch_indices = mx.array([0, 0, 0, 1, 1])

        >>> pool(node_features, batch_indices).shape
        (2, 32)
    """

    def __init__(self, node_features_dim: int, processing_steps: int):
        super().__init__()

        self.node_features_dim = node_features_dim
        self.out_features_dim = 2 * node_features_dim
        self.processing_steps = processing_steps

        self.lstm = nn.LSTM(self.out_features_dim, node_features_dim)

    def __call__(
        self,
        node_features: mx.array,
        batch_indices: Optional[mx.array] = None,
        num_graphs: Optional[int] = None,
    ) -> mx.array:
        """Pools the node features of each graph with Set2Set.

        Args:
            node_features: Node features array.
            batch_indices: Batch array of shape (node_features.shape[0]),
                indicating for each node its batch index. If not provided, all
                nodes belong to a single graph.
            num_graphs: Number of graphs in the batch. By default, inferred as
                ``batch_indices.max() + 1``.

        Returns:
            An array of shape (num_graphs, 2 * node_features_dim) with the pooled
            features of each graph, (1, 2 * node_features_dim) if `batch_indices`
            is None.
        """
        if batch_indices is None:
            batch_indices = mx.zeros((node_features.shape[0],), dtype=mx.int32)
            num_graphs = 1
        elif num_graphs is None:
            num_graphs = batch_indices.max().item() + 1

        dtype = node_features.dtype
        hidden = mx.zeros((num_graphs, self.node_features_dim), dtype=dtype)
        cell = mx.zeros((num_graphs, self.node_features_dim), dtype=dtype)
        q_star = mx.zeros((num_graphs, self.out_features_dim), dtype=dtype)

        for _ in range(self.processing_steps):
            # one LSTM step per graph: sequences of length 1
            all_hidden, all_cell = self.lstm(q_star[:, None, :], hidden, cell)
            hidden, cell = all_hidden[:, 0], all_cell[:, 0]
            query = hidden  # [num_graphs, node_features_dim]

            # attention of each node to the query of its graph
            energy = (node_features * query[batch_indices]).sum(axis=-1, keepdims=True)
            alpha = scatter(energy, batch_indices, num_graphs, aggr="softmax")
            readout = scatter(
                alpha * node_features, batch_indices, num_graphs, aggr="add"
            )

            q_star = mx.concatenate([query, readout], axis=-1)

        return q_star
