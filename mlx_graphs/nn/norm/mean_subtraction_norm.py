from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.utils.scatter import scatter


class MeanSubtractionNormalization(nn.Module):
    r"""Applies layer normalization by subtracting the mean from the inputs
    as described in the `"Revisiting 'Over-smoothing' in Deep GCNs"
    <https://arxiv.org/abs/2003.13663>`_ paper.

    .. math::
        \mathbf{x}_i = \mathbf{x}_i - \frac{1}{|\mathcal{V}|}
        \sum_{j \in \mathcal{V}} \mathbf{x}_j

    The mean is computed separately for each graph of a batch. Also available
    as :class:`MeanSubtractionNorm`, its PyG name.

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import MeanSubtractionNormalization

        norm = MeanSubtractionNormalization()
        node_features = mx.array([[1.0], [3.0], [5.0]])
        batch_indices = mx.array([0, 0, 1])

        >>> norm(node_features, batch_indices)
        array([[-1],
               [1],
               [0]], dtype=float32)
    """

    def __call__(
        self,
        node_features: mx.array,
        batch_indices: Optional[mx.array] = None,
        num_graphs: Optional[int] = None,
    ) -> mx.array:
        """Subtracts the mean node features of each graph.

        Args:
            node_features: Node features of shape ``[num_nodes, num_features]``.
            batch_indices: Graph of each node, of shape ``[num_nodes]``. Default:
                ``None`` (all the nodes belong to a single graph).
            num_graphs: Number of graphs. Default: ``None`` (inferred from
                ``batch_indices``).

        Returns:
            The centered node features.
        """
        if batch_indices is None:
            return node_features - node_features.mean(axis=0, keepdims=True)
        if num_graphs is None:
            num_graphs = batch_indices.max().item() + 1
        mean = scatter(node_features, batch_indices, num_graphs, aggr="mean")
        return node_features - mean[batch_indices]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
