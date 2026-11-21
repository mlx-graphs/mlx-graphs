from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.utils.scatter import scatter


def _batch_or_single(
    node_features: mx.array,
    batch_indices: Optional[mx.array],
    num_graphs: Optional[int],
) -> tuple[mx.array, int]:
    """Returns the batch indices of the nodes (all zeros without batch) and the
    number of graphs."""
    if batch_indices is None:
        return mx.zeros((node_features.shape[0],), dtype=mx.int32), 1
    if num_graphs is None:
        num_graphs = batch_indices.max().item() + 1 if batch_indices.size > 0 else 1
    return batch_indices, num_graphs


class GraphNormalization(nn.Module):
    r"""Applies graph normalization over individual graphs as described in the
    `"GraphNorm: A Principled Approach to Accelerating Graph Neural Network
    Training" <https://arxiv.org/abs/2009.03294>`_ paper.

    .. math::
        \mathbf{x}^{\prime}_i = \frac{\mathbf{x} - \alpha \odot
        \textrm{E}[\mathbf{x}]}
        {\sqrt{\textrm{Var}[\mathbf{x} - \alpha \odot \textrm{E}[\mathbf{x}]]
        + \epsilon}} \odot \gamma + \beta

    where :math:`\alpha` denotes parameters that learn how much information
    to keep in the mean. The statistics are computed per graph and per
    feature. Also available as :class:`GraphNorm`, its PyG name.

    Args:
        node_features_dim: Size of the input node features.
        eps: A value added to the denominator for numerical stability.
            Default: ``1e-5``

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import GraphNormalization

        norm = GraphNormalization(16)
        node_features = mx.random.normal((5, 16))
        batch_indices = mx.array([0, 0, 0, 1, 1])

        >>> norm(node_features, batch_indices).shape
        (5, 16)
    """

    def __init__(self, node_features_dim: int, eps: float = 1e-5):
        super().__init__()
        self.node_features_dim = node_features_dim
        self.eps = eps
        self.weight = mx.ones((node_features_dim,))
        self.bias = mx.zeros((node_features_dim,))
        self.mean_scale = mx.ones((node_features_dim,))

    def __call__(
        self,
        node_features: mx.array,
        batch_indices: Optional[mx.array] = None,
        num_graphs: Optional[int] = None,
    ) -> mx.array:
        """Normalizes the node features of each graph.

        Args:
            node_features: Node features of shape ``[num_nodes, node_features_dim]``.
            batch_indices: Graph of each node, of shape ``[num_nodes]``. Default:
                ``None`` (all the nodes belong to a single graph).
            num_graphs: Number of graphs. Default: ``None`` (inferred from
                ``batch_indices``).

        Returns:
            The normalized node features.
        """
        batch_indices, num_graphs = _batch_or_single(
            node_features, batch_indices, num_graphs
        )
        mean = scatter(node_features, batch_indices, num_graphs, aggr="mean")
        out = node_features - mean[batch_indices] * self.mean_scale
        var = scatter(out * out, batch_indices, num_graphs, aggr="mean")
        std = mx.sqrt(var + self.eps)[batch_indices]
        return self.weight * out / std + self.bias

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.node_features_dim})"
