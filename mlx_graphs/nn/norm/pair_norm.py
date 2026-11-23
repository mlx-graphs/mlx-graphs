from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.nn.norm.graph_norm import _batch_or_single
from mlx_graphs.utils.array_ops import p_norm
from mlx_graphs.utils.scatter import scatter


class PairNormalization(nn.Module):
    r"""Applies pair normalization over node features as described in the
    `"PairNorm: Tackling Oversmoothing in GNNs"
    <https://arxiv.org/abs/1909.12223>`_ paper.

    .. math::
        \mathbf{x}_i^c &= \mathbf{x}_i - \frac{1}{n}
        \sum_{i=1}^n \mathbf{x}_i \\

        \mathbf{x}_i^{\prime} &= s \cdot
        \frac{\mathbf{x}_i^c}{\sqrt{\frac{1}{n} \sum_{i=1}^n
        {\| \mathbf{x}_i^c \|}^2_2}}

    The statistics are computed separately for each graph of a batch. Also
    available as :class:`PairNorm`, its PyG name.

    Args:
        scale: Scaling factor :math:`s` of the normalization. Default: ``1.0``
        scale_individually: If set to ``True``, computes the scaling step as
            :math:`\mathbf{x}^{\prime}_i = s \cdot \frac{\mathbf{x}_i^c}
            {{\| \mathbf{x}_i^c \|}_2}`. Default: ``False``
        eps: A value added to the denominator for numerical stability.
            Default: ``1e-5``

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import PairNormalization

        norm = PairNormalization(scale=2.0)
        node_features = mx.random.normal((5, 16))
        batch_indices = mx.array([0, 0, 0, 1, 1])

        >>> norm(node_features, batch_indices).shape
        (5, 16)
    """

    def __init__(
        self, scale: float = 1.0, scale_individually: bool = False, eps: float = 1e-5
    ):
        super().__init__()
        self.scale = scale
        self.scale_individually = scale_individually
        self.eps = eps

    def __call__(
        self,
        node_features: mx.array,
        batch_indices: Optional[mx.array] = None,
        num_graphs: Optional[int] = None,
    ) -> mx.array:
        """Normalizes the node features of each graph.

        Args:
            node_features: Node features of shape ``[num_nodes, num_features]``.
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
        x = node_features - mean[batch_indices]

        if self.scale_individually:
            norm = p_norm(x, axis=-1, keepdims=True)
            return self.scale * x / (self.eps + norm)

        mean_sq_norm = scatter(
            (x * x).sum(axis=-1, keepdims=True), batch_indices, num_graphs, aggr="mean"
        )
        return self.scale * x / mx.sqrt(self.eps + mean_sq_norm[batch_indices])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
