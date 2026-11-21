from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.nn.norm.graph_norm import _batch_or_single
from mlx_graphs.utils.scatter import degree


class GraphSizeNormalization(nn.Module):
    r"""Applies graph size normalization over each individual graph in a batch
    of node features as described in the `"Benchmarking Graph Neural Networks"
    <https://arxiv.org/abs/2003.00982>`_ paper.

    .. math::
        \mathbf{x}^{\prime}_i = \frac{\mathbf{x}_i}{\sqrt{|\mathcal{V}|}}

    where :math:`|\mathcal{V}|` is the number of nodes of the graph of node
    :math:`i`. Also available as :class:`GraphSizeNorm`, its PyG name.

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import GraphSizeNormalization

        norm = GraphSizeNormalization()
        node_features = mx.ones((5, 2))
        batch_indices = mx.array([0, 0, 0, 0, 1])

        >>> norm(node_features, batch_indices)
        array([[0.5, 0.5],
               [0.5, 0.5],
               [0.5, 0.5],
               [0.5, 0.5],
               [1, 1]], dtype=float32)
    """

    def __call__(
        self,
        node_features: mx.array,
        batch_indices: Optional[mx.array] = None,
        num_graphs: Optional[int] = None,
    ) -> mx.array:
        """Divides the features of each node by the square root of the size of
        its graph.

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
        inv_sqrt_size = mx.rsqrt(degree(batch_indices, num_graphs))
        inv_sqrt_size = inv_sqrt_size.astype(node_features.dtype)
        return node_features * inv_sqrt_size[batch_indices].reshape(-1, 1)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
