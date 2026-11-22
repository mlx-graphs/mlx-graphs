from typing import Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.nn.conv.edge_conv import EdgeConv
from mlx_graphs.utils.spatial import knn


class DynamicEdgeConv(EdgeConv):
    r"""Dynamic edge convolution layer from the `"Dynamic Graph CNN for Learning
    on Point Clouds" <https://arxiv.org/abs/1801.07829>`_ paper (see
    :class:`EdgeConv`), where the graph is dynamically constructed using the
    nearest neighbors in the feature space.

    .. math::
        \mathbf{h}_i = \bigoplus_{j \in \mathcal{N}_k(i)}
        \text{MLP} \left( [ \mathbf{x}_i, \, \mathbf{x}_j - \mathbf{x}_i ] \right),

    where :math:`\mathcal{N}_k(i)` are the :math:`k` nearest neighbors of node
    :math:`i` (including :math:`i` itself) among the nodes of its graph, under
    the euclidean distance between the node features.

    The layer also works on bipartite graphs: pass ``node_features`` as a
    ``(src_features, dst_features)`` tuple (and ``batch_indices`` as a tuple of
    two batch vectors); each destination node aggregates its :math:`k` nearest
    source nodes.

    Args:
        mlp: Callable :class:`mlx.nn.Module` applied on each edge, mapping the
            concatenated features of shape `[num_edges, 2 * node_features_dim]` to
            shape `[num_edges, out_features_dim]`
        k: Number of nearest neighbors.
        **kwargs: Arguments of :class:`MessagePassing`, e.g. ``aggr`` (``"max"``
            by default, or ``"add"``, ``"mean"``, ``"min"``)

    Example:

    .. code-block:: python

        import mlx.core as mx
        import mlx.nn as nn
        from mlx_graphs.nn import DynamicEdgeConv

        mlp = nn.Sequential(nn.Linear(2 * 3, 32), nn.ReLU(), nn.Linear(32, 32))
        conv = DynamicEdgeConv(mlp, k=4)

        points = mx.random.normal((10, 3))
        batch_indices = mx.array([0] * 5 + [1] * 5)

        h = conv(points, batch_indices)
        >>> h.shape
        (10, 32)
    """

    def __init__(self, mlp: nn.Module, k: int, **kwargs):
        super().__init__(mlp, **kwargs)
        self.k = k

    def __call__(
        self,
        node_features: Union[mx.array, Tuple[mx.array, mx.array]],
        batch_indices: Optional[
            Union[mx.array, Tuple[Optional[mx.array], Optional[mx.array]]]
        ] = None,
    ) -> mx.array:
        """Builds the k-NN graph of the node features and applies EdgeConv.

        Args:
            node_features: Input node features of shape ``[num_nodes,
                node_features_dim]``, or a ``(src_features, dst_features)``
                tuple for bipartite graphs.
            batch_indices: Graph of each node, of shape ``[num_nodes]``, or a
                tuple of the batch vectors of the source and destination nodes.
                Default: ``None`` (a single graph).

        Returns:
            mx.array: The computed node embeddings, one row per destination node
        """
        if isinstance(node_features, tuple):
            src_features, dst_features = node_features
        else:
            src_features = dst_features = node_features
        if src_features.ndim != 2:
            raise ValueError("Static graphs are not supported in DynamicEdgeConv")

        if isinstance(batch_indices, tuple):
            src_batch, dst_batch = batch_indices
        else:
            src_batch = dst_batch = batch_indices

        # knn returns (dst, src) pairs: flip them into src -> dst edges.
        edge_index = knn(
            mx.stop_gradient(src_features),
            mx.stop_gradient(dst_features),
            self.k,
            src_batch,
            dst_batch,
        )[::-1]
        return super().__call__(edge_index, (src_features, dst_features))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mlp={self.mlp}, k={self.k})"
