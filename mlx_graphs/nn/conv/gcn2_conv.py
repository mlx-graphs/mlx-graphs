from math import log
from typing import Optional, Tuple

import mlx.core as mx

from mlx_graphs.nn.conv._norm import cached_gcn_norm
from mlx_graphs.nn.linear import glorot
from mlx_graphs.nn.message_passing import MessagePassing, fused_message_passing


@fused_message_passing
class GCN2Conv(MessagePassing):
    r"""Graph convolution with initial residual connections and identity mapping
    (GCNII) from the `"Simple and Deep Graph Convolutional Networks"
    <https://arxiv.org/abs/2007.02133>`_ paper.

    .. math::
        \mathbf{H}^{(\ell + 1)} = \left( (1 - \alpha) \hat{\mathbf{P}}
        \mathbf{H}^{(\ell)} + \alpha \mathbf{H}^{(0)} \right) \left( (1 -
        \beta) \mathbf{I} + \beta \mathbf{W} \right)

    with :math:`\hat{\mathbf{P}} = \hat{\mathbf{D}}^{-1/2} \hat{\mathbf{A}}
    \hat{\mathbf{D}}^{-1/2}`, where :math:`\hat{\mathbf{A}} = \mathbf{A} +
    \mathbf{I}` is the (weighted) adjacency matrix with self-loops,
    :math:`\mathbf{H}^{(0)}` the initial node features, and :math:`\beta =
    \log(\theta / \ell + 1)` (``beta = 1`` if ``theta`` and ``layer`` are not
    given). With ``shared_weights=False``, the initial features get their own
    weight matrix:

    .. math::
        \mathbf{H}^{(\ell + 1)} = (1 - \alpha) \hat{\mathbf{P}}
        \mathbf{H}^{(\ell)} \left( (1 - \beta) \mathbf{I} + \beta
        \mathbf{W}_1 \right) + \alpha \mathbf{H}^{(0)} \left( (1 - \beta)
        \mathbf{I} + \beta \mathbf{W}_2 \right)

    Args:
        node_features_dim: Size of the input and output node features
        alpha: Strength :math:`\alpha` of the initial residual connection
        theta: Hyperparameter :math:`\theta` used to compute the strength of the
            identity mapping. Default: ``None``
        layer: Index :math:`\ell` of the layer (starting from 1), used to
            compute the strength of the identity mapping. Default: ``None``
        shared_weights: If ``False``, uses a separate weight matrix for the
            initial features. Default: ``True``
        cached: If ``True``, the normalized graph is computed on the first call
            and reused afterwards (transductive learning only). Call
            :meth:`clear_cache` to drop it. Default: ``False``
        add_self_loops: Whether to add a self-loop to every node that has none.
            Default: ``True``
        normalize: Whether to apply the symmetric degree normalization.
            Default: ``True``
        **kwargs: Arguments of :class:`MessagePassing`

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import GCN2Conv

        edge_index = mx.array([[0, 1, 2, 3, 4], [0, 0, 1, 1, 3]])
        initial_features = mx.random.normal((5, 16))

        conv = GCN2Conv(16, alpha=0.1, theta=0.5, layer=1)
        h = conv(edge_index, initial_features, initial_features)
        >>> h.shape
        (5, 16)

    References:
        [1] `Chen et al. Simple and Deep Graph Convolutional Networks.
        <https://arxiv.org/abs/2007.02133>`_
    """

    # what follows the aggregation expects the size of one aggregation (as in PyG)
    supports_concat_aggr = False

    def __init__(
        self,
        node_features_dim: int,
        alpha: float,
        theta: Optional[float] = None,
        layer: Optional[int] = None,
        shared_weights: bool = True,
        cached: bool = False,
        add_self_loops: bool = True,
        normalize: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super(GCN2Conv, self).__init__(**kwargs)

        self.node_features_dim = node_features_dim
        self.alpha = alpha
        self.beta = 1.0
        if theta is not None or layer is not None:
            if theta is None or layer is None:
                raise ValueError("`theta` and `layer` must be given together.")
            self.beta = log(theta / layer + 1)
        self.cached = cached
        self.normalize = normalize
        self._add_self_loops = add_self_loops
        self._cached_graph: Optional[Tuple[mx.array, mx.array]] = None

        # Applied as `x @ weight`, as in PyG.
        self.weight1 = glorot((node_features_dim, node_features_dim))
        if not shared_weights:
            self.weight2 = glorot((node_features_dim, node_features_dim))

    def clear_cache(self) -> None:
        """Drops the normalized graph stored when ``cached=True``."""
        self._cached_graph = None

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        initial_node_features: mx.array,
        edge_weights: Optional[mx.array] = None,
    ) -> mx.array:
        """Computes the forward pass of GCN2Conv.

        Args:
            edge_index: Input edge index of shape `[2, num_edges]`
            node_features: Node features of the previous layer
                :math:`\\mathbf{H}^{(\\ell)}`
            initial_node_features: Initial node features :math:`\\mathbf{H}^{(0)}`
                (usually the output of the first projection of the model)
            edge_weights: Optional scalar weight per edge, of shape `[num_edges]`
                or `[num_edges, 1]`. Default: ``None``

        Returns:
            mx.array: The computed node embeddings
        """
        if self.normalize:
            if self._cached_graph is not None:
                edge_index, edge_weights = self._cached_graph
            else:
                edge_index, edge_weights = cached_gcn_norm(
                    edge_index,
                    node_features.shape[0],
                    edge_weights,
                    add_self_loops=self._add_self_loops,
                )
                if self.cached:
                    self._cached_graph = (edge_index, edge_weights)

        h = self.propagate(
            edge_index=edge_index,
            node_features=node_features,
            message_kwargs={"edge_weights": edge_weights},
        )
        h = (1 - self.alpha) * h
        h_0 = self.alpha * initial_node_features[: h.shape[0]]

        if "weight2" not in self:
            h = h + h_0
            return (1 - self.beta) * h + self.beta * (h @ self.weight1)

        out = (1 - self.beta) * h + self.beta * (h @ self.weight1)
        return out + (1 - self.beta) * h_0 + self.beta * (h_0 @ self.weight2)
