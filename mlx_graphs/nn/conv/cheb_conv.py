from typing import Literal, Optional, Tuple, Union

import mlx.core as mx

from mlx_graphs.nn.conv._norm import _norm_cache, laplacian
from mlx_graphs.nn.linear import Linear
from mlx_graphs.nn.message_passing import MessagePassing, fused_message_passing


@fused_message_passing
class ChebConv(MessagePassing):
    r"""Chebyshev spectral graph convolution layer from `"Convolutional Neural
    Networks on Graphs with Fast Localized Spectral Filtering"
    <https://arxiv.org/abs/1606.09375>`_ paper.

    .. math::
        \mathbf{H} = \sum_{k=1}^{K} \mathbf{Z}^{(k)} \mathbf{W}_k + \mathbf{b},

    where the :math:`\mathbf{Z}^{(k)}` are computed recursively by

    .. math::
        \mathbf{Z}^{(1)} &= \mathbf{X}

        \mathbf{Z}^{(2)} &= \mathbf{\hat{L}} \mathbf{X}

        \mathbf{Z}^{(k)} &= 2 \mathbf{\hat{L}} \mathbf{Z}^{(k-1)} -
        \mathbf{Z}^{(k-2)}

    and :math:`\mathbf{\hat{L}} = \frac{2 \mathbf{L}}{\lambda_{\max}} -
    \mathbf{I}` is the scaled graph Laplacian.

    Args:
        node_features_dim: Size of input node features
        out_features_dim: Size of output node embeddings
        K: Number of Chebyshev terms, i.e. the filter reaches ``K - 1`` hops
        normalization: Normalization of the graph Laplacian
            :math:`\mathbf{L}`, with :math:`\mathbf{D}` the (weighted) degree
            matrix (default: ``"sym"``):

            * ``None``: :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`
            * ``"sym"``: :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2}
              \mathbf{A} \mathbf{D}^{-1/2}`
            * ``"rw"``: :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1}
              \mathbf{A}`
        bias: Whether to add a bias to the output
        **kwargs: Arguments of :class:`MessagePassing`

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import ChebConv

        edge_index = mx.array([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
        node_features = mx.ones((5, 16))

        conv = ChebConv(16, 32, K=3)
        h = conv(edge_index, node_features)
        >>> h.shape
        (5, 32)

        # Largest eigenvalue of the Laplacian, if it is known
        h = conv(edge_index, node_features, lambda_max=1.8)
    """

    def __init__(
        self,
        node_features_dim: int,
        out_features_dim: int,
        K: int,
        normalization: Optional[Literal["sym", "rw"]] = "sym",
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super(ChebConv, self).__init__(**kwargs)

        if K < 1:
            raise ValueError(f"K must be at least 1, got {K}")
        if normalization not in (None, "sym", "rw"):
            raise ValueError(
                f"normalization must be None, 'sym' or 'rw', got {normalization!r}"
            )

        self.node_features_dim = node_features_dim
        self.out_features_dim = out_features_dim
        self.K = K
        self.normalization = normalization

        self.lins = [
            Linear(node_features_dim, out_features_dim, bias=False) for _ in range(K)
        ]
        if bias:
            self.bias = mx.zeros((out_features_dim,))

    def _scaled_laplacian(
        self,
        edge_index: mx.array,
        num_nodes: int,
        edge_weights: Optional[mx.array],
        batch: Optional[mx.array],
        lambda_max: Optional[Union[float, mx.array]],
    ) -> Tuple[mx.array, mx.array]:
        """Edge list form of ``2 L / lambda_max - I``."""
        num_edges = edge_index.shape[1]
        edge_index, weights = laplacian(
            edge_index, num_nodes, edge_weights, self.normalization
        )

        if lambda_max is None:
            lambda_max = 2.0 * weights.max()
        elif isinstance(lambda_max, mx.array) and lambda_max.size > 1:
            if batch is None:
                raise ValueError(
                    "`batch` is required when `lambda_max` has one value per graph"
                )
            lambda_max = lambda_max.reshape(-1)[batch[edge_index[0]]]

        weights = (2.0 * weights) / lambda_max
        weights = mx.where(
            mx.abs(weights) == float("inf"), mx.zeros_like(weights), weights
        )

        # Subtract the identity: the last `num_nodes` entries are the diagonal.
        identity = mx.concatenate(
            [
                mx.zeros((num_edges,), dtype=weights.dtype),
                mx.ones((num_nodes,), dtype=weights.dtype),
            ]
        )
        return edge_index, weights - identity

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_weights: Optional[mx.array] = None,
        batch: Optional[mx.array] = None,
        lambda_max: Optional[Union[float, mx.array]] = None,
    ) -> mx.array:
        r"""Computes the forward pass of ChebConv.

        Args:
            edge_index: Input edge index of shape `[2, num_edges]`
            node_features: Input node features
            edge_weights: Optional scalar weight per edge, of shape `[num_edges]`
                or `[num_edges, 1]`. Default: ``None``
            batch: Graph index of every node, of shape `[num_nodes]`. Only
                needed when ``lambda_max`` has one value per graph.
                Default: ``None``
            lambda_max: Largest eigenvalue of the graph Laplacian, either one
                value or an array with one value per graph of the batch. If
                ``None``, twice the largest entry of the Laplacian is used:
                2 for the ``"sym"`` and ``"rw"`` normalizations, twice the
                largest degree without normalization. Default: ``None``

        Returns:
            mx.array: The computed node embeddings
        """
        num_nodes = node_features.shape[0]
        if edge_weights is None and (
            lambda_max is None or isinstance(lambda_max, (int, float))
        ):
            # The scaled Laplacian of an unweighted graph only depends on the
            # graph: computed once per edge_index array.
            graph = edge_index
            edge_index, norm = _norm_cache.get(
                graph,
                ("cheb", num_nodes, self.normalization, lambda_max),
                lambda: self._scaled_laplacian(
                    graph, num_nodes, None, batch, lambda_max
                ),
            )
        else:
            edge_index, norm = self._scaled_laplacian(
                edge_index, num_nodes, edge_weights, batch, lambda_max
            )

        previous = node_features
        out = self.lins[0](previous)

        if len(self.lins) > 1:
            current = self.propagate(
                edge_index=edge_index,
                node_features=node_features,
                message_kwargs={"edge_weights": norm},
            )
            out = out + self.lins[1](current)

            for lin in self.lins[2:]:
                following = self.propagate(
                    edge_index=edge_index,
                    node_features=current,
                    message_kwargs={"edge_weights": norm},
                )
                following = 2.0 * following - previous
                out = out + lin(following)
                previous, current = current, following

        if "bias" in self:
            out = out + self.bias

        return out
