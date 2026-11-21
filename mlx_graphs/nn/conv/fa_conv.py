from typing import Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.nn import Linear  # PyTorch's default initialization, as in PyG

from mlx_graphs.nn.conv._norm import cached_gcn_norm
from mlx_graphs.nn.message_passing import MessagePassing, fused_message_passing


@fused_message_passing
class FAConv(MessagePassing):
    r"""The Frequency Adaptive Graph Convolution operator from the
    `"Beyond Low-Frequency Information in Graph Convolutional Networks"
    <https://arxiv.org/abs/2101.00797>`_ paper.

    .. math::
        \mathbf{x}^{\prime}_i= \epsilon \cdot \mathbf{x}^{(0)}_i +
        \sum_{j \in \mathcal{N}(i)} \frac{\alpha_{i,j}}{\sqrt{d_i d_j}}
        \mathbf{x}_{j}

    where :math:`\mathbf{x}^{(0)}_i` and :math:`d_i` denote the initial feature
    representation and node degree of node :math:`i`, respectively.
    The attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \mathbf{\alpha}_{i,j} = \textrm{tanh}(\mathbf{a}^{\top}[\mathbf{x}_i,
        \mathbf{x}_j])

    based on the trainable parameter vector :math:`\mathbf{a}`.

    The normalization :math:`1 / \sqrt{d_i d_j}` of a graph without edge
    weights is cached per ``edge_index`` array (see
    :func:`~mlx_graphs.nn.clear_norm_cache`), which replaces PyG's ``cached``
    argument.

    Args:
        node_features_dim: Size of input (and output) node features
        eps: :math:`\epsilon`-value weighting the initial features.
            Default: ``0.1``
        dropout: Dropout probability of the normalized attention coefficients.
            Default: ``0.0``
        add_self_loops: Whether to add self-loops to the input graph when
            ``normalize`` is ``True``. Default: ``True``
        normalize: Whether to add self-loops (if ``add_self_loops``) and compute
            the symmetric normalization on the fly. If ``False``, normalized
            ``edge_weights`` must be passed to the forward pass.
            Default: ``True``

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import FAConv

        conv = FAConv(16, eps=0.1)
        edge_index = mx.array([[0, 1, 2, 3, 4], [0, 0, 1, 1, 3]])
        x_0 = mx.random.uniform(low=0, high=1, shape=(5, 16))

        h = conv(edge_index, x_0, x_0)

        >>> h.shape
        (5, 16)

    References:
        `Bo et al. Beyond Low-Frequency Information in Graph Convolutional
        Networks <https://arxiv.org/abs/2101.00797>`_
    """

    # what follows the aggregation expects the size of one aggregation (as in PyG)
    supports_concat_aggr = False

    def __init__(
        self,
        node_features_dim: int,
        eps: float = 0.1,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        normalize: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)

        self.node_features_dim = node_features_dim
        self.out_features_dim = node_features_dim
        self.eps = eps
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self.att_l = Linear(node_features_dim, 1, bias=False)
        self.att_r = Linear(node_features_dim, 1, bias=False)
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        initial_node_features: mx.array,
        edge_weights: Optional[mx.array] = None,
        return_attention_weights: bool = False,
    ) -> Union[mx.array, Tuple[mx.array, Tuple[mx.array, mx.array]]]:
        """Computes the forward pass of FAConv.

        Args:
            edge_index: Input edge index of shape `[2, num_edges]`
            node_features: Input node features
            initial_node_features: Initial node features :math:`\\mathbf{x}^{(0)}`
                (e.g. the output of the first layer of the model)
            edge_weights: Normalized edge weights, required if
                ``normalize=False`` and not allowed otherwise. Default: ``None``
            return_attention_weights: If ``True``, also returns the tuple
                ``(edge_index, attention_weights)``, where ``edge_index``
                includes the added self-loops. Default: ``False``

        Returns:
            The computed node embeddings, and the attention weights if
            ``return_attention_weights`` is ``True``.
        """
        num_nodes = node_features.shape[0]
        if self.normalize:
            if edge_weights is not None:
                raise ValueError(
                    "FAConv computes the edge weights when `normalize=True`, "
                    "`edge_weights` must be None."
                )
            edge_index, edge_weights = cached_gcn_norm(
                edge_index, num_nodes, None, self.add_self_loops
            )
        elif edge_weights is None:
            raise ValueError("FAConv needs `edge_weights` when `normalize=False`.")
        edge_weights = edge_weights.reshape(-1)

        alpha_l = self.att_l(node_features)[:, 0]
        alpha_r = self.att_r(node_features)[:, 0]
        alpha = mx.tanh(alpha_l[edge_index[0]] + alpha_r[edge_index[1]])

        weights = alpha
        if "dropout" in self:
            weights = self.dropout(weights)

        out = self.propagate(
            edge_index=edge_index,
            node_features=node_features,
            message_kwargs={"edge_weights": weights * edge_weights},
        )

        if self.eps != 0.0:
            out = out + self.eps * initial_node_features

        if return_attention_weights:
            return out, (edge_index, alpha)
        return out
