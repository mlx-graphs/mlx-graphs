import math
from typing import Literal, Optional, Tuple, Union

import mlx.core as mx
from mlx.nn import Linear  # PyTorch's default initialization, as in PyG

from mlx_graphs.nn.conv._norm import attention_leaky_relu
from mlx_graphs.nn.linear import glorot
from mlx_graphs.nn.message_passing import MessagePassing
from mlx_graphs.utils import scatter
from mlx_graphs.utils.array_ops import p_norm


class GeneralConv(MessagePassing):
    r"""A general GNN layer adapted from the `"Design Space for Graph Neural
    Networks" <https://arxiv.org/abs/2011.08843>`_ paper.

    Messages are linear projections of the source features (plus, optionally,
    of the destination features and of the edge features), computed for
    ``heads`` independent ensembles that are averaged:

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_{\textrm{self}} \mathbf{x}_i +
        \frac{1}{H} \sum_{h=1}^{H} \bigoplus_{j \in \mathcal{N}(i)}
        \alpha^{h}_{i,j} \left( \mathbf{W}^{h} \mathbf{x}_j +
        \mathbf{W}_e^{h} \mathbf{e}_{j,i} \right)

    where :math:`\alpha^{h}_{i,j} = 1` without attention, and otherwise a
    softmax over the neighbors of :math:`i` of either an additive score
    (``attention_type="additive"``) or a scaled dot product between the
    messages in both directions (``attention_type="dot_product"``), followed by
    a LeakyReLU with slope 0.2. :math:`\mathbf{W}_{\textrm{self}}` is the
    identity if ``skip_linear=False`` and the input and output sizes match.

    The layer also works on bipartite graphs: pass ``node_features`` as a
    ``(src_features, dst_features)`` tuple and ``node_features_dim`` as a
    ``(src_dim, dst_dim)`` tuple. The output has one row per destination node.

    Args:
        node_features_dim: Size of input node features, or a ``(src_dim, dst_dim)``
            tuple for bipartite graphs
        out_features_dim: Size of output node embeddings
        edge_features_dim: Size of edge features. Default: ``None``
        aggr: Aggregation of the messages (``"add"``, ``"mean"``, ``"max"``,
            ``"min"``). Default: ``"add"``
        skip_linear: Whether to apply a linear layer in the skip connection
            even if the input and output sizes match. Default: ``False``
        directed_msg: If ``False``, messages also include a projection of the
            destination node features (bi-directed messages). Default: ``True``
        heads: Number of message passing ensembles (attention heads if
            ``attention=True``). Default: ``1``
        attention: Whether to weigh messages with attention. Default: ``False``
        attention_type: ``"additive"`` or ``"dot_product"``.
            Default: ``"additive"``
        l2_normalize: Whether to :math:`\ell_2`-normalize the output features.
            Default: ``False``
        bias: Whether the linear layers use a bias. Default: ``True``

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import GeneralConv

        conv = GeneralConv(16, 32, heads=2, attention=True)
        edge_index = mx.array([[0, 1, 2, 3, 4], [0, 0, 1, 1, 3]])
        node_features = mx.random.uniform(low=0, high=1, shape=(5, 16))

        h = conv(edge_index, node_features)

        >>> h.shape
        (5, 32)

    References:
        `You et al. Design Space for Graph Neural Networks
        <https://arxiv.org/abs/2011.08843>`_
    """

    def __init__(
        self,
        node_features_dim: Union[int, Tuple[int, int]],
        out_features_dim: int,
        edge_features_dim: Optional[int] = None,
        aggr: Literal["add", "mean", "max", "min"] = "add",
        skip_linear: bool = False,
        directed_msg: bool = True,
        heads: int = 1,
        attention: bool = False,
        attention_type: Literal["additive", "dot_product"] = "additive",
        l2_normalize: bool = False,
        bias: bool = True,
    ):
        super().__init__(aggr=aggr)
        # the messages are reduced with `scatter` directly
        if not isinstance(self.aggr, str) or "aggr_module" in self:
            raise ValueError(
                "GeneralConv only supports a single scatter aggregation "
                "('add', 'mean', 'max', 'min', ...), not a list or an "
                f"Aggregation module (got {self.aggr!r})."
            )
        if attention and attention_type not in ("additive", "dot_product"):
            raise ValueError(f"Attention type '{attention_type}' not supported")

        if isinstance(node_features_dim, int):
            src_dim = dst_dim = node_features_dim
        else:
            src_dim, dst_dim = node_features_dim

        self.node_features_dim = node_features_dim
        self.out_features_dim = out_features_dim
        self.edge_features_dim = edge_features_dim
        self.directed_msg = directed_msg
        self.heads = heads
        self.attention = attention
        self.attention_type = attention_type
        self.l2_normalize = l2_normalize

        self.lin_msg = Linear(src_dim, out_features_dim * heads, bias=bias)
        if not directed_msg:
            self.lin_msg_i = Linear(src_dim, out_features_dim * heads, bias=bias)

        # As in PyG, a tuple of input sizes always gets a skip projection.
        if skip_linear or node_features_dim != out_features_dim:
            self.lin_self = Linear(dst_dim, out_features_dim, bias=bias)

        if edge_features_dim is not None:
            self.lin_edge = Linear(
                edge_features_dim, out_features_dim * heads, bias=bias
            )

        if attention and attention_type == "additive":
            self.att_msg = glorot((1, heads, out_features_dim))

    def _message_basic(
        self, x_i: mx.array, x_j: mx.array, edge_features: Optional[mx.array]
    ) -> mx.array:
        out = self.lin_msg(x_j)
        if not self.directed_msg:
            out = out + self.lin_msg_i(x_i)
        if edge_features is not None:
            out = out + self.lin_edge(edge_features)
        return out.reshape(-1, self.heads, self.out_features_dim)

    def __call__(
        self,
        edge_index: mx.array,
        node_features: Union[mx.array, Tuple[mx.array, mx.array]],
        edge_features: Optional[mx.array] = None,
    ) -> mx.array:
        """Computes the forward pass of GeneralConv.

        Args:
            edge_index: Input edge index of shape `[2, num_edges]`
            node_features: Input node features, or a ``(src_features,
                dst_features)`` tuple for bipartite graphs
            edge_features: Optional edge features of shape
                `[num_edges, edge_features_dim]`

        Returns:
            mx.array: The computed node embeddings, one row per destination node
        """
        if isinstance(node_features, tuple):
            x_src, x_dst = node_features
        else:
            x_src = x_dst = node_features

        if edge_features is not None:
            if "lin_edge" not in self:
                raise ValueError(
                    "GeneralConv needs `edge_features_dim` to use edge features."
                )
            if edge_features.ndim == 1:
                edge_features = edge_features.reshape(-1, 1)

        src_idx, dst_idx = edge_index[0], edge_index[1]
        num_nodes = x_dst.shape[0]
        x_j, x_i = x_src[src_idx], x_dst[dst_idx]

        messages = self._message_basic(x_i, x_j, edge_features)  # [E, H, C]

        if self.attention:
            if self.attention_type == "dot_product":
                reverse = self._message_basic(x_j, x_i, edge_features)
                alpha = (reverse * messages).sum(-1) / math.sqrt(self.out_features_dim)
            else:
                alpha = (messages * self.att_msg).sum(-1)
            alpha = attention_leaky_relu(alpha, 0.2)
            alpha = scatter(alpha, dst_idx, num_nodes, "softmax")
            messages = messages * alpha[..., None]

        out = scatter(messages, dst_idx, num_nodes, self.aggr)  # [N, H, C]
        out = out.mean(axis=1)

        out = out + (self.lin_self(x_dst) if "lin_self" in self else x_dst)

        if self.l2_normalize:
            norm = p_norm(out, axis=-1, keepdims=True)
            out = out / mx.maximum(norm, 1e-12)
        return out
