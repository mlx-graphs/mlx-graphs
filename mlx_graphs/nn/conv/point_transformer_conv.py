from typing import Callable, Optional, Tuple, Union

import mlx.core as mx
from mlx.nn import Linear  # PyG's default (kaiming uniform) initialization

from mlx_graphs.nn.conv.point_conv import Pair, _pair, _run_mlp, _set_self_loops
from mlx_graphs.nn.message_passing import MessagePassing
from mlx_graphs.utils.scatter import scatter


class PointTransformerConv(MessagePassing):
    r"""The Point Transformer layer from the `"Point Transformer"
    <https://arxiv.org/abs/2012.09164>`_ paper.

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i) \cup \{ i \}}
        \alpha_{i,j} \left(\mathbf{W}_3 \mathbf{x}_j + \delta_{ij} \right),

    where the attention coefficients :math:`\alpha_{i,j}` and the positional
    embedding :math:`\delta_{ij}` are

    .. math::
        \alpha_{i,j}= \textrm{softmax} \left( \gamma_\mathbf{\Theta}
        (\mathbf{W}_1 \mathbf{x}_i - \mathbf{W}_2 \mathbf{x}_j +
        \delta_{i,j}) \right)
        \qquad
        \delta_{i,j}= h_{\mathbf{\Theta}}(\mathbf{p}_i - \mathbf{p}_j).

    The softmax is taken over the neighbors of each node, separately for each
    channel. :math:`\gamma_\mathbf{\Theta}` (``attn_nn``) and
    :math:`h_\mathbf{\Theta}` (``pos_nn``) are neural networks.

    Args:
        node_features_dim: Size of the input features, or a ``(src_dim,
            dst_dim)`` tuple for bipartite graphs.
        out_features_dim: Size of the output features.
        pos_nn: Network :math:`h_\mathbf{\Theta}` mapping the relative
            positions of shape ``[num_edges, 3]`` to ``[num_edges,
            out_features_dim]``. Default: ``None`` (a linear layer).
        attn_nn: Network :math:`\gamma_\mathbf{\Theta}` mapping attention
            logits of shape ``[num_edges, out_features_dim]`` to ``[num_edges,
            out_features_dim]``. Default: ``None``
        add_self_loops: If ``True``, removes the self-loops of the graph and
            adds one for each node. Default: ``True``
        **kwargs: Arguments of :class:`MessagePassing` (``aggr="add"`` by
            default).

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import PointTransformerConv
        from mlx_graphs.utils import knn_graph

        pos = mx.random.normal((100, 3))
        x = mx.random.normal((100, 16))
        conv = PointTransformerConv(16, 32)
        >>> conv(knn_graph(pos, k=8), x, pos).shape
        (100, 32)
    """

    def __init__(
        self,
        node_features_dim: Union[int, Tuple[int, int]],
        out_features_dim: int,
        pos_nn: Optional[Callable] = None,
        attn_nn: Optional[Callable] = None,
        add_self_loops: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)
        self.node_features_dim = node_features_dim
        self.out_features_dim = out_features_dim
        self.add_self_loops = add_self_loops
        src_dim, dst_dim = _pair(node_features_dim)
        self.pos_nn = pos_nn if pos_nn is not None else Linear(3, out_features_dim)
        self.attn_nn = attn_nn
        self.lin = Linear(src_dim, out_features_dim, bias=False)
        self.lin_src = Linear(src_dim, out_features_dim, bias=False)
        self.lin_dst = Linear(dst_dim, out_features_dim, bias=False)

    def __call__(
        self,
        edge_index: mx.array,
        node_features: Pair,
        pos: Pair,
    ) -> mx.array:
        """Computes the forward pass of PointTransformerConv.

        Args:
            edge_index: Edges of shape ``[2, num_edges]``, from the neighbors
                (sources) to the center points (destinations).
            node_features: Features of the points, or a ``(src_features,
                dst_features)`` tuple for bipartite graphs.
            pos: Positions of the points, of shape ``[N, 3]``, or a
                ``(src_pos, dst_pos)`` tuple.

        Returns:
            The features of the destination points.
        """
        x_src, x_dst = _pair(node_features)
        pos_src, pos_dst = _pair(pos)
        if self.add_self_loops:
            edge_index = _set_self_loops(
                edge_index, min(pos_src.shape[0], pos_dst.shape[0])
            )
        src, dst = edge_index[0], edge_index[1]
        num_dst = x_dst.shape[0]

        delta = _run_mlp(self.pos_nn, pos_dst[dst] - pos_src[src])
        alpha = self.lin_dst(x_dst)[dst] - self.lin_src(x_src)[src] + delta
        alpha = _run_mlp(self.attn_nn, alpha)
        alpha = scatter(alpha, dst, num_dst, "softmax")
        messages = alpha * (self.lin(x_src)[src] + delta)
        self.num_nodes = num_dst
        return self.update_nodes(self.aggregate(messages, dst))
