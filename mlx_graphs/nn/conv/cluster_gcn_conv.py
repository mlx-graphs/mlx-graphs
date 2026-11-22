import mlx.core as mx

from mlx_graphs.nn.linear import Linear
from mlx_graphs.nn.message_passing import MessagePassing, fused_message_passing
from mlx_graphs.utils import scatter


@fused_message_passing
class ClusterGCNConv(MessagePassing):
    r"""Graph convolution layer from the `"Cluster-GCN: An Efficient Algorithm
    for Training Deep and Large Graph Convolutional Networks"
    <https://arxiv.org/abs/1905.07953>`_ paper.

    .. math::
        \mathbf{H} = \left( \mathbf{\hat{A}} + \lambda \cdot
        \textrm{diag}(\mathbf{\hat{A}}) \right) \mathbf{X} \mathbf{W}_1 +
        \mathbf{X} \mathbf{W}_2

    where :math:`\mathbf{\hat{A}} = {(\mathbf{D} + \mathbf{I})}^{-1}
    (\mathbf{A} + \mathbf{I})` is the row-normalized adjacency matrix with
    self-loops.

    Args:
        node_features_dim: Size of input node features
        out_features_dim: Size of output node embeddings
        diag_lambda: Diagonal enhancement value :math:`\lambda`. Default: ``0``
        add_self_loops: Whether to replace the self-loops of the graph by one
            self-loop per node. Default: ``True``
        bias: Whether to add a bias to the output. Default: ``True``
        **kwargs: Arguments of :class:`MessagePassing`

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import ClusterGCNConv

        edge_index = mx.array([[0, 1, 2, 3, 4], [1, 0, 3, 2, 0]])
        node_features = mx.random.normal((5, 16))

        conv = ClusterGCNConv(16, 32, diag_lambda=1.0)
        h = conv(edge_index, node_features)
        >>> h.shape
        (5, 32)

    References:
        [1] `Chiang et al. Cluster-GCN: An Efficient Algorithm for Training Deep
        and Large Graph Convolutional Networks.
        <https://arxiv.org/abs/1905.07953>`_
    """

    # what follows the aggregation expects the size of one aggregation (as in PyG)
    supports_concat_aggr = False

    def __init__(
        self,
        node_features_dim: int,
        out_features_dim: int,
        diag_lambda: float = 0.0,
        add_self_loops: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super(ClusterGCNConv, self).__init__(**kwargs)

        self.node_features_dim = node_features_dim
        self.out_features_dim = out_features_dim
        self.diag_lambda = diag_lambda
        self._add_self_loops = add_self_loops

        # Same names as PyG.
        self.lin_out = Linear(node_features_dim, out_features_dim, bias=bias)
        self.lin_root = Linear(node_features_dim, out_features_dim, bias=False)

    def __call__(self, edge_index: mx.array, node_features: mx.array) -> mx.array:
        """Computes the forward pass of ClusterGCNConv.

        Args:
            edge_index: Input edge index of shape `[2, num_edges]`
            node_features: Input node features

        Returns:
            mx.array: The computed node embeddings
        """
        num_nodes = node_features.shape[0]
        row, col = edge_index[0], edge_index[1]
        is_loop = row == col

        if self._add_self_loops:
            # Existing self-loops get a weight of 0 (instead of being removed,
            # which would need a data-dependent shape), and one self-loop per
            # node is appended.
            keep = (~is_loop).astype(node_features.dtype)
            deg = scatter(keep, col, num_nodes, "add") + 1
            deg_inv = 1.0 / deg
            weights = mx.concatenate(
                [keep * deg_inv[col], (1 + self.diag_lambda) * deg_inv]
            )
            loop = mx.arange(num_nodes, dtype=edge_index.dtype)
            edge_index = mx.concatenate([edge_index, mx.stack([loop, loop])], axis=1)
        else:
            deg = scatter(mx.ones(col.shape, node_features.dtype), col, num_nodes)
            deg_inv = 1.0 / mx.maximum(deg, 1)
            weights = deg_inv[col] * (1 + self.diag_lambda * is_loop)

        # Propagation and the weight matrix commute, so propagate the smaller
        # of the two feature sizes (the bias is added after propagation).
        project_first = self.node_features_dim > self.out_features_dim
        features = node_features
        if project_first:
            features = features @ self.lin_out.weight.T

        out = self.propagate(
            edge_index=edge_index,
            node_features=features,
            message_kwargs={"edge_weights": weights},
        )
        if not project_first:
            out = out @ self.lin_out.weight.T
        if "bias" in self.lin_out:
            out = out + self.lin_out.bias

        return out + self.lin_root(node_features)
