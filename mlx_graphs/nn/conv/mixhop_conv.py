from typing import List, Optional

import mlx.core as mx
from mlx.nn import Identity, Linear  # PyTorch's default initialization, as in PyG

from mlx_graphs.nn.conv._norm import cached_gcn_norm
from mlx_graphs.nn.message_passing import MessagePassing, fused_message_passing


@fused_message_passing
class MixHopConv(MessagePassing):
    r"""The Mix-Hop graph convolutional operator from the `"MixHop: Higher-Order
    Graph Convolutional Architectures via Sparsified Neighborhood Mixing"
    <https://arxiv.org/abs/1905.00067>`_ paper.

    .. math::
        \mathbf{X}^{\prime}={\Bigg\Vert}_{p\in P}
        {\left( \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \right)}^p \mathbf{X} \mathbf{\Theta}_p,

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
    The output concatenates one block of ``out_features_dim`` features per
    power, so it has ``len(powers) * out_features_dim`` features.

    Args:
        node_features_dim: Size of input node features
        out_features_dim: Size of the output of every power
        powers: Powers of the adjacency matrix to use. Default: ``[0, 1, 2]``
        add_self_loops: Whether to add self-loops to the input graph.
            Default: ``True``
        bias: Whether to add a learnable bias to the output. Default: ``True``

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import MixHopConv

        conv = MixHopConv(16, 32, powers=[0, 1, 2])
        edge_index = mx.array([[0, 1, 2, 3, 4], [0, 0, 1, 1, 3]])
        node_features = mx.random.uniform(low=0, high=1, shape=(5, 16))

        h = conv(edge_index, node_features)

        >>> h.shape
        (5, 96)

    References:
        `Abu-El-Haija et al. MixHop: Higher-Order Graph Convolutional
        Architectures via Sparsified Neighborhood Mixing
        <https://arxiv.org/abs/1905.00067>`_
    """

    # what follows the aggregation expects the size of one aggregation (as in PyG)
    supports_concat_aggr = False

    def __init__(
        self,
        node_features_dim: int,
        out_features_dim: int,
        powers: Optional[List[int]] = None,
        add_self_loops: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)

        if powers is None:
            powers = [0, 1, 2]
        if len(powers) == 0 or min(powers) < 0:
            raise ValueError(f"powers must be non-negative integers, got {powers}")

        self.node_features_dim = node_features_dim
        self.out_features_dim = out_features_dim
        self.powers = list(powers)
        self.add_self_loops = add_self_loops

        # One linear layer per power, at the index of the power (``Identity``
        # for the unused ones), as in PyG. A list, not a dict keyed by the
        # power: MLX turns digit keys back into list indices when loading
        # weights (``load_weights`` / ``tree_unflatten``).
        self.lins = [
            Linear(node_features_dim, out_features_dim, bias=False)
            if p in self.powers
            else Identity()
            for p in range(max(self.powers) + 1)
        ]
        if bias:
            self.bias = mx.zeros((len(powers) * out_features_dim,))

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_weights: Optional[mx.array] = None,
    ) -> mx.array:
        """Computes the forward pass of MixHopConv.

        Args:
            edge_index: Input edge index of shape `[2, num_edges]`
            node_features: Input node features
            edge_weights: Optional weight of each edge. Default: ``None``

        Returns:
            mx.array: The computed node embeddings, of shape
            `[num_nodes, len(powers) * out_features_dim]`
        """
        edge_index, norm = cached_gcn_norm(
            edge_index, node_features.shape[0], edge_weights, self.add_self_loops
        )

        outs = {}
        x = node_features
        for p in range(max(self.powers) + 1):
            if p > 0:
                x = self.propagate(
                    edge_index=edge_index,
                    node_features=x,
                    message_kwargs={"edge_weights": norm},
                )
            if p in self.powers:
                outs[p] = self.lins[p](x)

        out = mx.concatenate([outs[p] for p in self.powers], axis=-1)
        if "bias" in self:
            out = out + self.bias
        return out
