from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.nn.linear import Linear
from mlx_graphs.nn.message_passing import MessagePassing


class GINConv(MessagePassing):
    r"""Graph Isomorphism Network convolution layer from `"How Powerful are
    Graph Neural Networks?" <https://arxiv.org/abs/1810.00826>`_ paper.

    .. math::
        \mathbf{h}_i = \text{MLP} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \right)

    where :math:`\mathbf{x}_i` and  :math:`\mathbf{h}_i` represent the input features
    and output embeddings of node :math:`i`, respectively. :math:`\text{MLP}` denotes
    a custom neural network provided by the user and :math:`\epsilon` is an epsilon
    value either fixed or learned.

    Setting ``edge_features_dim`` produces a `GINEConv` model, where `edge_features`
    are expected to be passed in the forward. In this case, edge features are first
    projected onto the same dimension as node embeddings and are summed, then passed to
    a relu activation.
    To use `GINEConv`, setting `node_features_dim` is also required.

    Args:
        mlp: Callable :class:`mlx.core.nn.Module` applied on the final node embeddings
        eps: Initial value of the :math:`\epsilon` term. Default: ``0``
        learn_eps: Whether to learn :math:`\epsilon` or not. Default ``False``
        edge_features_dim: Size of the edge features passed in the GINE layer
        node_features_dim: Size of the node features (only required if GINE is used)

    Example:

    .. code-block:: python

        import mlx.core as mx
        import mlx.nn as nn
        from mlx_graphs.nn import GINConv

        node_feat_dim = 16
        edge_feat_dim = 10
        out_feat_dim = 32

        mlp = nn.Sequential(
            nn.Linear(node_feat_dim, node_feat_dim * 2),
            nn.ReLU(),
            nn.Linear(node_feat_dim * 2, out_feat_dim),
        )

        # original GINConv for node features

        conv = GINConv(mlp)

        edge_index = mx.array([[0, 1, 2, 3, 4], [0, 0, 1, 1, 3]])
        node_features = mx.random.uniform(low=0, high=1, shape=(5, 16))

        >>> conv(edge_index, node_features)
        array([[-0.536501, 0.154826, 0.745569, ..., 0.31547, -0.0962588, -0.108504],
            [-0.415889, -0.0498145, 0.597379, ..., 0.194553, -0.251498, -0.207561],
            [-0.119966, -0.0159533, 0.276559, ..., 0.0258303, -0.194533, -0.15515],
            [-0.21477, -0.169684, 0.485867, ..., 0.0194768, -0.145761, -0.139433],
            [-0.133289, -0.0279559, 0.358095, ..., -0.0443346, -0.11571, -0.114396]],
            dtype=float32)

        # GINEConv including edge features:

        conv = GINConv(mlp, edge_features_dim=edge_feat_dim,
            node_features_dim=node_feat_dim)
        edge_features = mx.random.uniform(low=0, high=1, shape=(5, edge_feat_dim))

        >>> conv(edge_index, node_features, edge_features)
        array([[-0.175581, 0.67481, -0.260592, ..., -1.13234, -0.631736, 0.572239],
            [0.0536669, 0.496115, -0.319334, ..., -1.165, -0.573817, 0.495315],
            [-0.0505168, 0.102068, 0.0221924, ..., -0.516901, -0.331266, 0.317491],
            [-0.00632942, 0.433597, -0.162906, ..., -0.957552, -0.41922, 0.670711],
            [-0.119726, 0.173545, 0.0951687, ..., -0.577839, -0.244039, 0.399055]],
            dtype=float32)
    """

    def __init__(
        self,
        mlp: nn.Module,
        eps: float = 0.0,
        learn_eps: bool = False,
        node_features_dim: Optional[int] = None,
        edge_features_dim: Optional[int] = None,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super(GINConv, self).__init__(**kwargs)

        self.mlp = mlp
        self.eps = mx.array([eps]) if learn_eps else eps

        if edge_features_dim is not None:
            self.edge_projection = Linear(
                edge_features_dim, node_features_dim, bias=True
            )

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_features: Optional[mx.array] = None,
        edge_weights: Optional[mx.array] = None,
    ) -> mx.array:
        """Computes the forward pass of GINConv.

        Args:
            edge_index: Input edge index of shape `[2, num_edges]`
            node_features: Input node features
            edge_features: Input edge features. Defautl: ``None``
            edge_weights: Edge weights leveraged in message passing. Default: ``None``

        Returns:
            The computed node embeddings
        """
        if isinstance(node_features, mx.array):
            node_features = (node_features, node_features)

        dst_features = node_features[1]

        aggr_features = self.propagate(
            edge_index=edge_index,
            node_features=node_features,
            message_kwargs={
                "edge_weights": edge_weights,
                "edge_features": edge_features,
            },
        )
        node_features = self.mlp(aggr_features + (1 + self.eps) * dst_features)

        return node_features

    def message(
        self, src_features: mx.array, dst_features: mx.array, **kwargs
    ) -> mx.array:
        edge_features = kwargs.get("edge_features", None)
        if edge_features is not None:
            # GINEConv
            if "edge_projection" in self:
                edge_emb = self.edge_projection(edge_features)
            return nn.relu(src_features + edge_emb)

        # GINConv
        return super(self.__class__, self).message(src_features, dst_features, **kwargs)
