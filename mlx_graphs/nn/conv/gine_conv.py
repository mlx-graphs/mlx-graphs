from typing import Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.nn.conv.gin_conv import GINConv


def _infer_input_dim(mlp: nn.Module) -> Optional[int]:
    """Input size of the first linear layer of ``mlp``, if it can be found."""
    if isinstance(mlp, nn.Sequential) and len(mlp.layers) > 0:
        mlp = mlp.layers[0]
    if isinstance(mlp, nn.Linear):
        return mlp.weight.shape[-1]
    return None


class GINEConv(GINConv):
    r"""Graph Isomorphism Network convolution with edge features, from the
    `"Strategies for Pre-training Graph Neural Networks"
    <https://arxiv.org/abs/1905.12265>`_ paper.

    .. math::
        \mathbf{h}_i = \text{MLP} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathrm{ReLU}
        ( \mathbf{x}_j + \mathbf{e}_{j,i} ) \right)

    where :math:`\mathbf{e}_{j,i}` are the features of the edge from node
    :math:`j` to node :math:`i`. If ``edge_features_dim`` is set, edge features
    are first projected with a linear layer onto the size of the node features;
    otherwise both must have the same size.

    This is :class:`GINConv` with edge features made mandatory, mirroring PyG's
    ``GINEConv``.

    Args:
        mlp: Callable :class:`mlx.nn.Module` applied on the final node embeddings
        eps: Initial value of the :math:`\epsilon` term. Default: ``0``
        learn_eps: Whether to learn :math:`\epsilon` or not. Default ``False``
        edge_features_dim: Size of the edge features. If set, edge features are
            projected onto the size of the node features. Default: ``None``
        node_features_dim: Size of the node features, only used when
            ``edge_features_dim`` is set. If ``None``, it is inferred from the
            first linear layer of ``mlp``. Default: ``None``
        **kwargs: Arguments of :class:`MessagePassing`, e.g. ``aggr``
            (default: ``"add"``)

    Example:

    .. code-block:: python

        import mlx.core as mx
        import mlx.nn as nn
        from mlx_graphs.nn import GINEConv

        mlp = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 32))
        conv = GINEConv(mlp, edge_features_dim=4)

        edge_index = mx.array([[0, 1, 2, 3, 4], [0, 0, 1, 1, 3]])
        node_features = mx.random.normal((5, 16))
        edge_features = mx.random.normal((5, 4))

        h = conv(edge_index, node_features, edge_features)
        >>> h.shape
        (5, 32)

    References:
        [1] `Hu et al. Strategies for Pre-training Graph Neural Networks.
        <https://arxiv.org/abs/1905.12265>`_
    """

    # what follows the aggregation expects the size of one aggregation (as in PyG)
    supports_concat_aggr = False

    def __init__(
        self,
        mlp: nn.Module,
        eps: float = 0.0,
        learn_eps: bool = False,
        edge_features_dim: Optional[int] = None,
        node_features_dim: Optional[int] = None,
        **kwargs,
    ):
        if edge_features_dim is not None and node_features_dim is None:
            node_features_dim = _infer_input_dim(mlp)
            if node_features_dim is None:
                raise ValueError(
                    "Could not infer the node features size from `mlp`, "
                    "set `node_features_dim`."
                )
        super(GINEConv, self).__init__(
            mlp,
            eps=eps,
            learn_eps=learn_eps,
            node_features_dim=node_features_dim,
            edge_features_dim=edge_features_dim,
            **kwargs,
        )
        self.edge_features_dim = edge_features_dim

    def __call__(
        self,
        edge_index: mx.array,
        node_features: Union[mx.array, Tuple[mx.array, mx.array]],
        edge_features: mx.array,
    ) -> mx.array:
        """Computes the forward pass of GINEConv.

        Args:
            edge_index: Input edge index of shape `[2, num_edges]`
            node_features: Input node features, or a ``(src_features,
                dst_features)`` tuple for bipartite graphs
            edge_features: Edge features of shape `[num_edges, edge_features_dim]`
                (or `[num_edges, node_features_dim]` without ``edge_features_dim``)

        Returns:
            The computed node embeddings
        """
        if edge_features is None:
            raise ValueError("GINEConv requires `edge_features`.")
        src_dim = (
            node_features[0] if isinstance(node_features, tuple) else node_features
        ).shape[-1]
        if self.edge_features_dim is None and edge_features.shape[-1] != src_dim:
            raise ValueError(
                "Node and edge feature sizes do not match. Consider setting "
                "`edge_features_dim` in GINEConv."
            )
        return super(GINEConv, self).__call__(
            edge_index, node_features, edge_features=edge_features
        )
