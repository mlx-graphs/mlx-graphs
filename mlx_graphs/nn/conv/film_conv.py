import copy
from typing import Callable, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.nn import Linear  # PyTorch's default initialization, as in PyG

from mlx_graphs.nn.message_passing import MessagePassing


class FiLMConv(MessagePassing):
    r"""Feature-wise linear modulation (FiLM) graph convolution layer from the
    `"GNN-FiLM: Graph Neural Networks with Feature-wise Linear Modulation"
    <https://arxiv.org/abs/1906.12192>`_ paper.

    .. math::
        \mathbf{h}_i = \sigma \left(
        \boldsymbol{\gamma}_{\textrm{skip},i} \odot \mathbf{\Theta}_\textrm{skip}
        \mathbf{x}_i + \boldsymbol{\beta}_{\textrm{skip},i} \right) +
        \sum_{r \in \mathcal{R}} \bigoplus_{j \in \mathcal{N}_r(i)}
        \sigma \left( \boldsymbol{\gamma}_{r,i} \odot \mathbf{\Theta}_r
        \mathbf{x}_j + \boldsymbol{\beta}_{r,i} \right)

    where :math:`\boldsymbol{\beta}_{r,i}, \boldsymbol{\gamma}_{r,i} =
    g_r(\mathbf{x}_i)`, with :math:`g_r` a linear layer by default (or a copy of
    ``mlp``) whose output is split in two halves, and :math:`\mathcal{N}_r(i)`
    the neighbours of :math:`i` through edges of relation :math:`r`.

    Args:
        node_features_dim: Size of input node features, or a ``(src_dim,
            dst_dim)`` tuple for bipartite graphs
        out_features_dim: Size of output node embeddings
        num_relations: Number of relations (edge types). Default: ``1``
        mlp: Module mapping destination node features to ``2 *
            out_features_dim`` values :math:`[\boldsymbol{\beta},
            \boldsymbol{\gamma}]`. It is deep-copied for every relation and for
            the skip connection. Default: ``None`` (a linear layer)
        act: Activation function :math:`\sigma`, or ``None``.
            Default: :func:`mlx.nn.relu`
        aggr: Aggregation :math:`\bigoplus` within each relation.
            Default: ``"mean"``
        **kwargs: Arguments of :class:`MessagePassing`

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import FiLMConv

        edge_index = mx.array([[0, 1, 2, 3, 4], [1, 0, 3, 2, 0]])
        edge_type = mx.array([0, 1, 0, 1, 1])
        node_features = mx.random.normal((5, 16))

        conv = FiLMConv(16, 32, num_relations=2)
        h = conv(edge_index, node_features, edge_type)
        >>> h.shape
        (5, 32)

    References:
        [1] `Brockschmidt. GNN-FiLM: Graph Neural Networks with Feature-wise
        Linear Modulation. <https://arxiv.org/abs/1906.12192>`_
    """

    # what follows the aggregation expects the size of one aggregation (as in PyG)
    supports_concat_aggr = False

    def __init__(
        self,
        node_features_dim: Union[int, Tuple[int, int]],
        out_features_dim: int,
        num_relations: int = 1,
        mlp: Optional[nn.Module] = None,
        act: Optional[Callable[[mx.array], mx.array]] = nn.relu,
        aggr: str = "mean",
        **kwargs,
    ):
        super(FiLMConv, self).__init__(aggr=aggr, **kwargs)

        self.node_features_dim = node_features_dim
        self.out_features_dim = out_features_dim
        self.num_relations = max(num_relations, 1)
        self.act = act

        if isinstance(node_features_dim, int):
            src_dim = dst_dim = node_features_dim
        else:
            src_dim, dst_dim = node_features_dim

        # Same names as PyG.
        self.lins = [
            Linear(src_dim, out_features_dim, bias=False)
            for _ in range(self.num_relations)
        ]
        self.films = [
            Linear(dst_dim, 2 * out_features_dim) if mlp is None else copy.deepcopy(mlp)
            for _ in range(self.num_relations)
        ]
        self.lin_skip = Linear(dst_dim, out_features_dim, bias=False)
        self.film_skip = (
            Linear(dst_dim, 2 * out_features_dim, bias=False)
            if mlp is None
            else copy.deepcopy(mlp)
        )

    def __call__(
        self,
        edge_index: mx.array,
        node_features: Union[mx.array, Tuple[mx.array, mx.array]],
        edge_type: Optional[mx.array] = None,
    ) -> mx.array:
        """Computes the forward pass of FiLMConv.

        Args:
            edge_index: Input edge index of shape `[2, num_edges]`
            node_features: Input node features, or a ``(src_features,
                dst_features)`` tuple for bipartite graphs
            edge_type: Relation of each edge, of shape `[num_edges]`, with values
                in ``[0, num_relations)``. Required if ``num_relations > 1``.
                Default: ``None``

        Returns:
            mx.array: The computed node embeddings, one row per destination node
        """
        if isinstance(node_features, tuple):
            src_features, dst_features = node_features
        else:
            src_features = dst_features = node_features
        num_dst, R = dst_features.shape[0], self.num_relations

        beta, gamma = mx.split(self.film_skip(dst_features), 2, axis=-1)
        out = gamma * self.lin_skip(dst_features) + beta
        if self.act is not None:
            out = self.act(out)

        if R == 1:
            x_src = self.lins[0](src_features)
            film = self.films[0](dst_features)
        else:
            if edge_type is None:
                raise ValueError("`edge_type` is required when num_relations > 1.")
            # All relations in one pass: node j of relation r becomes node
            # r * num_nodes + j, so messages are aggregated per (node,
            # relation) pair, and then summed over the relations.
            num_src = src_features.shape[0]
            edge_type = edge_type.reshape(-1).astype(edge_index.dtype)
            edge_index = mx.stack(
                [
                    edge_index[0] + edge_type * num_src,
                    edge_index[1] + edge_type * num_dst,
                ]
            )
            x_src = mx.concatenate([lin(src_features) for lin in self.lins])
            film = mx.concatenate([f(dst_features) for f in self.films])

        # The FiLM parameters [beta, gamma] travel as destination features.
        aggregated = self.propagate(edge_index=edge_index, node_features=(x_src, film))
        return out + aggregated.reshape(R, num_dst, -1).sum(axis=0)

    def message(
        self, src_features: mx.array, dst_features: mx.array, **kwargs
    ) -> mx.array:
        beta, gamma = mx.split(dst_features, 2, axis=-1)
        out = gamma * src_features + beta
        if self.act is not None:
            out = self.act(out)
        return out
