import math
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.nn.conv.simple_conv import SimpleConv


class _ActFirstMLP(nn.Module):
    """MLP applying ``linear -> relu -> batch norm -> dropout`` on the hidden
    layers and a plain linear output layer, as PyG's
    ``MLP(..., act_first=True)`` used by LINKX."""

    def __init__(self, dims: List[int], dropout: float = 0.0):
        super().__init__()
        self.lins = [nn.Linear(i, o) for i, o in zip(dims[:-1], dims[1:])]
        self.norms = [nn.BatchNorm(d) for d in dims[1:-1]]
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    def __call__(self, x: mx.array) -> mx.array:
        for lin, norm in zip(self.lins[:-1], self.norms):
            x = norm(nn.relu(lin(x)))
            if self.dropout is not None:
                x = self.dropout(x)
        return self.lins[-1](x)


class LINKX(nn.Module):
    r"""The LINKX model from the `"Large Scale Learning on Non-Homophilous
    Graphs: New Benchmarks and Strong Simple Methods"
    <https://arxiv.org/abs/2110.14446>`_ paper.

    .. math::
        \mathbf{H}_{\mathbf{A}} &= \textrm{MLP}_{\mathbf{A}}(\mathbf{A})

        \mathbf{H}_{\mathbf{X}} &= \textrm{MLP}_{\mathbf{X}}(\mathbf{X})

        \mathbf{Y} &= \textrm{MLP}_{f} \left( \sigma \left( \mathbf{W}
        [\mathbf{H}_{\mathbf{A}}, \mathbf{H}_{\mathbf{X}}] +
        \mathbf{H}_{\mathbf{A}} + \mathbf{H}_{\mathbf{X}} \right) \right)

    The first layer of :math:`\textrm{MLP}_{\mathbf{A}}` multiplies the
    (sparse) adjacency matrix with a ``[num_nodes, hidden_features_dim]``
    weight, i.e. node ``i`` gets the sum of the weights of the nodes with an
    edge towards it.

    Args:
        num_nodes: The number of nodes in the graph.
        node_features_dim: Size of the input node features.
        hidden_features_dim: Size of the hidden embeddings.
        out_features_dim: Size of the output.
        num_layers: Number of layers of :math:`\textrm{MLP}_{f}`.
        num_edge_layers: Number of layers of :math:`\textrm{MLP}_{\mathbf{A}}`.
            Default: ``1``
        num_node_layers: Number of layers of :math:`\textrm{MLP}_{\mathbf{X}}`.
            Default: ``1``
        dropout: Dropout probability of the hidden embeddings of
            :math:`\textrm{MLP}_{f}`. Default: ``0.0``

    Example:

    .. code-block:: python

        from mlx_graphs.nn import LINKX

        model = LINKX(num_nodes, node_features_dim, 64, num_classes, num_layers=2)
        logits = model(edge_index, node_features)

    References:
        [1] `Lim et al. Large Scale Learning on Non-Homophilous Graphs: New
        Benchmarks and Strong Simple Methods <https://arxiv.org/abs/2110.14446>`_
    """

    def __init__(
        self,
        num_nodes: int,
        node_features_dim: int,
        hidden_features_dim: int,
        out_features_dim: int,
        num_layers: int,
        num_edge_layers: int = 1,
        num_node_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_features_dim = node_features_dim
        self.out_features_dim = out_features_dim
        self.num_edge_layers = num_edge_layers

        # same initialization as PyG's SparseLinear (kaiming uniform)
        bound = 1.0 / math.sqrt(num_nodes)
        self.edge_weight = mx.random.uniform(
            -bound, bound, (num_nodes, hidden_features_dim)
        )
        self.edge_bias = mx.random.uniform(-bound, bound, (hidden_features_dim,))
        self._propagate = SimpleConv(aggr="add")
        if num_edge_layers > 1:
            self.edge_norm = nn.BatchNorm(hidden_features_dim)
            self.edge_mlp = _ActFirstMLP([hidden_features_dim] * num_edge_layers)

        self.node_mlp = _ActFirstMLP(
            [node_features_dim] + [hidden_features_dim] * num_node_layers
        )
        self.cat_lin1 = nn.Linear(hidden_features_dim, hidden_features_dim)
        self.cat_lin2 = nn.Linear(hidden_features_dim, hidden_features_dim)
        self.final_mlp = _ActFirstMLP(
            [hidden_features_dim] * num_layers + [out_features_dim], dropout=dropout
        )

    def __call__(
        self,
        edge_index: mx.array,
        node_features: Optional[mx.array] = None,
        edge_weights: Optional[mx.array] = None,
    ) -> mx.array:
        """Computes the node predictions.

        Args:
            edge_index: Input edge index of shape `[2, num_edges]`
            node_features: Optional input node features. Without them, only
                the graph structure is used. Default: ``None``
            edge_weights: Optional weight of each edge. Default: ``None``

        Returns:
            An array of shape ``[num_nodes, out_features_dim]``.
        """
        out = self._propagate(edge_index, self.edge_weight, edge_weights)
        out = out + self.edge_bias
        if "edge_mlp" in self:
            out = self.edge_mlp(self.edge_norm(nn.relu(out)))
        out = out + self.cat_lin1(out)

        if node_features is not None:
            x = self.node_mlp(node_features)
            out = out + x
            out = out + self.cat_lin2(x)

        return self.final_mlp(nn.relu(out))
