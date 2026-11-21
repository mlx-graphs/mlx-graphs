from typing import Optional, Union

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.nn.aggr.utils import DenseLayout
from mlx_graphs.nn.attention import PolynormerAttention, SGFormerAttention
from mlx_graphs.nn.conv.gat_conv import GATConv
from mlx_graphs.nn.conv.gcn_conv import GCNConv
from mlx_graphs.nn.conv.gps_conv import _dense_layout


class _SGFormerGraphModule(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.fcs = [nn.Linear(in_dim, hidden_dim)]
        self.bns = [nn.BatchNorm(hidden_dim) for _ in range(num_layers + 1)]
        self.convs = [
            GCNConv(hidden_dim, hidden_dim, add_self_loops=True)
            for _ in range(num_layers)
        ]
        self.dropout = nn.Dropout(dropout)

    def __call__(self, edge_index, x):
        x = self.dropout(nn.relu(self.bns[0](self.fcs[0](x))))
        last_x = x
        for i, conv in enumerate(self.convs):
            x = conv(edge_index, x)
            x = self.dropout(nn.relu(self.bns[i + 1](x)))
            x = x + last_x
        return x


class _SGFormerTransModule(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, num_heads, dropout):
        super().__init__()
        self.fcs = [nn.Linear(in_dim, hidden_dim)]
        self.bns = [nn.LayerNorm(hidden_dim) for _ in range(num_layers + 1)]
        self.attns = [
            SGFormerAttention(hidden_dim, num_heads, hidden_dim)
            for _ in range(num_layers)
        ]
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x, batch_indices, layout=None):
        layout = _dense_layout(layout, batch_indices, x.shape[0])
        x, mask = layout.to_dense(x), layout.mask
        x = self.dropout(nn.relu(self.bns[0](self.fcs[0](x))))
        layers = [x]
        for i, attn in enumerate(self.attns):
            x = (attn(x, mask) + layers[i]) / 2.0
            x = self.dropout(nn.relu(self.bns[i + 1](x)))
            layers.append(x)
        return layout.from_dense(x)


class SGFormer(nn.Module):
    r"""The SGFormer model from the `"SGFormer: Simplifying and Empowering
    Transformers for Large-Graph Representations"
    <https://arxiv.org/abs/2306.10759>`_ paper, as PyG's ``SGFormer``.

    A single-layer-style linear global attention branch
    (:class:`~mlx_graphs.nn.attention.SGFormerAttention`) and a GCN branch
    are combined, then a linear layer and a log-softmax give per-node class
    log-probabilities.

    .. note::
        As in PyG, the global attention runs on the zero-padded dense batch
        of each graph, so the padding nodes take part in the attention
        normalizer.

    Args:
        node_features_dim: Size of the input node features.
        hidden_features_dim: Size of the hidden node embeddings.
        out_features_dim: Number of classes.
        trans_num_layers: Number of global attention layers. Default: ``2``
        trans_num_heads: Number of attention heads. Default: ``1``
        trans_dropout: Dropout probability of the attention branch.
            Default: ``0.5``
        gnn_num_layers: Number of GCN layers. Default: ``3``
        gnn_dropout: Dropout probability of the GCN branch. Default: ``0.5``
        graph_weight: Weight of the GCN branch when ``aggregate="add"``
            (the attention branch gets ``1 - graph_weight``). Default: ``0.5``
        aggregate: How the branches are combined: ``"add"`` (weighted sum)
            or ``"cat"`` (concatenation). Default: ``"add"``

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import SGFormer

        model = SGFormer(16, 32, 4)
        edge_index = mx.array([[0, 1, 1, 2], [1, 0, 2, 1]])
        node_features = mx.random.normal((3, 16))
        >>> model(edge_index, node_features).shape
        (3, 4)

    References:
        - `Wu et al., "SGFormer: Simplifying and Empowering Transformers for
          Large-Graph Representations", NeurIPS 2023
          <https://arxiv.org/abs/2306.10759>`_
    """

    def __init__(
        self,
        node_features_dim: int,
        hidden_features_dim: int,
        out_features_dim: int,
        trans_num_layers: int = 2,
        trans_num_heads: int = 1,
        trans_dropout: float = 0.5,
        gnn_num_layers: int = 3,
        gnn_dropout: float = 0.5,
        graph_weight: float = 0.5,
        aggregate: str = "add",
    ):
        super().__init__()
        self.trans_conv = _SGFormerTransModule(
            node_features_dim,
            hidden_features_dim,
            trans_num_layers,
            trans_num_heads,
            trans_dropout,
        )
        self.graph_conv = _SGFormerGraphModule(
            node_features_dim, hidden_features_dim, gnn_num_layers, gnn_dropout
        )
        self.graph_weight = graph_weight
        self.aggregate = aggregate
        if aggregate == "add":
            self.fc = nn.Linear(hidden_features_dim, out_features_dim)
        elif aggregate == "cat":
            self.fc = nn.Linear(2 * hidden_features_dim, out_features_dim)
        else:
            raise ValueError(f"Invalid aggregate type: {aggregate}")
        self.out_features_dim = out_features_dim

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        batch_indices: Optional[mx.array] = None,
        layout: Optional[DenseLayout] = None,
    ) -> mx.array:
        """Computes the class log-probabilities of the nodes.

        Args:
            edge_index: Edge index of shape ``[2, num_edges]``.
            node_features: Node features of shape
                ``[num_nodes, node_features_dim]``.
            batch_indices: Graph of each node; attention is computed within
                each graph. Default: ``None`` (a single graph).
            layout: Optional precomputed dense layout of the nodes per graph
                (:class:`~mlx_graphs.nn.aggr.DenseLayout`), to avoid
                computing it on the host at every call (a GPU
                synchronization that also prevents :func:`mx.compile`), e.g.
                ``DenseLayout.from_sizes(batch_indices, num_nodes,
                num_graphs, max_num_nodes)``. Default: ``None``

        Returns:
            Log-probabilities of shape ``[num_nodes, out_features_dim]``.
        """
        x1 = self.trans_conv(node_features, batch_indices, layout)
        x2 = self.graph_conv(edge_index, node_features)
        if self.aggregate == "add":
            x = self.graph_weight * x2 + (1 - self.graph_weight) * x1
        else:
            x = mx.concatenate([x1, x2], axis=1)
        return nn.log_softmax(self.fc(x), axis=-1)


class Polynormer(nn.Module):
    r"""The Polynormer model from the `"Polynormer: Polynomial-Expressive
    Graph Transformer in Linear Time" <https://arxiv.org/abs/2403.01232>`_
    paper, as PyG's ``Polynormer``.

    A stack of local (GCN or GAT based) polynomial layers is followed, once
    ``_global`` is set to ``True`` (usually after some warm-up epochs, as in
    the paper), by global linear attention layers
    (:class:`~mlx_graphs.nn.attention.PolynormerAttention`). The output is a
    per-node log-softmax.

    .. note::
        Unlike PyG 2.8, the global attention layers take
        ``heads * hidden_features_dim`` input features, so that
        ``heads > 1`` also works with ``_global = True`` (identical for
        ``heads = 1``).

    Args:
        node_features_dim: Size of the input node features.
        hidden_features_dim: Size of the hidden embeddings of each head.
        out_features_dim: Number of classes.
        local_layers: Number of local layers. Default: ``7``
        global_layers: Number of global attention layers. Default: ``2``
        in_dropout: Dropout probability of the input. Default: ``0.15``
        dropout: Dropout probability of the local layers. Default: ``0.5``
        global_dropout: Dropout probability of the global layers.
            Default: ``0.5``
        heads: Number of heads. Default: ``1``
        beta: Polynormer :math:`\beta`. Default: ``0.9``
        qk_shared: Whether queries and keys share their weights in the
            global attention. Default: ``False``
        pre_ln: Whether to apply a layer norm before each local layer.
            Default: ``False``
        post_bn: Whether to apply a batch norm after each local layer.
            Default: ``True``
        local_attn: Whether the local layers use :class:`~mlx_graphs.nn.GATConv`
            (``True``) or :class:`~mlx_graphs.nn.GCNConv` (``False``).
            Default: ``False``

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import Polynormer

        model = Polynormer(16, 32, 4, local_layers=2, global_layers=1)
        model._global = True  # enable the global attention layers
        edge_index = mx.array([[0, 1, 1, 2], [1, 0, 2, 1]])
        node_features = mx.random.normal((3, 16))
        >>> model(edge_index, node_features).shape
        (3, 4)

    References:
        - `Deng et al., "Polynormer: Polynomial-Expressive Graph Transformer
          in Linear Time", ICLR 2024 <https://arxiv.org/abs/2403.01232>`_
    """

    def __init__(
        self,
        node_features_dim: int,
        hidden_features_dim: int,
        out_features_dim: int,
        local_layers: int = 7,
        global_layers: int = 2,
        in_dropout: float = 0.15,
        dropout: float = 0.5,
        global_dropout: float = 0.5,
        heads: int = 1,
        beta: float = 0.9,
        qk_shared: bool = False,
        pre_ln: bool = False,
        post_bn: bool = True,
        local_attn: bool = False,
    ):
        super().__init__()
        self._global = False
        self.pre_ln = pre_ln
        self.post_bn = post_bn
        self.beta = beta
        self.in_drop = nn.Dropout(in_dropout)
        self.drop = nn.Dropout(dropout)

        inner = heads * hidden_features_dim
        dims_in = [node_features_dim] + [inner] * (local_layers - 1)
        self.h_lins = [nn.Linear(d, inner) for d in dims_in]
        if local_attn:
            self.local_convs = [
                GATConv(
                    d,
                    hidden_features_dim,
                    heads=heads,
                    concat=True,
                    bias=False,
                    add_self_loops=False,  # as in PyG's Polynormer
                )
                for d in dims_in
            ]
        else:
            self.local_convs = [GCNConv(d, inner, add_self_loops=True) for d in dims_in]
        self.lins = [nn.Linear(d, inner) for d in dims_in]
        self.lns = [nn.LayerNorm(inner) for _ in dims_in]
        if pre_ln:
            self.pre_lns = [nn.LayerNorm(d) for d in dims_in]
        if post_bn:
            self.post_bns = [nn.BatchNorm(inner) for _ in dims_in]

        self.lin_in = nn.Linear(node_features_dim, inner)
        self.ln = nn.LayerNorm(inner)
        self.global_attn = [
            PolynormerAttention(
                # PyG passes `hidden_channels` here, which only works with
                # `heads=1` (the input has `heads * hidden_channels` features).
                channels=inner,
                heads=heads,
                head_channels=hidden_features_dim,
                beta=beta,
                dropout=global_dropout,
                qk_shared=qk_shared,
            )
            for _ in range(global_layers)
        ]
        self.pred_local = nn.Linear(inner, out_features_dim)
        self.pred_global = nn.Linear(inner, out_features_dim)

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        batch_indices: Optional[mx.array] = None,
        layout: Optional[DenseLayout] = None,
    ) -> mx.array:
        """Computes the class log-probabilities of the nodes.

        Args:
            edge_index: Edge index of shape ``[2, num_edges]``.
            node_features: Node features of shape
                ``[num_nodes, node_features_dim]``.
            batch_indices: Graph of each node; the global attention is
                computed within each graph. Default: ``None`` (one graph).
            layout: Optional precomputed dense layout of the nodes per graph
                (:class:`~mlx_graphs.nn.aggr.DenseLayout`), to avoid
                computing it on the host at every call (a GPU
                synchronization that also prevents :func:`mx.compile`), e.g.
                ``DenseLayout.from_sizes(batch_indices, num_nodes,
                num_graphs, max_num_nodes)``. Default: ``None``

        Returns:
            Log-probabilities of shape ``[num_nodes, out_features_dim]``.
        """
        x = self.in_drop(node_features)
        x_local: Union[int, mx.array] = 0
        for i, local_conv in enumerate(self.local_convs):
            if self.pre_ln:
                x = self.pre_lns[i](x)
            h = nn.relu(self.h_lins[i](x))
            x = local_conv(edge_index, x) + self.lins[i](x)
            if self.post_bn:
                x = self.post_bns[i](x)
            x = self.drop(nn.relu(x))
            x = (1 - self.beta) * self.lns[i](h * x) + self.beta * x
            x_local = x_local + x

        if self._global:
            layout = _dense_layout(layout, batch_indices, x_local.shape[0])
            x_global = layout.to_dense(self.ln(x_local))
            for attn in self.global_attn:
                x_global = attn(x_global, layout.mask)
            x = self.pred_global(layout.from_dense(x_global))
        else:
            x = self.pred_local(x_local)
        return nn.log_softmax(x, axis=-1)
