from typing import Literal, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.nn.conv._norm import attention_leaky_relu
from mlx_graphs.nn.linear import Linear, glorot
from mlx_graphs.nn.message_passing import MessagePassing
from mlx_graphs.utils import scatter

RGATMods = Literal["additive", "scaled", "f-additive", "f-scaled"]
RGATMechanisms = Literal["within-relation", "across-relation"]
RGATModes = Literal["additive-self-attention", "multiplicative-self-attention"]


class RGATConv(MessagePassing):
    r"""The relational graph attentional operator from the `"Relational Graph
    Attention Networks" <https://arxiv.org/abs/1904.05811>`_ paper.

    Every edge :math:`j \to i` of relation :math:`r` is projected with the
    weight of its relation, :math:`\mathbf{g}_j = \mathbf{W}_r \mathbf{x}_j` and
    :math:`\mathbf{g}_i = \mathbf{W}_r \mathbf{x}_i`, and weighted by the
    attention coefficient

    .. math::
        \alpha_{i,j}^{(r)} = \mathrm{softmax}
        \left( \mathbf{q}_i^{(r)} + \mathbf{k}_j^{(r)} \right)
        \quad \textrm{or} \quad
        \mathrm{softmax}
        \left( \mathbf{q}_i^{(r)} \cdot \mathbf{k}_j^{(r)} \right),

    for the additive (with a LeakyReLU) and multiplicative self-attention
    modes, with :math:`\mathbf{q}_i^{(r)} = \mathbf{g}_i \mathbf{Q}` and
    :math:`\mathbf{k}_j^{(r)} = \mathbf{g}_j \mathbf{K}`:

    .. math::
        \mathbf{h}_i = \bigoplus_{r \in \mathcal{R}} \bigoplus_{j \in
        \mathcal{N}_r(i)} \alpha_{i,j}^{(r)} \mathbf{g}_j

    The softmax runs over all the neighbors of :math:`i`
    (``"across-relation"``) or over its neighbors of each relation separately
    (``"within-relation"``). With multiplicative self-attention, ``dim``
    attention coefficients are computed per head. The ``mod`` argument
    selects the cardinality preservation variants of `"Improving Attention
    Mechanism in Graph Neural Networks via Cardinality Preservation"
    <https://arxiv.org/abs/1907.02204>`_ (``"additive"``, ``"scaled"``,
    ``"f-additive"``, ``"f-scaled"``).

    .. note::
        Relation weights are applied either to every node for every relation
        (a ``[num_nodes, num_relations, heads * out_features_dim]`` array) or
        to every edge, whichever builds the smaller intermediate array. Both
        give the same result.

    Args:
        node_features_dim: Size of input node features
        out_features_dim: Size of output node embeddings (per head)
        num_relations: Number of relations (edge types)
        num_bases: If set, uses basis decomposition of the relation weights
            with this number of bases. Default: ``None``
        num_blocks: If set, uses block-diagonal decomposition of the relation
            weights with this number of blocks. Default: ``None``
        mod: Cardinality preservation variant, or ``None``. Default: ``None``
        attention_mechanism: ``"across-relation"`` or ``"within-relation"``.
            Default: ``"across-relation"``
        attention_mode: ``"additive-self-attention"`` or
            ``"multiplicative-self-attention"``.
            Default: ``"additive-self-attention"``
        heads: Number of attention heads. Default: ``1``
        dim: Number of attention coefficients per head (only for
            multiplicative self-attention). Default: ``1``
        concat: Whether to concatenate the heads or average them.
            Default: ``True``
        negative_slope: Slope of the LeakyReLU. Default: ``0.2``
        dropout: Dropout probability of the attention coefficients (only with
            ``mod=None``). Default: ``0.0``
        edge_features_dim: Size of edge features. Default: ``None``
        bias: Whether to add a learnable bias to the output. Default: ``True``
        **kwargs: Arguments of :class:`MessagePassing`, e.g. ``aggr``
            (``"add"`` by default)

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import RGATConv

        conv = RGATConv(16, 32, num_relations=3, heads=2)

        edge_index = mx.array([[0, 1, 2, 3, 4], [0, 0, 1, 1, 3]])
        edge_type = mx.array([0, 0, 1, 2, 2])
        node_features = mx.random.uniform(low=0, high=1, shape=(5, 16))

        h = conv(edge_index, node_features, edge_type)

        >>> h.shape
        (5, 64)

    References:
        `Busbridge et al. Relational Graph Attention Networks
        <https://arxiv.org/abs/1904.05811>`_
    """

    def __init__(
        self,
        node_features_dim: int,
        out_features_dim: int,
        num_relations: int,
        num_bases: Optional[int] = None,
        num_blocks: Optional[int] = None,
        mod: Optional[RGATMods] = None,
        attention_mechanism: RGATMechanisms = "across-relation",
        attention_mode: RGATModes = "additive-self-attention",
        heads: int = 1,
        dim: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        edge_features_dim: Optional[int] = None,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)
        # the messages are reduced with `scatter` directly
        if not isinstance(self.aggr, str) or "aggr_module" in self:
            raise ValueError(
                "RGATConv only supports a single scatter aggregation "
                "('add', 'mean', 'max', 'min', ...), not a list or an "
                f"Aggregation module (got {self.aggr!r})."
            )

        if attention_mechanism not in ("within-relation", "across-relation"):
            raise ValueError(
                'attention mechanism must either be "within-relation" or '
                '"across-relation"'
            )
        if attention_mode not in (
            "additive-self-attention",
            "multiplicative-self-attention",
        ):
            raise ValueError(
                'attention mode must either be "additive-self-attention" or '
                '"multiplicative-self-attention"'
            )
        if attention_mode == "additive-self-attention" and dim > 1:
            raise ValueError(
                '"additive-self-attention" mode cannot be applied when value of '
                'd is greater than 1. Use "multiplicative-self-attention" instead.'
            )
        if mod is not None and mod not in (
            "additive",
            "scaled",
            "f-additive",
            "f-scaled",
        ):
            raise ValueError(f"Unknown mod '{mod}'")
        if dropout > 0.0 and mod is not None:
            raise ValueError(
                "mod must be None with dropout value greater than 0 in order to "
                "sample attention coefficients stochastically"
            )
        if num_bases is not None and num_blocks is not None:
            raise ValueError(
                "Can not apply both basis-decomposition and "
                "block-diagonal-decomposition at the same time."
            )

        self.node_features_dim = node_features_dim
        self.out_features_dim = out_features_dim
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.num_blocks = num_blocks
        self.mod = mod
        self.attention_mechanism = attention_mechanism
        self.attention_mode = attention_mode
        self.heads = heads
        self.dim = dim
        self.concat = concat
        self.negative_slope = negative_slope
        self.edge_features_dim = edge_features_dim

        HC = heads * out_features_dim

        self.q = glorot((HC, heads * dim))
        self.k = glorot((HC, heads * dim))

        if bias:
            self.bias = mx.zeros(((heads if concat else 1) * dim * out_features_dim,))

        if edge_features_dim is not None:
            self.lin_edge = Linear(edge_features_dim, HC, bias=False)
            self.e = glorot((HC, heads * dim))

        if num_bases is not None:
            self.att = glorot((num_relations, num_bases))
            self.basis = glorot((num_bases, node_features_dim, HC))
        elif num_blocks is not None:
            if node_features_dim % num_blocks != 0 or HC % num_blocks != 0:
                raise ValueError(
                    "both 'node_features_dim' and 'heads * out_features_dim' must "
                    "be multiple of 'num_blocks' used"
                )
            self.weight = glorot(
                (
                    num_relations,
                    num_blocks,
                    node_features_dim // num_blocks,
                    HC // num_blocks,
                )
            )
        else:
            self.weight = glorot((num_relations, node_features_dim, HC))

        # Parameters of the cardinality preservation variants.
        if mod == "additive":
            self.w = mx.ones((out_features_dim,))
        if mod == "scaled":
            self.l1 = mx.ones((1, out_features_dim))
            self.b1 = mx.zeros((1, out_features_dim))
            self.l2 = mx.full(
                (out_features_dim, out_features_dim), 1.0 / out_features_dim
            )
            self.b2 = mx.zeros((1, out_features_dim))

        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)

    def _relation_weight(self) -> mx.array:
        if self.num_bases is not None:
            w = self.att @ self.basis.reshape(self.num_bases, -1)
            return w.reshape(
                self.num_relations,
                self.node_features_dim,
                self.heads * self.out_features_dim,
            )
        return self.weight

    def _use_node_path(self, num_nodes: int, num_edges: int) -> bool:
        """Whether to project every node with every relation weight instead of
        every edge with its relation weight: picks the smaller array, i.e.
        ``[num_nodes, num_relations, HC]`` vs ``[num_edges, in_dim, HC]``."""
        return num_nodes * self.num_relations <= num_edges * self.node_features_dim

    def _project(
        self,
        x: mx.array,
        src_idx: mx.array,
        dst_idx: mx.array,
        edge_type: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """Projections ``W_r x_i`` and ``W_r x_j`` of the endpoints of each edge."""
        weight = self._relation_weight()
        HC = self.heads * self.out_features_dim
        N, E = x.shape[0], edge_type.shape[0]

        if self.num_blocks is not None:
            B = self.num_blocks
            # explicit block size: reshape cannot infer it for 0 nodes/edges
            D = self.node_features_dim // B
            if self._use_node_path(N, E):
                # [N, B, in/B] x [R, B, in/B, HC/B] -> [N, R, B, HC/B]
                xw = mx.einsum("ncd,rcde->nrce", x.reshape(N, B, D), weight)
                xw = xw.reshape(N, self.num_relations, HC)
                return xw[dst_idx, edge_type], xw[src_idx, edge_type]
            w = weight[edge_type]  # [E, B, in/B, HC/B]

            def project(idx):
                return mx.einsum("acd,acde->ace", x[idx].reshape(E, B, D), w).reshape(
                    E, HC
                )

            return project(dst_idx), project(src_idx)

        if self._use_node_path(N, E):
            xw = mx.einsum("ni,rio->nro", x, weight)  # [N, R, HC]
            return xw[dst_idx, edge_type], xw[src_idx, edge_type]
        w = weight[edge_type]  # [E, in, HC]
        return (
            (x[dst_idx][:, None, :] @ w)[:, 0, :],
            (x[src_idx][:, None, :] @ w)[:, 0, :],
        )

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_type: mx.array,
        edge_features: Optional[mx.array] = None,
        return_attention_weights: bool = False,
    ) -> Union[mx.array, Tuple[mx.array, Tuple[mx.array, mx.array]]]:
        """Computes the forward pass of RGATConv.

        Args:
            edge_index: Input edge index of shape `[2, num_edges]`
            node_features: Input node features
            edge_type: Relation of each edge, integer array of shape `[num_edges]`
                with values in ``[0, num_relations)``
            edge_features: Optional edge features of shape
                `[num_edges, edge_features_dim]`
            return_attention_weights: If ``True``, also returns the tuple
                ``(edge_index, attention_weights)``, with attention weights of
                shape `[num_edges, heads * dim]`. Default: ``False``

        Returns:
            The computed node embeddings, and the attention weights if
            ``return_attention_weights`` is ``True``.
        """
        if not (isinstance(edge_index, mx.array) and edge_index.shape[0] == 2):
            raise ValueError("Edge index should be an array with shape (2, |E|)")
        if edge_type.ndim != 1 or edge_type.shape[0] != edge_index.shape[1]:
            raise ValueError(
                "`edge_type` should have shape [num_edges], "
                f"found {edge_type.shape} for {edge_index.shape[1]} edges."
            )

        H, C, D = self.heads, self.out_features_dim, self.dim
        src_idx, dst_idx = edge_index[0], edge_index[1]
        num_nodes = node_features.shape[0]
        self.num_nodes = num_nodes
        E = edge_type.shape[0]
        additive = self.attention_mode == "additive-self-attention"

        out_i, out_j = self._project(node_features, src_idx, dst_idx, edge_type)
        qi = out_i @ self.q  # [E, H * D]
        kj = out_j @ self.k

        alpha_edge = None
        if edge_features is not None:
            if "lin_edge" not in self:
                raise ValueError(
                    "Please set 'edge_features_dim' to use edge features in RGATConv"
                )
            if edge_features.ndim == 1:
                edge_features = edge_features.reshape(-1, 1)
            alpha_edge = self.lin_edge(edge_features) @ self.e

        if additive:
            alpha = qi + kj
            if alpha_edge is not None:
                alpha = alpha + alpha_edge
            alpha = attention_leaky_relu(alpha, self.negative_slope)
        else:
            alpha = qi * kj
            if alpha_edge is not None:
                alpha = alpha * alpha_edge

        if self.attention_mechanism == "within-relation":
            group = dst_idx * self.num_relations + edge_type
            alpha = scatter(alpha, group, num_nodes * self.num_relations, "softmax")
        else:
            alpha = scatter(alpha, dst_idx, num_nodes, "softmax")
        attention = alpha

        # Value of each edge, with the head and attention-dim axes.
        values = out_j.reshape(E, H, 1, C)
        weights = alpha.reshape(E, H, D, 1)

        if self.mod in ("scaled", "f-scaled"):
            degree = mx.zeros((num_nodes,), dtype=alpha.dtype).at[dst_idx].add(1)
            degree = degree[dst_idx][:, None]  # [E, 1]

        if self.mod == "additive":
            messages = values * weights + self.w * values
        elif self.mod == "scaled":
            scale = nn.relu(degree @ self.l1 + self.b1) @ self.l2 + self.b2
            messages = values * weights * scale.reshape(E, 1, 1, C)
        else:
            if self.mod == "f-additive":
                weights = mx.where(weights > 0, weights + 1, weights)
            elif self.mod == "f-scaled":
                weights = weights * degree.reshape(E, 1, 1, 1)
            elif "dropout" in self:
                weights = self.dropout(weights)
            messages = values * weights  # [E, H, D, C]

        out = scatter(messages, dst_idx, num_nodes, self.aggr)  # [N, H, D, C]

        if self.concat:
            out = out.reshape(num_nodes, H * D * C)
        else:
            out = out.mean(axis=1).reshape(num_nodes, D * C)

        if "bias" in self:
            out = out + self.bias

        if return_attention_weights:
            return out, (edge_index, attention)
        return out
