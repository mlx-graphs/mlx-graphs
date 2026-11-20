import math
from typing import Optional, Tuple, Union

import mlx.core as mx

from mlx_graphs.nn.message_passing import MessagePassing
from mlx_graphs.utils import scatter


def _glorot(shape: Tuple[int, ...]) -> mx.array:
    """Glorot uniform init using the last two dimensions as fan-in and fan-out."""
    limit = math.sqrt(6.0 / (shape[-2] + shape[-1]))
    return mx.random.uniform(-limit, limit, shape)


class RGCNConv(MessagePassing):
    r"""Relational graph convolution layer from `"Modeling Relational Data with
    Graph Convolutional Networks" <https://arxiv.org/abs/1703.06103>`_ paper.

    .. math::
        \mathbf{h}_i = \mathbf{W}_{\textrm{root}} \mathbf{x}_i +
        \sum_{r \in \mathcal{R}} \bigoplus_{j \in \mathcal{N}_r(i)}
        \mathbf{W}_r \mathbf{x}_j,

    where :math:`\mathcal{R}` is the set of relations (edge types),
    :math:`\mathcal{N}_r(i)` are the neighbors of node :math:`i` under relation
    :math:`r` and :math:`\bigoplus` denotes the aggregation function, set by default
    to `mean`, i.e. :math:`\frac{1}{|\mathcal{N}_r(i)|} \sum_{j \in \mathcal{N}_r(i)}`.

    With ``num_bases`` set, the relation weights share a small set of basis matrices
    :math:`\mathbf{V}_b` (basis decomposition), which reduces the number of
    parameters when there are many relations:

    .. math::
        \mathbf{W}_r = \sum_{b=1}^{B} a_{rb} \mathbf{V}_b

    The layer also works on bipartite graphs: pass ``node_features`` as a
    ``(src_features, dst_features)`` tuple and, if the two sets have different
    feature sizes, ``node_features_dim`` as a ``(src_dim, dst_dim)`` tuple. The
    output has one row per destination node.

    .. note::
        Small inputs are aggregated in one shot into a dense
        `[num_nodes, num_relations, node_features_dim]` array. When that array would
        be both larger than the per-edge features and larger than
        ``RGCNConv.dense_max_elements`` elements, the layer instead loops over the
        relations, each time using only the edges of that relation, so memory stays
        proportional to the number of edges and nodes. Both paths give the same
        result.

    Args:
        node_features_dim: Size of input node features, or a ``(src_dim, dst_dim)``
            tuple for bipartite graphs
        out_features_dim: Size of output node embeddings
        num_relations: Number of relations (edge types)
        num_bases: If set, uses basis decomposition with this number of bases.
            Default: ``None``
        root_weight: Whether to add the transformed features of the node itself
            to the output. Default: ``True``
        bias: Whether to add a learnable bias to the output. Default: ``True``
        **kwargs: Arguments of :class:`MessagePassing`, e.g. ``aggr`` (``"mean"`` by
            default, or ``"add"``, ``"max"``, ``"min"``)

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn.conv.rgcn_conv import RGCNConv

        conv = RGCNConv(16, 32, num_relations=3, num_bases=2)

        edge_index = mx.array([[0, 1, 2, 3, 4], [0, 0, 1, 1, 3]])
        edge_type = mx.array([0, 0, 1, 2, 2])
        node_features = mx.random.uniform(low=0, high=1, shape=(5, 16))

        h = conv(edge_index, node_features, edge_type)

        >>> h.shape
        (5, 32)
    """

    # The dense path is always used below this number of elements in the
    # [num_nodes * num_relations, node_features_dim] aggregate (4M floats = 16 MB).
    dense_max_elements: int = 1 << 22

    def __init__(
        self,
        node_features_dim: Union[int, Tuple[int, int]],
        out_features_dim: int,
        num_relations: int,
        num_bases: Optional[int] = None,
        root_weight: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "mean")
        super(RGCNConv, self).__init__(**kwargs)

        if self.aggr == "softmax":
            raise ValueError("RGCNConv does not support the 'softmax' aggregation.")
        if num_relations < 1:
            raise ValueError(f"num_relations must be positive, got {num_relations}")
        if num_bases is not None and num_bases < 1:
            raise ValueError(f"num_bases must be positive, got {num_bases}")

        if isinstance(node_features_dim, int):
            src_dim = dst_dim = node_features_dim
        else:
            src_dim, dst_dim = node_features_dim

        self.node_features_dim = node_features_dim
        self.out_features_dim = out_features_dim
        self.src_features_dim = src_dim
        self.num_relations = num_relations
        self.num_bases = num_bases

        # Weights are stored as [..., in, out] (applied as `x @ weight`), the same
        # layout as PyG's RGCNConv.
        if num_bases is not None:
            self.weight = _glorot((num_bases, src_dim, out_features_dim))
            self.comp = _glorot((num_relations, num_bases))
        else:
            self.weight = _glorot((num_relations, src_dim, out_features_dim))

        if root_weight:
            self.root = _glorot((dst_dim, out_features_dim))

        if bias:
            self.bias = mx.zeros((out_features_dim,))

    def __call__(
        self,
        edge_index: mx.array,
        node_features: Union[mx.array, Tuple[mx.array, mx.array]],
        edge_type: mx.array,
    ) -> mx.array:
        """Computes the forward pass of RGCNConv.

        Args:
            edge_index: Input edge index of shape `[2, num_edges]`
            node_features: Input node features, or a ``(src_features,
                dst_features)`` tuple for bipartite graphs
            edge_type: Relation of each edge, integer array of shape `[num_edges]`
                with values in ``[0, num_relations)``

        Returns:
            mx.array: The computed node embeddings, one row per destination node
        """
        if not (isinstance(edge_index, mx.array) and edge_index.shape[0] == 2):
            raise ValueError("Edge index should be an array with shape (2, |E|)")
        if edge_type.ndim != 1 or edge_type.shape[0] != edge_index.shape[1]:
            raise ValueError(
                "`edge_type` should have shape [num_edges], "
                f"found {edge_type.shape} for {edge_index.shape[1]} edges."
            )

        if isinstance(node_features, tuple):
            src_features, dst_features = node_features
        else:
            src_features = dst_features = node_features

        self.num_nodes = dst_features.shape[0]

        # shape: [num_relations, src_dim, out_features_dim]
        weight = self.weight
        if self.num_bases is not None:
            weight = (self.comp @ weight.reshape(self.num_bases, -1)).reshape(
                self.num_relations, self.src_features_dim, self.out_features_dim
            )

        if self._use_dense_path(self.num_nodes, edge_index.shape[1]):
            out = self._aggregate_dense(edge_index, src_features, edge_type, weight)
        else:
            out = self._aggregate_per_relation(
                edge_index, src_features, edge_type, weight
            )

        if "root" in self:
            out = out + dst_features @ self.root

        if "bias" in self:
            out = out + self.bias

        return out

    def _use_dense_path(self, num_nodes: int, num_edges: int) -> bool:
        """Whether to aggregate into a dense ``[num_nodes * num_relations, dim]``
        array (one scatter and one matmul) instead of looping over relations.

        Dense is used when that array is no larger than the gathered per-edge
        features (which every layer builds anyway), or small in absolute terms.
        """
        num_segments = num_nodes * self.num_relations
        return (
            num_segments <= num_edges
            or num_segments * self.src_features_dim <= self.dense_max_elements
        )

    def _aggregate_dense(
        self,
        edge_index: mx.array,
        src_features: mx.array,
        edge_type: mx.array,
        weight: mx.array,
    ) -> mx.array:
        """Aggregates neighbors per (destination node, relation) pair in one
        scatter, then applies all the relation weights in one matmul."""
        R = self.num_relations
        src_idx, dst_idx = edge_index[0], edge_index[1]

        # shape: [num_nodes * num_relations, src_dim]; empty segments are zero.
        aggregated = scatter(
            src_features[src_idx],
            dst_idx * R + edge_type,
            self.num_nodes * R,
            self.aggr,
        )

        # [num_nodes, R * src_dim] @ [R * src_dim, out] sums over the relations.
        return aggregated.reshape(self.num_nodes, -1) @ weight.reshape(
            -1, self.out_features_dim
        )

    def _aggregate_per_relation(
        self,
        edge_index: mx.array,
        src_features: mx.array,
        edge_type: mx.array,
        weight: mx.array,
    ) -> mx.array:
        """Same result as :meth:`_aggregate_dense` with memory proportional to the
        number of edges: edges are sorted by relation and every relation only
        processes its own slice of edges."""
        R, num_nodes, num_edges = self.num_relations, self.num_nodes, edge_type.size
        out = mx.zeros((num_nodes, self.out_features_dim), dtype=src_features.dtype)
        if num_edges == 0:
            return out

        # Number of edges per relation. This evaluates `edge_type`.
        counts = mx.zeros((R,), dtype=mx.int32).at[edge_type].add(1).tolist()

        # Sort the edges by (relation, destination): every relation is a slice.
        key = edge_type.astype(mx.int64) * num_nodes + edge_index[1]
        perm = mx.argsort(key)
        src_idx, dst_idx = edge_index[0][perm], edge_index[1][perm]

        # add/mean are linear, so the weight can be applied per edge and all the
        # relations aggregated with a single scatter. max/min have to aggregate the
        # input features of each relation before applying its weight.
        linear = self.aggr in ("add", "mean")
        messages = []

        start = 0
        for relation, count in enumerate(counts):
            if count == 0:
                continue
            src_r = src_idx[start : start + count]
            dst_r = dst_idx[start : start + count]
            start += count

            features = src_features[src_r]  # [num_edges_r, src_dim]
            if linear:
                messages.append(features @ weight[relation])  # [num_edges_r, out]
            else:
                aggregated = scatter(features, dst_r, num_nodes, self.aggr)
                out = out + aggregated @ weight[relation]

        if not linear:
            return out

        messages = mx.concatenate(messages, axis=0)  # [num_edges, out]
        if self.aggr == "mean":
            # 1 / |N_r(i)| per edge: edges with the same (relation, destination)
            # are contiguous, so number these groups and count their edges.
            key = key[perm]
            is_new_group = mx.concatenate([mx.array([False]), key[1:] != key[:-1]])
            group = mx.cumsum(is_new_group.astype(mx.int32))
            group_size = mx.zeros((num_edges,), dtype=mx.int32).at[group].add(1)
            messages = messages / group_size[group][:, None]

        return scatter(messages, dst_idx, num_nodes, "add")
