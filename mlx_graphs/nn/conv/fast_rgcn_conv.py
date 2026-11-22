from typing import Optional, Tuple, Union

import mlx.core as mx

from mlx_graphs.nn.conv.rgcn_conv import RGCNConv
from mlx_graphs.utils import scatter


class FastRGCNConv(RGCNConv):
    r"""Relational graph convolution layer with the same semantics as
    :class:`RGCNConv`, from the `"Modeling Relational Data with Graph
    Convolutional Networks" <https://arxiv.org/abs/1703.06103>`_ paper.

    .. math::
        \mathbf{h}_i = \mathbf{W}_{\textrm{root}} \mathbf{x}_i +
        \sum_{r \in \mathcal{R}} \bigoplus_{j \in \mathcal{N}_r(i)}
        \mathbf{W}_r \mathbf{x}_j

    Unlike :class:`RGCNConv`, which aggregates the neighbors of each relation
    before applying the relation weights, this layer applies the weight of its
    relation to every edge (one batched matmul over the edges, as PyG's
    ``FastRGCNConv``). This avoids aggregating per (node, relation) pair, at
    the cost of gathering one ``[node_features_dim, out_features_dim]`` weight
    matrix per edge, so it is best suited to small graphs and feature sizes.
    Only the ``"add"`` and ``"mean"`` aggregations are supported.

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
        **kwargs: Arguments of :class:`MessagePassing`, i.e. ``aggr``
            (``"mean"`` by default, or ``"add"``)

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import FastRGCNConv

        conv = FastRGCNConv(16, 32, num_relations=3, num_bases=2)

        edge_index = mx.array([[0, 1, 2, 3, 4], [0, 0, 1, 1, 3]])
        edge_type = mx.array([0, 0, 1, 2, 2])
        node_features = mx.random.uniform(low=0, high=1, shape=(5, 16))

        h = conv(edge_index, node_features, edge_type)

        >>> h.shape
        (5, 32)

    References:
        `Schlichtkrull et al. Modeling Relational Data with Graph Convolutional
        Networks <https://arxiv.org/abs/1703.06103>`_
    """

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
        super().__init__(
            node_features_dim,
            out_features_dim,
            num_relations,
            num_bases=num_bases,
            root_weight=root_weight,
            bias=bias,
            **kwargs,
        )
        if self.aggr not in ("add", "mean"):
            raise ValueError(
                f"FastRGCNConv only supports the 'add' and 'mean' aggregations, "
                f"got '{self.aggr}'."
            )

    # RGCNConv.__call__ validates the inputs, builds the relation weights and
    # adds the root and bias terms; only the aggregation differs. It is always
    # done per edge, so both code paths of RGCNConv lead to it.
    def _use_dense_path(self, num_nodes: int, num_edges: int) -> bool:
        return True

    def _aggregate_dense(
        self,
        edge_index: mx.array,
        src_features: mx.array,
        edge_type: mx.array,
        weight: mx.array,
    ) -> mx.array:
        """Applies the weight of its relation to every edge, then aggregates."""
        src_idx, dst_idx = edge_index[0], edge_index[1]

        # [num_edges, 1, src_dim] @ [num_edges, src_dim, out] -> [num_edges, out]
        messages = (src_features[src_idx][:, None, :] @ weight[edge_type])[:, 0, :]

        if self.aggr == "mean":
            # Normalize separately for every (destination, relation) pair.
            group = dst_idx * self.num_relations + edge_type
            counts = mx.zeros((self.num_nodes * self.num_relations,), dtype=mx.int32)
            counts = counts.at[group].add(1)
            messages = messages / counts[group][:, None].astype(messages.dtype)

        return scatter(messages, dst_idx, self.num_nodes, "add")
