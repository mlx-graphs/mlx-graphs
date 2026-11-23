from typing import Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.nn.conv._norm import attention_leaky_relu
from mlx_graphs.nn.conv.hetero_conv import EdgeType, _edge_type_to_key
from mlx_graphs.nn.linear import glorot
from mlx_graphs.utils import scatter

Metadata = Tuple[List[str], List[EdgeType]]


class HANConv(nn.Module):
    r"""The Heterogeneous Graph Attention Operator from the `"Heterogenous Graph
    Attention Network" <https://arxiv.org/abs/1903.07293>`_ paper.

    Node features of every node type are first projected to
    ``out_features_dim`` features. Then, for every edge type
    :math:`\Phi = (s, r, d)`, a node-level multi-head attention (as in GAT, with
    attention vectors specific to the edge type) aggregates the neighbors of
    each node of type :math:`d`, followed by a ReLU:

    .. math::
        \mathbf{z}_i^{\Phi} = \mathrm{ReLU} \left( \Big\Vert_{h=1}^{H}
        \sum_{j \in \mathcal{N}^{\Phi}(i)} \alpha_{i,j}^{\Phi,h}
        \mathbf{W}_{s}^{h} \mathbf{x}_j \right)

    Finally, a semantic-level attention combines the embeddings of all the
    edge types pointing to the same node type :math:`d`:

    .. math::
        \mathbf{z}_i = \sum_{\Phi} \beta_{\Phi} \mathbf{z}_i^{\Phi}, \quad
        \beta_{\Phi} = \mathrm{softmax}_{\Phi} \left( \frac{1}{|\mathcal{V}_d|}
        \sum_{i \in \mathcal{V}_d} \mathbf{q}^{\top} \tanh(\mathbf{W}
        \mathbf{z}_i^{\Phi} + \mathbf{b}) \right)

    Node and edge types are given as in :class:`HeteroConv`: node features as a
    dict keyed by node type and edge indices as a dict keyed by
    ``(src, rel, dst)`` tuples. Node types that no edge type points to are not
    in the output (PyG maps them to ``None``).

    Args:
        node_features_dim: Size of input node features, the same for every node
            type, or a dict mapping each node type to its size
        out_features_dim: Size of output node embeddings. Must be divisible by
            ``heads``.
        metadata: The node types and the ``(src, rel, dst)`` edge types of the
            graph, as a ``(node_types, edge_types)`` tuple (PyG's
            ``HeteroData.metadata()``)
        heads: Number of attention heads. Default: ``1``
        negative_slope: Slope of the LeakyReLU of the node-level attention.
            Default: ``0.2``
        dropout: Dropout probability of the node-level attention coefficients.
            Default: ``0.0``

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import HANConv

        x_dict = {"author": mx.ones((4, 16)), "paper": mx.ones((5, 8))}
        edge_index_dict = {
            ("author", "writes", "paper"): mx.array([[0, 1, 3], [0, 0, 4]]),
            ("paper", "cites", "paper"): mx.array([[1, 2], [0, 0]]),
            ("paper", "rev_writes", "author"): mx.array([[0, 0, 4], [0, 1, 3]]),
        }
        metadata = (list(x_dict), list(edge_index_dict))
        conv = HANConv({"author": 16, "paper": 8}, 32, metadata, heads=4)

        out = conv(x_dict, edge_index_dict)
        >>> {k: v.shape for k, v in out.items()}
        {'author': (4, 32), 'paper': (5, 32)}

    References:
        `Wang et al. Heterogeneous Graph Attention Network
        <https://arxiv.org/abs/1903.07293>`_
    """

    def __init__(
        self,
        node_features_dim: Union[int, Dict[str, int]],
        out_features_dim: int,
        metadata: Metadata,
        heads: int = 1,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
    ):
        super().__init__()
        node_types, edge_types = metadata
        if out_features_dim % heads != 0:
            raise ValueError(
                f"'out_features_dim' (got {out_features_dim}) must be divisible "
                f"by the number of heads (got {heads})"
            )
        if not isinstance(node_features_dim, dict):
            node_features_dim = {nt: node_features_dim for nt in node_types}

        self.node_features_dim = dict(node_features_dim)
        self.out_features_dim = out_features_dim
        self.heads = heads
        self.negative_slope = negative_slope
        self.node_types = list(node_types)
        self.edge_types = [tuple(et) for et in edge_types]

        # PyG uses PyTorch's default initialization for these linear layers.
        self.proj = {
            nt: nn.Linear(dim, out_features_dim)
            for nt, dim in self.node_features_dim.items()
        }
        dim = out_features_dim // heads
        # MLX tracks parameters in dicts with string keys only, so edge types
        # are stored as "src__rel__dst".
        self.lin_src = {
            _edge_type_to_key(et): glorot((1, heads, dim)) for et in self.edge_types
        }
        self.lin_dst = {
            _edge_type_to_key(et): glorot((1, heads, dim)) for et in self.edge_types
        }

        self.k_lin = nn.Linear(out_features_dim, out_features_dim)
        self.q = glorot((1, out_features_dim))

        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)

    def __call__(
        self,
        x_dict: Dict[str, mx.array],
        edge_index_dict: Dict[EdgeType, mx.array],
        return_semantic_attention_weights: bool = False,
    ) -> Union[Dict[str, mx.array], Tuple[Dict[str, mx.array], Dict[str, mx.array]]]:
        """Computes new node embeddings for every destination node type.

        Args:
            x_dict: Node features per node type.
            edge_index_dict: Edge index of shape ``[2, num_edges]`` per edge type.
            return_semantic_attention_weights: If ``True``, also returns the
                semantic-level attention weights :math:`\\beta` per destination
                node type, of shape ``[num_edge_types_to_that_node_type]``.
                Default: ``False``

        Returns:
            Node embeddings per destination node type, and the semantic
            attention weights if ``return_semantic_attention_weights`` is
            ``True``.
        """
        H, D = self.heads, self.out_features_dim // self.heads

        x_node_dict = {
            nt: self.proj[nt](x).reshape(-1, H, D) for nt, x in x_dict.items()
        }

        outs: Dict[str, List[mx.array]] = {}
        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            key = _edge_type_to_key(edge_type)
            x_src, x_dst = x_node_dict[src_type], x_node_dict[dst_type]
            src_idx, dst_idx = edge_index[0], edge_index[1]

            alpha_src = (x_src * self.lin_src[key]).sum(-1)  # [N_src, H]
            alpha_dst = (x_dst * self.lin_dst[key]).sum(-1)  # [N_dst, H]
            alpha = alpha_src[src_idx] + alpha_dst[dst_idx]
            alpha = attention_leaky_relu(alpha, self.negative_slope)
            alpha = scatter(alpha, dst_idx, x_dst.shape[0], "softmax")
            if "dropout" in self:
                alpha = self.dropout(alpha)

            messages = x_src[src_idx] * alpha[..., None]  # [E, H, D]
            out = scatter(messages, dst_idx, x_dst.shape[0], "add")
            outs.setdefault(dst_type, []).append(
                nn.relu(out.reshape(-1, self.out_features_dim))
            )

        out_dict, attn_dict = {}, {}
        for node_type, node_outs in outs.items():
            out_dict[node_type], attn_dict[node_type] = self._group(node_outs)

        if return_semantic_attention_weights:
            return out_dict, attn_dict
        return out_dict

    def _group(self, xs: List[mx.array]) -> Tuple[mx.array, Optional[mx.array]]:
        """Semantic-level attention over the edge types of one node type."""
        out = mx.stack(xs)  # [num_edge_types, N, F]
        if out.size == 0:
            return out.reshape(0, out.shape[-1]), None
        score = (self.q * mx.tanh(self.k_lin(out)).mean(axis=1)).sum(-1)
        attn = mx.softmax(score, axis=0)
        return (attn[:, None, None] * out).sum(axis=0), attn
