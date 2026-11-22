import math
from typing import Dict, List, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.nn.conv.hetero_conv import EdgeType, _edge_type_to_key
from mlx_graphs.nn.linear import HeteroDictLinear
from mlx_graphs.utils import scatter

Metadata = Tuple[List[str], List[EdgeType]]


class HGTConv(nn.Module):
    r"""The Heterogeneous Graph Transformer (HGT) operator from the
    `"Heterogeneous Graph Transformer" <https://arxiv.org/abs/2003.01332>`_
    paper.

    Every node type :math:`\tau` has its own key, query and value projections
    :math:`\mathbf{K}_\tau, \mathbf{Q}_\tau, \mathbf{V}_\tau`, and every edge
    type :math:`\phi = (s, r, d)` its own per-head key and value matrices
    :math:`\mathbf{W}^{\textrm{att}}_{\phi}, \mathbf{W}^{\textrm{msg}}_{\phi}`
    and prior :math:`\mu_{\phi}`. For each head, the attention of a node
    :math:`i` over all of its neighbors :math:`j` (of any edge type) is

    .. math::
        \alpha_{i,j} = \mathrm{softmax}_{j} \left( \mu_{\phi(j,i)}
        \frac{(\mathbf{K}_{s} \mathbf{x}_j)^{\top}
        \mathbf{W}^{\textrm{att}}_{\phi(j,i)}
        \mathbf{Q}_{d} \mathbf{x}_i}{\sqrt{D}} \right)

    and the messages :math:`\mathbf{V}_{s} \mathbf{x}_j
    \mathbf{W}^{\textrm{msg}}_{\phi(j,i)}` are summed with these weights. The
    heads are concatenated and passed through a GELU and a node-type specific
    linear layer. When the input and output sizes match, the result is mixed
    with the input with a learned gate :math:`\sigma(s_\tau)`.

    Node and edge types are given as in :class:`HeteroConv`: node features as a
    dict keyed by node type and edge indices as a dict keyed by
    ``(src, rel, dst)`` tuples. Only node types that are the destination of an
    edge type of ``metadata`` are in the output.

    Args:
        node_features_dim: Size of input node features, the same for every node
            type, or a dict mapping each node type to its size
        out_features_dim: Size of output node embeddings. Must be divisible by
            ``heads``.
        metadata: The node types and the ``(src, rel, dst)`` edge types of the
            graph, as a ``(node_types, edge_types)`` tuple (PyG's
            ``HeteroData.metadata()``)
        heads: Number of attention heads. Default: ``1``

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import HGTConv

        x_dict = {"author": mx.ones((4, 16)), "paper": mx.ones((5, 8))}
        edge_index_dict = {
            ("author", "writes", "paper"): mx.array([[0, 1, 3], [0, 0, 4]]),
            ("paper", "cites", "paper"): mx.array([[1, 2], [0, 0]]),
            ("paper", "rev_writes", "author"): mx.array([[0, 0, 4], [0, 1, 3]]),
        }
        metadata = (list(x_dict), list(edge_index_dict))
        conv = HGTConv({"author": 16, "paper": 8}, 32, metadata, heads=4)

        out = conv(x_dict, edge_index_dict)
        >>> {k: v.shape for k, v in out.items()}
        {'author': (4, 32), 'paper': (5, 32)}

    References:
        `Hu et al. Heterogeneous Graph Transformer
        <https://arxiv.org/abs/2003.01332>`_
    """

    def __init__(
        self,
        node_features_dim: Union[int, Dict[str, int]],
        out_features_dim: int,
        metadata: Metadata,
        heads: int = 1,
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
        self.node_types = list(node_types)
        self.edge_types = [tuple(et) for et in edge_types]
        # Destination node types, in the order of the metadata.
        self.dst_node_types = list(dict.fromkeys(et[-1] for et in self.edge_types))

        self.kqv_lin = HeteroDictLinear(self.node_features_dim, 3 * out_features_dim)
        self.out_lin = HeteroDictLinear(
            {nt: out_features_dim for nt in self.node_types}, out_features_dim
        )

        D = out_features_dim // heads
        # Per edge type and head [D, D] matrices, initialized like PyG's
        # HeteroLinear. MLX tracks parameters in dicts with string keys only,
        # so edge types are stored as "src__rel__dst".
        bound = 1.0 / math.sqrt(D)
        keys = [_edge_type_to_key(et) for et in self.edge_types]
        self.k_rel = {k: mx.random.uniform(-bound, bound, (heads, D, D)) for k in keys}
        self.v_rel = {k: mx.random.uniform(-bound, bound, (heads, D, D)) for k in keys}
        self.p_rel = {k: mx.ones((1, heads)) for k in keys}
        self.skip = {nt: mx.ones((1,)) for nt in self.node_types}

    def __call__(
        self,
        x_dict: Dict[str, mx.array],
        edge_index_dict: Dict[EdgeType, mx.array],
    ) -> Dict[str, mx.array]:
        """Computes new node embeddings for every destination node type.

        Args:
            x_dict: Node features per node type.
            edge_index_dict: Edge index of shape ``[2, num_edges]`` per edge type.

        Returns:
            Node embeddings per destination node type.
        """
        F, H = self.out_features_dim, self.heads
        D = F // H

        k_dict, q_dict, v_dict = {}, {}, {}
        for node_type, kqv in self.kqv_lin(x_dict).items():
            k, q, v = mx.split(kqv, 3, axis=1)
            k_dict[node_type] = k.reshape(-1, H, D)
            q_dict[node_type] = q.reshape(-1, H, D)
            v_dict[node_type] = v.reshape(-1, H, D)

        # Attention logits, values and destinations of the edges of every
        # destination node type: the softmax runs over all incoming edges.
        incoming: Dict[str, List[Tuple[mx.array, mx.array, mx.array]]] = {}
        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            key = _edge_type_to_key(edge_type)
            src_idx, dst_idx = edge_index[0], edge_index[1]

            # [N, H, D] x [H, D, D] -> [N, H, D]
            k = mx.einsum("nhd,hde->nhe", k_dict[src_type], self.k_rel[key])
            v = mx.einsum("nhd,hde->nhe", v_dict[src_type], self.v_rel[key])

            alpha = (q_dict[dst_type][dst_idx] * k[src_idx]).sum(-1)  # [E, H]
            alpha = alpha * self.p_rel[key] / math.sqrt(D)
            incoming.setdefault(dst_type, []).append((alpha, v[src_idx], dst_idx))

        out_dict = {}
        for node_type in self.dst_node_types:
            if node_type not in q_dict:
                continue
            num_nodes = q_dict[node_type].shape[0]
            if node_type in incoming:
                alpha, values, dst_idx = (
                    mx.concatenate(parts, axis=0) for parts in zip(*incoming[node_type])
                )
                alpha = scatter(alpha, dst_idx, num_nodes, "softmax")
                out = scatter(values * alpha[..., None], dst_idx, num_nodes, "add")
                out = out.reshape(num_nodes, F)
            else:
                out = mx.zeros((num_nodes, F), dtype=q_dict[node_type].dtype)

            out = self.out_lin.lins[node_type](nn.gelu(out))
            x = x_dict[node_type]
            if out.shape[-1] == x.shape[-1]:
                gate = mx.sigmoid(self.skip[node_type])
                out = gate * out + (1 - gate) * x
            out_dict[node_type] = out

        return out_dict
