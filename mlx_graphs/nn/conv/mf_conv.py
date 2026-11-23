from typing import Tuple, Union

import mlx.core as mx
from mlx.nn import Linear  # PyTorch's default initialization, as in PyG

from mlx_graphs.nn.message_passing import MessagePassing, fused_message_passing
from mlx_graphs.utils import degree


@fused_message_passing
class MFConv(MessagePassing):
    r"""Graph convolution layer from the `"Convolutional Networks on Graphs for
    Learning Molecular Fingerprints" <https://arxiv.org/abs/1509.09292>`_
    paper.

    .. math::
        \mathbf{h}_i = \mathbf{W}^{(\deg(i))}_1 \mathbf{x}_i +
        \mathbf{W}^{(\deg(i))}_2 \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j

    which trains a distinct pair of weight matrices for each possible node
    (in-)degree, with degrees larger than ``max_degree`` sharing the weights of
    ``max_degree``.

    Args:
        node_features_dim: Size of input node features, or a ``(src_dim,
            dst_dim)`` tuple for bipartite graphs
        out_features_dim: Size of output node embeddings
        max_degree: Largest degree with its own weights. Default: ``10``
        bias: Whether the neighbour projections use a bias. Default: ``True``
        **kwargs: Arguments of :class:`MessagePassing`, e.g. ``aggr``
            (default: ``"add"``)

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import MFConv

        edge_index = mx.array([[0, 1, 2, 3, 4], [1, 0, 3, 2, 0]])
        node_features = mx.random.normal((5, 16))

        conv = MFConv(16, 32, max_degree=4)
        h = conv(edge_index, node_features)
        >>> h.shape
        (5, 32)

    References:
        [1] `Duvenaud et al. Convolutional Networks on Graphs for Learning
        Molecular Fingerprints. <https://arxiv.org/abs/1509.09292>`_
    """

    # what follows the aggregation expects the size of one aggregation (as in PyG)
    supports_concat_aggr = False

    def __init__(
        self,
        node_features_dim: Union[int, Tuple[int, int]],
        out_features_dim: int,
        max_degree: int = 10,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super(MFConv, self).__init__(**kwargs)

        self.node_features_dim = node_features_dim
        self.out_features_dim = out_features_dim
        self.max_degree = max_degree

        if isinstance(node_features_dim, int):
            src_dim = dst_dim = node_features_dim
        else:
            src_dim, dst_dim = node_features_dim

        # Same names as PyG: `lins_l` for the neighbours, `lins_r` for the root.
        self.lins_l = [
            Linear(src_dim, out_features_dim, bias=bias) for _ in range(max_degree + 1)
        ]
        self.lins_r = [
            Linear(dst_dim, out_features_dim, bias=False) for _ in range(max_degree + 1)
        ]

    def __call__(
        self,
        edge_index: mx.array,
        node_features: Union[mx.array, Tuple[mx.array, mx.array]],
    ) -> mx.array:
        """Computes the forward pass of MFConv.

        Args:
            edge_index: Input edge index of shape `[2, num_edges]`
            node_features: Input node features, or a ``(src_features,
                dst_features)`` tuple for bipartite graphs

        Returns:
            mx.array: The computed node embeddings, one row per destination node
        """
        if not isinstance(node_features, tuple):
            node_features = (node_features, node_features)
        dst_features = node_features[1]

        num_nodes = dst_features.shape[0]
        deg = degree(edge_index[1], num_nodes)
        deg = mx.minimum(deg, self.max_degree).astype(mx.uint32)

        h = self.propagate(edge_index=edge_index, node_features=node_features)

        # Every node uses the weights of its degree: one matrix-vector product
        # per node with the weights selected by `gather_mm`, i.e. the work of
        # a single linear layer (as PyG, which projects the nodes of each
        # degree separately), without building per-node weight matrices.
        weight = mx.concatenate(
            [
                mx.stack([lin.weight for lin in self.lins_l]),
                mx.stack([lin.weight for lin in self.lins_r]),
            ],
            axis=-1,
        )  # [max_degree + 1, out, src_dim + dst_dim]
        features = mx.concatenate([h, dst_features], axis=-1)[:, None, :]
        # Nodes sorted by degree: gather_mm (and its gradient with respect to
        # the weights) is much faster on sorted indices.
        order = mx.argsort(deg)
        out = mx.gather_mm(
            features[order],
            weight.swapaxes(-1, -2),
            None,
            deg[order],
            sorted_indices=True,
        )[:, 0, :]
        out = out[mx.argsort(order)]
        if "bias" in self.lins_l[0]:
            out = out + mx.stack([lin.bias for lin in self.lins_l])[deg]
        return out
