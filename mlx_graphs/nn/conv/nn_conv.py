from typing import Callable, Tuple, Union

import mlx.core as mx
from mlx.nn import Linear  # uniform initialization, as PyG's root weight

from mlx_graphs.nn.message_passing import MessagePassing


class NNConv(MessagePassing):
    r"""Edge-conditioned convolution layer from the `"Neural Message Passing
    for Quantum Chemistry" <https://arxiv.org/abs/1704.01212>`_ paper (also
    known as ECConv, from `"Dynamic Edge-Conditioned Filters in Convolutional
    Neural Networks on Graphs" <https://arxiv.org/abs/1704.02901>`_).

    .. math::
        \mathbf{h}_i = \mathbf{\Theta} \mathbf{x}_i + \bigoplus_{j \in
        \mathcal{N}(i)} \mathbf{x}_j \cdot h_{\mathbf{\Theta}}(\mathbf{e}_{j,i})
        + \mathbf{b}

    where :math:`h_{\mathbf{\Theta}}` is a neural network (``mlp``) that maps
    the features :math:`\mathbf{e}_{j,i}` of each edge to a
    ``[node_features_dim, out_features_dim]`` weight matrix (flattened, i.e. an
    output of size ``node_features_dim * out_features_dim``).

    Args:
        node_features_dim: Size of input node features, or a ``(src_dim,
            dst_dim)`` tuple for bipartite graphs
        out_features_dim: Size of output node embeddings
        mlp: Module mapping edge features of shape `[num_edges, edge_dim]` to
            weights of shape `[num_edges, src_dim * out_features_dim]`
        aggr: Aggregation :math:`\bigoplus`. Default: ``"add"``
        root_weight: Whether to add the transformed root node features
            :math:`\mathbf{\Theta} \mathbf{x}_i`. Default: ``True``
        bias: Whether to add a bias to the output. Default: ``True``
        **kwargs: Arguments of :class:`MessagePassing`

    Example:

    .. code-block:: python

        import mlx.core as mx
        import mlx.nn as nn
        from mlx_graphs.nn import NNConv

        edge_index = mx.array([[0, 1, 2, 3, 4], [1, 0, 3, 2, 0]])
        node_features = mx.random.normal((5, 16))
        edge_features = mx.random.normal((5, 4))

        mlp = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 16 * 8))
        conv = NNConv(16, 8, mlp, aggr="mean")
        h = conv(edge_index, node_features, edge_features)
        >>> h.shape
        (5, 8)

    References:
        [1] `Gilmer et al. Neural Message Passing for Quantum Chemistry.
        <https://arxiv.org/abs/1704.01212>`_

        [2] `Simonovsky and Komodakis. Dynamic Edge-Conditioned Filters in
        Convolutional Neural Networks on Graphs.
        <https://arxiv.org/abs/1704.02901>`_
    """

    # what follows the aggregation expects the size of one aggregation (as in PyG)
    supports_concat_aggr = False

    def __init__(
        self,
        node_features_dim: Union[int, Tuple[int, int]],
        out_features_dim: int,
        mlp: Callable[[mx.array], mx.array],
        aggr: str = "add",
        root_weight: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        super(NNConv, self).__init__(aggr=aggr, **kwargs)

        self.node_features_dim = node_features_dim
        self.out_features_dim = out_features_dim
        self.mlp = mlp
        self.root_weight = root_weight

        if isinstance(node_features_dim, int):
            src_dim = dst_dim = node_features_dim
        else:
            src_dim, dst_dim = node_features_dim
        self.src_features_dim = src_dim

        if root_weight:
            self.lin = Linear(dst_dim, out_features_dim, bias=False)
        if bias:
            self.bias = mx.zeros((out_features_dim,))

    def __call__(
        self,
        edge_index: mx.array,
        node_features: Union[mx.array, Tuple[mx.array, mx.array]],
        edge_features: mx.array,
    ) -> mx.array:
        """Computes the forward pass of NNConv.

        Args:
            edge_index: Input edge index of shape `[2, num_edges]`
            node_features: Input node features, or a ``(src_features,
                dst_features)`` tuple for bipartite graphs
            edge_features: Edge features of shape `[num_edges, edge_dim]`

        Returns:
            mx.array: The computed node embeddings, one row per destination node
        """
        if isinstance(node_features, tuple):
            dst_features = node_features[1]
        else:
            dst_features = node_features
            node_features = (node_features, node_features)

        out = self.propagate(
            edge_index=edge_index,
            node_features=node_features,
            message_kwargs={"edge_features": edge_features},
        )

        if self.root_weight:
            out = out + self.lin(dst_features)
        if "bias" in self:
            out = out + self.bias
        return out

    def message(
        self,
        src_features: mx.array,
        dst_features: mx.array,
        edge_features: mx.array,
        **kwargs,
    ) -> mx.array:
        # MLX 0.32 can crash (segfault) when a compiled function, such as
        # nn.softplus or nn.relu, runs on the output of an empty matmul:
        # without edges, the edge network is run on a single dummy row instead.
        num_edges = edge_features.shape[0]
        if num_edges == 0:
            edge_features = mx.zeros((1, *edge_features.shape[1:]), edge_features.dtype)
        weight = self.mlp(edge_features)[:num_edges].reshape(
            num_edges, self.src_features_dim, self.out_features_dim
        )
        return (src_features[:, None, :] @ weight)[:, 0, :]


ECConv = NNConv
