from typing import Callable, Optional, Tuple, Union

import mlx.core as mx
import numpy as np

from mlx_graphs.nn.message_passing import MessagePassing
from mlx_graphs.utils.transformations import _to_numpy

OptPair = Union[Optional[mx.array], Tuple[Optional[mx.array], Optional[mx.array]]]
Pair = Union[mx.array, Tuple[mx.array, mx.array]]


def _pair(value):
    return value if isinstance(value, tuple) else (value, value)


def _set_self_loops(edge_index: mx.array, num_nodes: int) -> mx.array:
    """Removes the self-loops of ``edge_index`` and adds one for each of the
    first ``num_nodes`` nodes (at the end), as PyG's point convolutions do."""
    ei = _to_numpy(edge_index)
    ei = ei[:, ei[0] != ei[1]]
    loops = np.arange(num_nodes, dtype=ei.dtype)
    ei = np.concatenate([ei, np.stack([loops, loops])], axis=1)
    return mx.array(ei.astype(np.int32))


def _run_mlp(mlp: Optional[Callable], features: mx.array) -> mx.array:
    """Applies ``mlp`` to per-edge features, also without edges.

    MLX 0.32 can crash (segfault) when a compiled function, such as nn.relu,
    runs on the output of an empty matmul: without edges, the MLP is run on a
    single dummy row instead.
    """
    if mlp is None:
        return features
    num_edges = features.shape[0]
    if num_edges == 0:
        features = mx.zeros((1, *features.shape[1:]), dtype=features.dtype)
    return mlp(features)[:num_edges]


class PointNetConv(MessagePassing):
    r"""The PointNet set layer from the `"PointNet: Deep Learning on Point Sets
    for 3D Classification and Segmentation" <https://arxiv.org/abs/1612.00593>`_
    and `"PointNet++: Deep Hierarchical Feature Learning on Point Sets in a
    Metric Space" <https://arxiv.org/abs/1706.02413>`_ papers.

    .. math::
        \mathbf{x}^{\prime}_i = \gamma_{\mathbf{\Theta}} \left( \max_{j \in
        \mathcal{N}(i) \cup \{ i \}} h_{\mathbf{\Theta}} ( \mathbf{x}_j,
        \mathbf{p}_j - \mathbf{p}_i) \right),

    where :math:`\gamma_{\mathbf{\Theta}}` (``global_nn``) and
    :math:`h_{\mathbf{\Theta}}` (``local_nn``) are neural networks and
    :math:`\mathbf{p}_j \in \mathbb{R}^3` are the positions of the points.

    Args:
        local_nn: Network :math:`h_{\mathbf{\Theta}}` applied to each edge, on
            the concatenation ``[x_j, pos_j - pos_i]`` (or ``pos_j - pos_i``
            only without node features). Default: ``None``
        global_nn: Network :math:`\gamma_{\mathbf{\Theta}}` applied to the
            aggregated features. Default: ``None``
        add_self_loops: If ``True``, removes the self-loops of the graph and
            adds one for each node (for the first
            ``min(num_src_nodes, num_dst_nodes)`` nodes of a bipartite graph,
            as in PyG). Default: ``True``
        **kwargs: Arguments of :class:`MessagePassing`, e.g. ``aggr``
            (``"max"`` by default).

    Example:

    .. code-block:: python

        import mlx.core as mx
        import mlx.nn as nn
        from mlx_graphs.nn import PointNetConv
        from mlx_graphs.utils import knn_graph

        pos = mx.random.normal((100, 3))
        x = mx.random.normal((100, 16))
        edge_index = knn_graph(pos, k=8)
        local_nn = nn.Sequential(nn.Linear(16 + 3, 32), nn.ReLU(), nn.Linear(32, 32))
        conv = PointNetConv(local_nn)
        >>> conv(edge_index, x, pos).shape
        (100, 32)
    """

    def __init__(
        self,
        local_nn: Optional[Callable] = None,
        global_nn: Optional[Callable] = None,
        add_self_loops: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "max")
        super().__init__(**kwargs)
        self.local_nn = local_nn
        self.global_nn = global_nn
        self.add_self_loops = add_self_loops

    def __call__(
        self,
        edge_index: mx.array,
        node_features: OptPair,
        pos: Pair,
    ) -> mx.array:
        """Computes the forward pass of PointNetConv.

        Args:
            edge_index: Edges of shape ``[2, num_edges]``, from the neighbors
                (sources) to the center points (destinations).
            node_features: Features of the points, of shape ``[N, F]``, a
                ``(src_features, dst_features)`` tuple for bipartite graphs
                (only the source features are used), or ``None``.
            pos: Positions of the points, of shape ``[N, 3]``, or a
                ``(src_pos, dst_pos)`` tuple for bipartite graphs.

        Returns:
            The features of the destination points.
        """
        x_src = node_features[0] if isinstance(node_features, tuple) else node_features
        pos = _pair(pos)
        if self.add_self_loops:
            edge_index = _set_self_loops(
                edge_index, min(pos[0].shape[0], pos[1].shape[0])
            )
        x_j = None if x_src is None else x_src[edge_index[0]]
        out = self.propagate(
            edge_index=edge_index,
            node_features=pos,
            message_kwargs={"x_j": x_j},
        )
        if self.global_nn is not None:
            out = self.global_nn(out)
        return out

    def message(
        self,
        src_features: mx.array,
        dst_features: mx.array,
        x_j: Optional[mx.array] = None,
        **kwargs,
    ) -> mx.array:
        msg = src_features - dst_features
        if x_j is not None:
            msg = mx.concatenate([x_j, msg], axis=1)
        return _run_mlp(self.local_nn, msg)


PointConv = PointNetConv
