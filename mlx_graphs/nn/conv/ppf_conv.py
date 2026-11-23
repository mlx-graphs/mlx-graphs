from typing import Callable, Optional

import mlx.core as mx

from mlx_graphs.nn.conv.point_conv import (
    OptPair,
    Pair,
    _pair,
    _run_mlp,
    _set_self_loops,
)
from mlx_graphs.nn.message_passing import MessagePassing


def _norm(v: mx.array) -> mx.array:
    """Euclidean norm of the rows of ``v``, with a zero gradient at zero (as
    in PyTorch) instead of NaN."""
    squared = (v * v).sum(axis=1)
    nonzero = squared > 0
    return mx.where(nonzero, mx.sqrt(mx.where(nonzero, squared, 1.0)), 0.0)


def get_angle(v1: mx.array, v2: mx.array) -> mx.array:
    """Angle between the rows of ``v1`` and ``v2`` (of shape ``[N, 3]``), in
    :math:`[0, \\pi]`, computed with ``atan2`` as in PyG (``0`` if a vector is
    zero)."""
    y = _norm(mx.linalg.cross(v1, v2, axis=1))
    x = (v1 * v2).sum(axis=1)
    zero = (y == 0) & (x == 0)
    return mx.arctan2(mx.where(zero, 0.0, y), mx.where(zero, 1.0, x))


def point_pair_features(
    pos_i: mx.array, pos_j: mx.array, normal_i: mx.array, normal_j: mx.array
) -> mx.array:
    r"""Point pair features of the `"PPFNet: Global Context Aware Local
    Features for Robust 3D Point Matching" <https://arxiv.org/abs/1802.02669>`_
    paper:

    .. math::
        \left( \| \mathbf{d}_{j,i} \|, \angle(\mathbf{n}_i, \mathbf{d}_{j,i}),
        \angle(\mathbf{n}_j, \mathbf{d}_{j,i}), \angle(\mathbf{n}_i,
        \mathbf{n}_j) \right), \quad \mathbf{d}_{j,i} = \mathbf{p}_j -
        \mathbf{p}_i.

    Args:
        pos_i: Positions of the first points, of shape ``[E, 3]``.
        pos_j: Positions of the second points, of shape ``[E, 3]``.
        normal_i: Normals of the first points, of shape ``[E, 3]``.
        normal_j: Normals of the second points, of shape ``[E, 3]``.

    Returns:
        The features, of shape ``[E, 4]``.
    """
    pseudo = pos_j - pos_i
    return mx.stack(
        [
            _norm(pseudo),
            get_angle(normal_i, pseudo),
            get_angle(normal_j, pseudo),
            get_angle(normal_i, normal_j),
        ],
        axis=1,
    )


class PPFConv(MessagePassing):
    r"""The PPFNet operator from the `"PPFNet: Global Context Aware Local
    Features for Robust 3D Point Matching" <https://arxiv.org/abs/1802.02669>`_
    paper.

    .. math::
        \mathbf{x}^{\prime}_i = \gamma_{\mathbf{\Theta}} \left( \max_{j \in
        \mathcal{N}(i) \cup \{ i \}} h_{\mathbf{\Theta}} ( \mathbf{x}_j, \|
        \mathbf{d_{j,i}} \|, \angle(\mathbf{n}_i, \mathbf{d_{j,i}}),
        \angle(\mathbf{n}_j, \mathbf{d_{j,i}}), \angle(\mathbf{n}_i,
        \mathbf{n}_j) \right)

    where :math:`\mathbf{d_{j,i}}` is the distance vector between the positions
    of the points, and :math:`\mathbf{n}` are their normals.

    Args:
        local_nn: Network :math:`h_{\mathbf{\Theta}}` applied to each edge, on
            ``[x_j, ppf]`` (the 4 point pair features, after the source node
            features if any). Default: ``None``
        global_nn: Network :math:`\gamma_{\mathbf{\Theta}}` applied to the
            aggregated features. Default: ``None``
        add_self_loops: If ``True``, removes the self-loops of the graph and
            adds one for each destination node. Default: ``True``
        **kwargs: Arguments of :class:`MessagePassing` (``aggr="max"`` by
            default).

    Example:

    .. code-block:: python

        import mlx.core as mx
        import mlx.nn as nn
        from mlx_graphs.nn import PPFConv
        from mlx_graphs.utils import knn_graph

        pos = mx.random.normal((100, 3))
        normal = mx.random.normal((100, 3))
        x = mx.random.normal((100, 16))
        conv = PPFConv(nn.Sequential(nn.Linear(16 + 4, 32), nn.ReLU()))
        >>> conv(knn_graph(pos, k=8), x, pos, normal).shape
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
        normal: Pair,
    ) -> mx.array:
        """Computes the forward pass of PPFConv.

        Args:
            edge_index: Edges of shape ``[2, num_edges]``, from the neighbors
                (sources) to the center points (destinations).
            node_features: Features of the points, a ``(src_features,
                dst_features)`` tuple (only the source features are used), or
                ``None``.
            pos: Positions of the points, of shape ``[N, 3]``, or a
                ``(src_pos, dst_pos)`` tuple.
            normal: Normals of the points, of shape ``[N, 3]``, or a
                ``(src_normal, dst_normal)`` tuple.

        Returns:
            The features of the destination points.
        """
        x_src = node_features[0] if isinstance(node_features, tuple) else node_features
        pos_src, pos_dst = _pair(pos)
        normal_src, normal_dst = _pair(normal)
        if self.add_self_loops:
            edge_index = _set_self_loops(edge_index, pos_dst.shape[0])
        src, dst = edge_index[0], edge_index[1]
        msg = point_pair_features(
            pos_dst[dst], pos_src[src], normal_dst[dst], normal_src[src]
        )
        if x_src is not None:
            msg = mx.concatenate([x_src[src], msg], axis=1)
        msg = _run_mlp(self.local_nn, msg)
        self.num_nodes = pos_dst.shape[0]
        out = self.update_nodes(self.aggregate(msg, dst))
        if self.global_nn is not None:
            out = self.global_nn(out)
        return out
