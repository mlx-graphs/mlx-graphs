from typing import Union

import mlx.core as mx
import numpy as np

from mlx_graphs.data import GraphData
from mlx_graphs.transforms.base_transform import BaseTransform
from mlx_graphs.transforms.features import (
    _edge_weights,
    _host_edges,
    _weighted_bincount,
)
from mlx_graphs.transforms.structure import _num_nodes


def _sym_norm_weights(
    src: np.ndarray, dst: np.ndarray, weights: np.ndarray, num_nodes: int
) -> np.ndarray:
    r"""Weights of :math:`\mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2}`, with
    :math:`\mathbf{D}` the weighted in-degree (no self-loops added)."""
    in_degree = _weighted_bincount(dst, weights, num_nodes)
    deg_inv_sqrt = np.zeros_like(in_degree)
    np.power(in_degree, -0.5, out=deg_inv_sqrt, where=in_degree > 0)
    return deg_inv_sqrt[src] * weights * deg_inv_sqrt[dst]


def _propagate(
    features: mx.array, src: mx.array, dst: mx.array, weights: mx.array, size: int
) -> mx.array:
    """Sums the weighted features of the source nodes of each edge into its
    destination node."""
    messages = features[src] * weights.reshape(-1, *([1] * (features.ndim - 1)))
    out = mx.zeros((size, *features.shape[1:]), dtype=messages.dtype)
    return out.at[dst].add(messages)


class SIGN(BaseTransform):
    r"""Precomputes the propagated node features of the Scalable Inception
    Graph Neural Network (SIGN) from `"SIGN: Scalable Inception Graph Neural
    Networks" <https://arxiv.org/abs/2004.11198>`_, as PyG's ``SIGN``:

    .. math::
        \mathbf{X}^{(i)} = {\left( \mathbf{D}^{-1/2} \mathbf{A}
        \mathbf{D}^{-1/2} \right)}^i \mathbf{X}

    for :math:`i \in \{ 1, \ldots, K \}`, stored in the attributes ``x1``,
    ``x2``, ... ``xK``. :math:`\mathbf{D}` is the weighted in-degree, the
    ``edge_weights`` attribute is used as :math:`\mathbf{A}` if it exists, and
    no self-loops are added. Messages flow from the source to the
    destination of each edge.

    Args:
        K: Number of hops.

    Example:

    .. code-block:: python

        graph = GraphData(
            edge_index=mx.array([[0, 1], [1, 0]]),
            node_features=mx.array([[1.0], [2.0]]),
        )
        graph = SIGN(K=2)(graph)
        graph.x1, graph.x2
        # mx.array([[2.0], [1.0]]), mx.array([[1.0], [2.0]])
    """

    def __init__(self, K: int):
        self.K = K

    def process(self, data: GraphData) -> GraphData:
        """Adds the propagated node features ``x1`` ... ``xK`` to ``data``.

        Args:
            data: The graph to transform.

        Returns:
            The graph with the propagated features.
        """
        if data.node_features is None:
            raise ValueError("SIGN requires node features")
        num_nodes = _num_nodes(data)
        src, dst = _host_edges(data)
        weights = _sym_norm_weights(src, dst, _edge_weights(data), num_nodes)
        src_mx, dst_mx = mx.array(src), mx.array(dst)
        weights_mx = mx.array(weights.astype(np.float32))

        features = data.node_features
        for i in range(1, self.K + 1):
            features = _propagate(features, src_mx, dst_mx, weights_mx, num_nodes)
            setattr(data, f"x{i}", features)
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(K={self.K})"


class FeaturePropagation(BaseTransform):
    r"""Fills in missing node features by propagating the known ones, with the
    feature propagation algorithm from `"On the Unreasonable Effectiveness of
    Feature propagation in Learning on Graphs with Missing Node Features"
    <https://arxiv.org/abs/2111.12128>`_, as PyG's ``FeaturePropagation``:

    .. math::
        \mathbf{X}^{(0)} &= (1 - \mathbf{M}) \cdot \mathbf{X}

        \mathbf{X}^{(\ell + 1)} &= \mathbf{X}^{(0)} + \mathbf{M} \cdot
        (\mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2} \mathbf{X}^{(\ell)})

    where :math:`\mathbf{M}` is the mask of the missing features and
    :math:`\mathbf{D}` is the weighted in-degree. The edge weights are taken
    from ``edge_weights`` if it exists, otherwise from ``edge_features`` if it
    holds one scalar per edge (as PyG uses ``edge_attr``), and are ones
    otherwise. No self-loops are added, and the weights of duplicate edges are
    summed (PyG instead collapses duplicate edges when the graph has no
    weights).

    Args:
        missing_mask: Boolean mask of shape ``[num_nodes, num_node_features]``
            that is ``True`` for the missing node features, e.g.
            ``mx.isnan(graph.node_features)``.
        num_iterations: Number of propagation steps. Default: ``40``

    Example:

    .. code-block:: python

        node_features = mx.array([[1.0], [float("nan")], [3.0]])
        graph = GraphData(
            edge_index=mx.array([[0, 1, 1, 2], [1, 0, 2, 1]]),
            node_features=node_features,
        )
        graph = FeaturePropagation(mx.isnan(node_features))(graph)
        graph.node_features
        # mx.array([[1.0], [2.8284], [3.0]]), i.e. (1 + 3) / sqrt(2)
    """

    def __init__(
        self, missing_mask: Union[mx.array, np.ndarray], num_iterations: int = 40
    ):
        self.missing_mask = mx.array(missing_mask).astype(mx.bool_)
        self.num_iterations = num_iterations

    def process(self, data: GraphData) -> GraphData:
        """Fills in the missing node features of ``data``.

        Args:
            data: The graph to transform.

        Returns:
            The graph with the completed ``node_features``.
        """
        features = data.node_features
        if features is None:
            raise ValueError("FeaturePropagation requires node features")
        if tuple(features.shape) != tuple(self.missing_mask.shape):
            raise ValueError(
                f"The missing mask has shape {tuple(self.missing_mask.shape)} but "
                f"the node features have shape {tuple(features.shape)}"
            )
        num_nodes = _num_nodes(data)
        src, dst = _host_edges(data)
        if getattr(data, "edge_weights", None) is not None:
            weights = _edge_weights(data)
        elif (
            data.edge_features is not None and data.edge_features.size == data.num_edges
        ):
            weights = np.asarray(data.edge_features).reshape(-1).astype(np.float64)
        else:
            weights = np.ones(data.num_edges, dtype=np.float64)
        weights = _sym_norm_weights(src, dst, weights, num_nodes)
        src_mx, dst_mx = mx.array(src), mx.array(dst)
        weights_mx = mx.array(weights.astype(np.float32))

        missing = self.missing_mask
        known = mx.where(missing, mx.zeros_like(features), features)
        out = known
        for _ in range(self.num_iterations):
            out = _propagate(out, src_mx, dst_mx, weights_mx, num_nodes)
            out = mx.where(missing, out, known)
        data.node_features = out.astype(features.dtype)
        return data

    def __repr__(self) -> str:
        missing = mx.mean(self.missing_mask.astype(mx.float32)).item()
        return (
            f"{self.__class__.__name__}(missing_features={100 * missing:.1f}%, "
            f"num_iterations={self.num_iterations})"
        )
