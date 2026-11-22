from typing import Optional

import mlx.core as mx
import numpy as np

from mlx_graphs.data import GraphData
from mlx_graphs.transforms.base_transform import BaseTransform
from mlx_graphs.transforms.features import _add_node_features, _host_edges
from mlx_graphs.transforms.structure import _num_nodes


class TargetIndegree(BaseTransform):
    r"""Saves the globally normalized in-degree of the target node of each edge
    in its edge features, as PyG's ``TargetIndegree``:

    .. math::
        \mathbf{u}(i,j) = \frac{\deg(j)}{\max_{v \in \mathcal{V}} \deg(v)}

    Args:
        norm: Whether to normalize the degrees. Default: ``True``
        max_value: The value to divide the degrees by if ``norm`` is
            ``True``. Defaults to the maximum in-degree of the graph.
        cat: If ``True``, the degrees are appended as a new column to the
            existing edge features, otherwise they replace them.
            Default: ``True``

    Example:

    .. code-block:: python

        graph = GraphData(edge_index=mx.array([[0, 1, 2], [1, 2, 1]]))
        graph = TargetIndegree()(graph)
        graph.edge_features
        # mx.array([[1.0], [0.5], [1.0]])
    """

    def __init__(
        self, norm: bool = True, max_value: Optional[float] = None, cat: bool = True
    ):
        self.norm = norm
        self.max_value = max_value
        self.cat = cat

    def process(self, data: GraphData) -> GraphData:
        """Adds the target in-degrees to the edge features of ``data``.

        Args:
            data: The graph to transform.

        Returns:
            The graph with the updated ``edge_features``.
        """
        col = _host_edges(data)[1]
        degree = np.bincount(col, minlength=_num_nodes(data)).astype(np.float64)
        if self.norm:
            scale = degree.max(initial=0) if self.max_value is None else self.max_value
            with np.errstate(divide="ignore", invalid="ignore"):
                degree = degree / scale
        pseudo = mx.array(degree[col].astype(np.float32).reshape(-1, 1))

        existing = data.edge_features
        if existing is not None and self.cat:
            if existing.ndim == 1:
                existing = existing.reshape(-1, 1)
            pseudo = mx.concatenate([existing, pseudo.astype(existing.dtype)], axis=-1)
        data.edge_features = pseudo
        return data

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(norm={self.norm}, max_value={self.max_value})"
        )


class LocalDegreeProfile(BaseTransform):
    r"""Appends the Local Degree Profile (LDP) from `"A Simple yet Effective
    Baseline for Non-attribute Graph Classification"
    <https://arxiv.org/abs/1811.03508>`_ to the node features, as PyG's
    ``LocalDegreeProfile``:

    .. math::
        \mathbf{x}_i = \mathbf{x}_i \, \Vert \, (\deg(i), \min(DN(i)),
        \max(DN(i)), \textrm{mean}(DN(i)), \textrm{std}(DN(i)))

    where :math:`\deg(i)` is the out-degree of node :math:`i` and
    :math:`DN(i) = \{ \deg(j) \mid (i, j) \in \mathcal{E} \}` holds the degrees
    of its neighbors. The statistics of a node without neighbors are zeros.
    As in PyG, the standard deviation is the population one, and values below
    :math:`\sqrt{10^{-5}}` are set to zero.

    Example:

    .. code-block:: python

        graph = GraphData(edge_index=mx.array([[0, 1, 1, 2], [1, 0, 2, 1]]))
        graph = LocalDegreeProfile()(graph)
        graph.node_features
        # mx.array([[1, 2, 2, 2, 0], [2, 1, 1, 1, 0], [1, 2, 2, 2, 0]])
    """

    def process(self, data: GraphData) -> GraphData:
        """Appends the local degree profile to the node features of ``data``.

        Args:
            data: The graph to transform.

        Returns:
            The graph with the updated ``node_features``.
        """
        num_nodes = _num_nodes(data)
        row, col = _host_edges(data)
        degree = np.bincount(row, minlength=num_nodes).astype(np.float64)
        neighbor_degree = degree[col]

        count = np.maximum(degree, 1.0)
        minimum = np.full(num_nodes, np.inf)
        maximum = np.full(num_nodes, -np.inf)
        np.minimum.at(minimum, row, neighbor_degree)
        np.maximum.at(maximum, row, neighbor_degree)
        minimum[degree == 0] = 0.0
        maximum[degree == 0] = 0.0
        mean = np.bincount(row, neighbor_degree, minlength=num_nodes) / count
        mean_sq = np.bincount(row, neighbor_degree**2, minlength=num_nodes) / count
        std = np.sqrt(np.maximum(mean_sq - mean**2, 1e-5))
        std[std <= np.sqrt(1e-5)] = 0.0

        profile = np.stack([degree, minimum, maximum, mean, std], axis=-1)
        return _add_node_features(data, mx.array(profile.astype(np.float32)), True)
