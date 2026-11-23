from typing import Literal, Optional

import mlx.core as mx

from mlx_graphs.data import GraphData
from mlx_graphs.transforms.base_transform import BaseTransform
from mlx_graphs.utils.spatial import knn_graph, radius_graph
from mlx_graphs.utils.transformations import to_undirected


def _positions(data: GraphData) -> mx.array:
    pos = getattr(data, "pos", None)
    if pos is None:
        raise ValueError(
            "The graph has no `pos` attribute holding the coordinates of its nodes"
        )
    return pos


def _batch_indices(data: GraphData) -> Optional[mx.array]:
    """Batch vector of a ``GraphDataBatch``, ``None`` for a single graph."""
    try:
        return getattr(data, "batch_indices", None)
    except AttributeError:
        return None


def _set_edges(data: GraphData, edge_index: mx.array, num_nodes: int) -> GraphData:
    """Replaces the edges of ``data``, dropping the previous per-edge features."""
    data.edge_index = edge_index
    data.edge_features = None
    if getattr(data, "edge_weights", None) is not None:
        data.edge_weights = None
    data.num_nodes = num_nodes
    return data


class KNNGraph(BaseTransform):
    """Creates a k-NN graph from the node positions ``pos`` of a graph.

    The previous edges of the graph are replaced, and its ``edge_features`` and
    ``edge_weights`` are dropped. For a ``GraphDataBatch``, only the nodes of
    the same graph are connected.

    Args:
        k: The number of neighbors of each node. Default: ``6``
        loop: If ``True``, the graph contains self-loops. Default: ``False``
        force_undirected: If ``True``, the reverse of every edge is added and
            duplicate edges are merged. Default: ``False``
        flow: Direction of the edges, for use with message passing. With
            ``"source_to_target"``, the edges go from the neighbors to the
            center node. Default: ``"source_to_target"``
        cosine: If ``True``, uses the cosine distance instead of the euclidean
            distance. Default: ``False``

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.data import GraphData
        from mlx_graphs.transforms import KNNGraph

        pos = mx.random.normal((10, 3))
        graph = GraphData(edge_index=mx.zeros((2, 0), dtype=mx.int32), pos=pos)
        graph = KNNGraph(k=3)(graph)
        >>> graph.edge_index.shape
        (2, 30)
    """

    def __init__(
        self,
        k: int = 6,
        loop: bool = False,
        force_undirected: bool = False,
        flow: Literal["source_to_target", "target_to_source"] = "source_to_target",
        cosine: bool = False,
    ):
        self.k = k
        self.loop = loop
        self.force_undirected = force_undirected
        self.flow = flow
        self.cosine = cosine

    def process(self, data: GraphData) -> GraphData:
        """Replaces the edges of ``data`` by its k-NN graph.

        Args:
            data: The graph to transform. It must have a ``pos`` attribute.

        Returns:
            The graph with the k-NN edges.
        """
        pos = _positions(data)
        edge_index = knn_graph(
            pos,
            self.k,
            _batch_indices(data),
            loop=self.loop,
            flow=self.flow,
            cosine=self.cosine,
        )
        if self.force_undirected:
            edge_index = to_undirected(
                edge_index, coalesce=True, num_nodes=pos.shape[0]
            )
        return _set_edges(data, edge_index, pos.shape[0])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(k={self.k})"


class RadiusGraph(BaseTransform):
    """Creates the graph connecting the nodes whose positions ``pos`` are
    within a given distance.

    The previous edges of the graph are replaced, and its ``edge_features`` and
    ``edge_weights`` are dropped. For a ``GraphDataBatch``, only the nodes of
    the same graph are connected.

    Args:
        r: The distance.
        loop: If ``True``, the graph contains self-loops. Default: ``False``
        max_num_neighbors: The maximum number of neighbors of each node; the
            closest ones are kept. ``None`` for no limit. Default: ``32``
        flow: Direction of the edges, for use with message passing. With
            ``"source_to_target"``, the edges go from the neighbors to the
            center node. Default: ``"source_to_target"``

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.data import GraphData
        from mlx_graphs.transforms import RadiusGraph

        pos = mx.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
        graph = GraphData(edge_index=mx.zeros((2, 0), dtype=mx.int32), pos=pos)
        graph = RadiusGraph(r=1.5)(graph)
        >>> graph.edge_index
        array([[1, 0],
               [0, 1]], dtype=int32)
    """

    def __init__(
        self,
        r: float,
        loop: bool = False,
        max_num_neighbors: Optional[int] = 32,
        flow: Literal["source_to_target", "target_to_source"] = "source_to_target",
    ):
        self.r = r
        self.loop = loop
        self.max_num_neighbors = max_num_neighbors
        self.flow = flow

    def process(self, data: GraphData) -> GraphData:
        """Replaces the edges of ``data`` by its radius graph.

        Args:
            data: The graph to transform. It must have a ``pos`` attribute.

        Returns:
            The graph with the radius edges.
        """
        pos = _positions(data)
        edge_index = radius_graph(
            pos,
            self.r,
            _batch_indices(data),
            loop=self.loop,
            max_num_neighbors=self.max_num_neighbors,
            flow=self.flow,
        )
        return _set_edges(data, edge_index, pos.shape[0])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(r={self.r})"
