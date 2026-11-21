from typing import Optional, Union

import mlx.core as mx

from mlx_graphs.data import GraphData
from mlx_graphs.transforms.base_transform import BaseTransform
from mlx_graphs.transforms.structure import _attribute_names, _num_nodes
from mlx_graphs.utils.loop import FillValue, add_remaining_self_loops
from mlx_graphs.utils.transformations import coalesce


class AddRemainingSelfLoops(BaseTransform):
    """Adds a self-loop to every node of a graph that does not already have
    one (see :func:`~mlx_graphs.utils.add_remaining_self_loops`).

    The self-loops are placed after the other edges, in node order. The
    existing self-loops keep their attributes. The attribute ``attr`` of the
    new self-loops is computed from ``fill_value``; the other per-edge
    attributes (e.g. ``edge_features``, ``edge_labels``) also follow the
    edges, with ``fill_value`` for the new self-loops.

    Args:
        attr: Name of the per-edge attribute (e.g. edge weights) filled with
            ``fill_value``. Default: ``"edge_weights"``
        fill_value: Value of the attributes of the added self-loops: a number,
            an array, or a reduction (``"add"``, ``"mean"``, ``"max"``,
            ``"min"``) of the attributes of the incoming edges of each node.
            Default: ``1.0``

    Example:

    .. code-block:: python

        graph = GraphData(
            edge_index=mx.array([[0, 1, 1], [1, 0, 1]]),
            edge_weights=mx.array([0.5, 0.5, 2.0]),
            num_nodes=3,
        )
        graph = AddRemainingSelfLoops()(graph)
        graph.edge_index
        # mx.array([[0, 1, 0, 1, 2], [1, 0, 0, 1, 2]])
        graph.edge_weights
        # mx.array([0.5, 0.5, 1.0, 2.0, 1.0])
    """

    def __init__(
        self,
        attr: str = "edge_weights",
        fill_value: FillValue = 1.0,
    ):
        self.attr = attr
        self.fill_value = fill_value

    def process(self, data: GraphData) -> GraphData:
        """Adds the missing self-loops to ``data``.

        Args:
            data: The graph to transform.

        Returns:
            The graph with a self-loop on every node.
        """
        num_nodes = _num_nodes(data)
        _, edge_keys = _attribute_names(data, num_nodes)
        if getattr(data, self.attr, None) is not None and self.attr not in edge_keys:
            edge_keys.append(self.attr)
        edge_index = data.edge_index
        for key in edge_keys:
            _, values = add_remaining_self_loops(
                edge_index, getattr(data, key), self.fill_value, num_nodes
            )
            setattr(data, key, values)
        data.edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=num_nodes)
        data.num_nodes = num_nodes
        return data

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(attr='{self.attr}', "
            f"fill_value={self.fill_value})"
        )


class RemoveDuplicatedEdges(BaseTransform):
    """Removes the duplicated edges of a graph and sorts the edges by source
    then destination (see :func:`~mlx_graphs.utils.coalesce`).

    The attributes of merged edges are combined with ``reduce``.

    Args:
        key: Name(s) of the per-edge attributes to merge with ``reduce``.
            Default: ``None`` (all the per-edge attributes, e.g.
            ``edge_features``, ``edge_weights`` and ``edge_labels``, so that
            every per-edge attribute stays aligned with the edges).
        reduce: How the attributes of merged edges are combined: ``"add"``,
            ``"mean"``, ``"max"`` or ``"min"``. Default: ``"add"``

    Example:

    .. code-block:: python

        graph = GraphData(
            edge_index=mx.array([[0, 1, 0], [1, 2, 1]]),
            edge_weights=mx.array([1.0, 2.0, 3.0]),
        )
        graph = RemoveDuplicatedEdges()(graph)
        graph.edge_index
        # mx.array([[0, 1], [1, 2]])
        graph.edge_weights
        # mx.array([4.0, 2.0])
    """

    def __init__(
        self, key: Optional[Union[str, list[str]]] = None, reduce: str = "add"
    ):
        self.keys = [key] if isinstance(key, str) else key
        self.reduce = reduce

    def process(self, data: GraphData) -> GraphData:
        """Merges the duplicated edges of ``data``.

        Args:
            data: The graph to transform.

        Returns:
            The graph without duplicated edges.
        """
        num_nodes = _num_nodes(data)
        if self.keys is None:
            _, keys = _attribute_names(data, num_nodes)
        else:
            keys = [key for key in self.keys if getattr(data, key, None) is not None]
        edge_index = data.edge_index
        for key in keys:
            values = getattr(data, key)
            if not isinstance(values, mx.array):
                continue
            _, values = coalesce(edge_index, values, num_nodes, self.reduce)
            setattr(data, key, values)
        data.edge_index = coalesce(edge_index, num_nodes=num_nodes)
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(keys={self.keys}, reduce='{self.reduce}')"
