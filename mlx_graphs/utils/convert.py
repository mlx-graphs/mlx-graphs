from collections import defaultdict
from typing import Any

import mlx.core as mx

from mlx_graphs.data import GraphData


def to_networkx(
    data: GraphData,
    remove_self_loops: bool = False,
) -> Any:
    r"""Converts a :class:`mlx_graphs.data.GraphData` instance to a
    a directed :obj:`networkx.DiGraph` otherwise.

    Args:
        data: Graph data object
        remove_self_loops: If set to :obj:`True`, will not
            include self-loops in the resulting graph. (default: :obj:`False`)

    Examples:
        >>> import mlx.core as mx
        >>> edge_index = mx.array(
        ...     [
        ...         [0, 1, 1, 2, 2, 3],
        ...         [1, 0, 2, 1, 3, 2],
        ...     ]
        ... )
        >>> node_features = mx.array([[1], [1], [1], [1]])
        >>> data = GraphData(node_features=node_features, edge_index=edge_index)
        >>> G = to_networkx(data)
        >>> G.edges
        OutEdgeView([(0, 0), (0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)])

    """
    import networkx as nx

    G = nx.DiGraph()

    if data.graph_features:
        G.graph["graph_features"] = data.graph_features.tolist()

    if data.graph_labels:
        G.graph["graph_labels"] = data.graph_labels.tolist()

    if data.num_nodes is None:
        return G

    # G.add_nodes_from(range(data.num_nodes))

    node_features_present = data.node_features is not None
    node_labels_present = data.node_labels is not None

    node_attrs = {}
    for i in range(data.num_nodes):
        if node_features_present:
            node_attrs["features"] = data.node_features[i].tolist()
        if node_labels_present:
            node_attrs["label"] = data.node_labels[i].item()
        G.add_node(i, **node_attrs)

    edge_features_present = data.edge_features is not None
    edge_attrs = {}
    for i, (v, w) in enumerate(data.edge_index.T.tolist()):
        if remove_self_loops and v == w:
            continue
        if edge_features_present:
            edge_attrs["features"] = data.edge_features[i].tolist()
        G.add_edge(v, w, **edge_attrs)

    return G


def from_networkx(data: Any) -> GraphData:
    data_dict = defaultdict(list)

    edge_index = mx.array(list(data.edges())).T
    data_dict["edge_index"] = edge_index

    for attr in ["features", "label"]:
        for entity in ["edge", "node", "graph"]:
            key = f"{entity}_{attr}"
            values = []

            if entity == "edge":
                for _, _, feat_dict in data.edges(data=True):
                    if attr in feat_dict:
                        values.append(feat_dict[attr])
            elif entity == "node":
                for _, feat_dict in data.nodes(data=True):
                    if attr in feat_dict:
                        values.append(feat_dict[attr])
            elif entity == "graph" and attr in data.graph:
                values = [data.graph[attr]]

            if values:
                data_dict[key] = mx.array(values)

    return GraphData(**data_dict)
