from collections import defaultdict

import mlx.core as mx

try:
    import networkx as nx
except ImportError:
    raise ImportError(
        "networkx is required to convert to/from nextworkx graphs",
        "run `pip install networkx`",
    )

from mlx_graphs.data import GraphData


def to_networkx(
    data: GraphData,
    remove_self_loops: bool = False,
) -> nx.DiGraph:
    r"""Converts a :class:`mlx_graphs.data.GraphData` instance to a
    a directed :obj:`networkx.DiGraph` otherwise.

    Args:
        data: Graph data object
        remove_self_loops: If set to :obj:`True`, will not
            include self-loops in the resulting graph. (default: :obj:`False`)

    Returns:
        A networkx graph

    Examples:

    .. code-block:: python

        import mlx.core as mx

        edge_index = mx.array(
            [
                [0, 1, 1, 2, 2, 3],
                [1, 0, 2, 1, 3, 2],
            ]
        )
        node_features = mx.array([[1], [1], [1], [1]])
        data = GraphData(node_features=node_features, edge_index=edge_index)
        G = to_networkx(data)
        print(G.edges)
        >>> OutEdgeView([(0, 0), (0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)])

    """

    G = nx.DiGraph()

    if data.graph_features is not None:
        G.graph["graph_features"] = data.graph_features.tolist()

    if data.graph_labels is not None:
        G.graph["graph_labels"] = data.graph_labels.tolist()

    if data.num_nodes is None:
        return G

    node_attrs = {}
    for i in range(data.num_nodes):
        if data.node_features is not None:
            node_attrs["features"] = data.node_features[i].tolist()
        if data.node_labels is not None:
            node_attrs["label"] = data.node_labels[i].item()
        G.add_node(i, **node_attrs)

    edge_attrs = {}
    for i, (v, w) in enumerate(data.edge_index.T.tolist()):
        if remove_self_loops and v == w:
            continue
        if data.edge_features is not None:
            edge_attrs["features"] = data.edge_features[i].tolist()
        G.add_edge(v, w, **edge_attrs)

    return G


def from_networkx(data: nx.Graph) -> GraphData:
    """Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
    :class:`mlx_graphs.data.GraphData` instance.

    Args:
        data: A networkx graph

    Returns:
        A GraphData object

    Examples:

    .. code-block:: python

        import networkx as nx
        from mlx_graphs.utils.convert import from_networkx

        G = nx.Graph()
        G.add_node(1)
        G.add_node(2)
        G.add_node(3)
        G.add_node(4)
        G.add_edge(1, 2)
        G.add_edge(1, 3)
        G.add_edge(2, 3)
        G.add_edge(3, 4)
        mlx_dataset = from_networkx(G)
        print(mlx_dataset)
        >>> GraphData(edge_index(shape=(2, 4), int32))
        print(mlx_dataset.num_nodes)
        >>> 4

    """
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
