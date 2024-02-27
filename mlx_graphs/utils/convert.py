from typing import Any

from mlx_graphs.data import GraphData


def to_networkx(
    data: GraphData,
    remove_self_loops: bool = False,
) -> Any:
    r"""Converts a :class:`mlx_graphs.data.GraphData` instance to a
    a directed :obj:`networkx.DiGraph` otherwise.

    Args:
        data (mlx_graphs.data.GraphData or torch_geometric.data.HeteroData):
        A graph data object.
        remove_self_loops (bool, optional): If set to :obj:`True`, will not
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
        >>> to_networkx(data)
        <networkx.classes.digraph.DiGraph at 0x2713fdb40d0>

    """
    import networkx as nx

    G = nx.DiGraph()

    G.add_nodes_from(range(data.num_nodes))

    for v, w in data.edge_index.T.tolist():
        if remove_self_loops and v == w:
            continue

        G.add_edge(v, w)

    return G
