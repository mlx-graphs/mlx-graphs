from typing import List

import mlx.core as mx
from mlx_graphs.data.data import GraphData


def collate(graph_list: List[GraphData]) -> dict:
    """
    Collate function to perform the unification of graphs in a batch.
    By default, it concatenates all default graph attributes in dim 0
    apart from `edge_index` which is concatenated along dim 1 and
    nodes indices are incremented

    Args:
        graph_list (List[GraphData]): the list of GraphData objects to collate

    Returns:
        dict: dict containing all the attributes of the unified and disconnected big graph as
        well as the slices corresponding to the portion of node for each graph from the provided
        list of graphs.
    """

    global_dict = {}
    num_graphs = len(graph_list)
    num_nodes = [graph.num_nodes() for graph in graph_list]
    cumsum = mx.cumsum(mx.array([0] + num_nodes))
    for attr in graph_list[0].to_dict():
        inc = graph_list[0].__inc__(key=attr)
        cat_dim = graph_list[0].__cat_dim__(key=attr)

        if inc:
            values = [
                getattr(graph, attr) + cumsum[i] for i, graph in enumerate(graph_list)
            ]
        else:
            values = [getattr(graph, attr) for graph in graph_list]

        global_dict[attr] = mx.concatenate(values, axis=cat_dim)

    nested_slices = [[idx] * num_node for idx, num_node in enumerate(num_nodes)]
    flattened_slices = []
    [flattened_slices.extend(inner_list) for inner_list in nested_slices]
    global_dict.update(
        {
            "slices": mx.array(flattened_slices),
            "cumsum": cumsum,
            "num_graphs": mx.array(num_graphs),
        }
    )

    return global_dict
