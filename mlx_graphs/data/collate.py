from typing import List

import mlx.core as mx

from mlx_graphs.data.data import GraphData
from mlx_graphs.utils.validators import validate_list_of_graph_data


@validate_list_of_graph_data
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
    batch_attr_dict = {}

    for attr in graph_list[0].to_dict():
        __inc__ = graph_list[0].__inc__(key=attr)
        __cat_dim__ = graph_list[0].__cat_dim__(key=attr)

        cumsum: mx.array = None
        if __inc__:
            num_nodes_list = [graph.num_nodes() for graph in graph_list]
            cumsum = mx.cumsum(mx.array([0] + num_nodes_list))

        attr_list, attr_sizes = [], []
        for i, graph in enumerate(graph_list):
            attr_array = getattr(graph, attr)

            if __inc__:
                attr_array = attr_array + cumsum[i]

            attr_list.append(attr_array)
            attr_sizes.append(attr_array.shape[__cat_dim__])

        batch_attr_dict[f"_size_{attr}"] = mx.array(attr_sizes)
        batch_attr_dict[f"_cat_dim_{attr}"] = __cat_dim__
        batch_attr_dict[f"_inc_{attr}"] = __inc__
        if __inc__:
            batch_attr_dict["_cumsum"] = cumsum

        batch_attr_dict[attr] = mx.concatenate(attr_list, axis=__cat_dim__)

    return batch_attr_dict
