import mlx.core as mx

from mlx_graphs.data.data import GraphData
from mlx_graphs.utils.validators import validate_list_of_graph_data


@validate_list_of_graph_data
def collate(graph_list: list[GraphData]) -> dict:
    """Concatenates attributes of multiple graphs based on the specifications
    of each `GraphData`.

    By default, concatenates all default array attributes in dim 0
    apart from `edge_index` which is concatenated along dim 1.
    Each graph remains independent in the final graph by incrementing
    the indices in `edge_index` based on the cumsum of previous number
    of nodes per graph.

    Args:
        graph_list: List of `GraphData` objects to collate

    Returns:
        Dict containing all the attributes of the unified and disconnected big graph as
            well as the "private" attributes used behind the hood by the batching. These private
            attributes start with an underscore "_" and can be ignore by the user.
    """
    batch_attr_dict = {}

    for attr in graph_list[0].to_dict():
        __inc__ = graph_list[0].__inc__(key=attr)
        __cat_dim__ = graph_list[0].__cat_dim__(key=attr)

        cumsum: mx.array = None
        if __inc__:
            num_attr_list = [getattr(graph, "__inc__")(attr) for graph in graph_list]
            cumsum = mx.cumsum(mx.array([0] + num_attr_list))

        attr_list, attr_sizes = [], []
        for i, graph in enumerate(graph_list):
            attr_array = getattr(graph, attr)

            if __inc__:
                attr_array = attr_array + cumsum[i]

            attr_list.append(attr_array)
            attr_sizes.append(attr_array.shape[__cat_dim__])

        # Private attributes are used later in batching for indexing
        batch_attr_dict[f"_size_{attr}"] = mx.array(attr_sizes)
        batch_attr_dict[f"_cat_dim_{attr}"] = __cat_dim__
        if __inc__:
            batch_attr_dict[f"_inc_{attr}"] = True
            batch_attr_dict[f"_cumsum_{attr}"] = cumsum

        if attr == "edge_index":
            batch_indices = []
            for i in range(len(cumsum) - 1):
                batch_indices.extend([i] * (cumsum[i + 1] - cumsum[i]).item())
            batch_attr_dict["_batch_indices"] = mx.array(batch_indices)

        batch_attr_dict[attr] = mx.concatenate(attr_list, axis=__cat_dim__)

    return batch_attr_dict
