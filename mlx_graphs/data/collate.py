import mlx.core as mx
import numpy as np

from mlx_graphs.data.data import GraphData
from mlx_graphs.data.utils import validate_list_of_graph_data


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
        well as the "private" attributes used behind the hood by the batching.
        These private attributes start with an underscore "_" and can be ignore by
        the user.
    """
    batch_attr_dict = {}

    # Pre-compute __inc__ and __cat_dim__ for all attributes outside the loop
    attrs_inc_cat_dim = [
        (attr, graph_list[0].__inc__(key=attr), graph_list[0].__cat_dim__(key=attr))
        for attr in graph_list[0].to_dict()
    ]

    # To store pre-computed cumsum for attributes where __inc__ is True
    cumsum_dict = {}

    for attr, __inc__, __cat_dim__ in attrs_inc_cat_dim:
        attr_list, attr_sizes = [], []

        # Compute cumsum outside the inner loop if __inc__ is True
        if __inc__ and attr not in cumsum_dict:
            num_attr_list = [
                getattr(graph, "__inc__")(attr) for graph in graph_list
            ]  # Assuming __num_nodes__ provides the needed value
            cumsum_dict[attr] = mx.cumsum(mx.array([0] + num_attr_list))

        cumsum = cumsum_dict.get(attr)

        for i, graph in enumerate(graph_list):
            attr_array = getattr(graph, attr)

            if __inc__:
                attr_array = attr_array + cumsum[i]  # type: ignore

            attr_list.append(attr_array)
            attr_sizes.append(attr_array.shape[__cat_dim__])

        # Concatenate all attributes at once outside the loop
        batch_attr_dict[attr] = mx.concatenate(attr_list, axis=__cat_dim__)
        batch_attr_dict[f"_size_{attr}"] = mx.array(attr_sizes)
        batch_attr_dict[f"_cat_dim_{attr}"] = __cat_dim__
        if __inc__:
            batch_attr_dict[f"_inc_{attr}"] = True
            batch_attr_dict[f"_cumsum_{attr}"] = cumsum

        # Special handling for "edge_index" to be optimized with vectorization
        if attr == "edge_index":
            cumsum = np.array(cumsum)
            batch_indices = np.hstack(
                [
                    np.full((cumsum[i + 1] - cumsum[i]).item(), i)
                    for i in range(len(cumsum) - 1)
                ]
            )
            batch_attr_dict["_batch_indices"] = mx.array(batch_indices)

    return batch_attr_dict
