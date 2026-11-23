import mlx.core as mx
import numpy as np

from mlx_graphs.data.data import GraphData
from mlx_graphs.data.utils import validate_list_of_graph_data

_SCALAR_TYPES = (bool, int, float, np.bool_, np.integer, np.floating)


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
    first_graph = graph_list[0]

    # `_num_nodes` is a scalar node count, not an array to concatenate.
    # It is handled separately at the end.
    attributes = [attr for attr in first_graph.to_dict() if attr != "_num_nodes"]

    # Offsets of incremented attributes (e.g. `edge_index`) only depend on the
    # value returned by `__inc__`, usually the number of nodes, so graphs are
    # only asked once per distinct attribute.
    for attr in attributes:
        arrays = [getattr(graph, attr) for graph in graph_list]

        # Non-array attributes, as in PyG: Python / numpy scalars (e.g. the
        # `batch_size` of NeighborLoader subgraphs) become an array with one
        # entry per graph, anything else (strings, ...) a list. Unbatching gives
        # each graph its original value back.
        # NumPy scalars have a `shape` too, but cannot be concatenated.
        if not all(isinstance(value, (mx.array, np.ndarray)) for value in arrays):
            if all(isinstance(value, _SCALAR_TYPES) for value in arrays):
                batch_attr_dict[attr] = mx.array(np.asarray(arrays))
                batch_attr_dict[f"_scalar_{attr}"] = True
            else:
                batch_attr_dict[attr] = arrays
                batch_attr_dict[f"_list_{attr}"] = True
            continue

        cat_dim = first_graph.__cat_dim__(key=attr)
        increment = first_graph.__inc__(key=attr)
        # 0-dim arrays are stacked into one entry per graph, as in PyG, and
        # unbatched back to 0-dim arrays.
        if all(array.ndim == 0 for array in arrays):
            arrays = [array.reshape(1) for array in arrays]
            cat_dim = 0
            batch_attr_dict[f"_stack_{attr}"] = True
        sizes = [array.shape[cat_dim] for array in arrays]
        batched = mx.concatenate(arrays, axis=cat_dim)

        # `None` means no increment; 0 (e.g. a first graph without nodes)
        # still needs the offsets of the following graphs.
        if increment is not None:
            increments = [graph.__inc__(attr) for graph in graph_list]
            cumsum = np.zeros(len(graph_list) + 1, dtype=np.int64)
            np.cumsum(increments, out=cumsum[1:])

            # One offset per element along the concatenation axis, added in a
            # single operation instead of one addition per graph.
            offsets = np.repeat(cumsum[:-1], sizes)
            shape = [1] * batched.ndim
            shape[cat_dim] = -1
            batched = batched + mx.array(offsets, dtype=batched.dtype).reshape(shape)

            batch_attr_dict[f"_inc_{attr}"] = True
            batch_attr_dict[f"_cumsum_{attr}"] = mx.array(cumsum)

            if attr == "edge_index":
                batch_indices = np.repeat(np.arange(len(graph_list)), increments)
                batch_attr_dict["_batch_indices"] = mx.array(
                    batch_indices.astype(np.int32)
                )

        batch_attr_dict[attr] = batched
        batch_attr_dict[f"_size_{attr}"] = mx.array(sizes)
        batch_attr_dict[f"_cat_dim_{attr}"] = cat_dim

    # Keep explicit node counts so the batch reports the right total and each
    # graph gets its own count back when unbatching.
    if "_num_nodes" in graph_list[0].to_dict():
        num_nodes_list = [graph.num_nodes for graph in graph_list]
        batch_attr_dict["_num_nodes"] = sum(num_nodes_list)
        batch_attr_dict["_num_nodes_per_graph"] = mx.array(num_nodes_list)

    return batch_attr_dict
