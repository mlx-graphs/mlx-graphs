from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.utils.sorting import sort_edge_index, sort_edge_index_and_features
from mlx_graphs.utils.validators import validate_edge_index_and_features


@validate_edge_index_and_features
def is_undirected(
    edge_index: mx.array, edge_features: Optional[mx.array] = None
) -> bool:
    """
    Determines whether a graph is undirected based on the given edge index
    and optional edge features.

    Args:
        edge_index: The edge index of the graph.
        edge_features: Edge features associated
            with each edge. If provided, the function considers both edge indices
            and features for the check.

    Returns:
        True if the graph is undirected, False otherwise.
    """
    # The function checks if the sorted order of source-destination pairs is equal
    # to the sorted order of destination-source pairs. If edge features are provided,
    # it also checks for equality in their order.
    if edge_features is None:
        src_dst_sort, _ = sort_edge_index(edge_index)
        dst_src_sort, _ = sort_edge_index(mx.stack([edge_index[1], edge_index[0]]))
        if mx.array_equal(src_dst_sort, dst_src_sort):
            return True
    else:
        src_dst_sort, src_dst_feat = sort_edge_index_and_features(
            edge_index, edge_features
        )
        dst_src_sort, dst_src_feat = sort_edge_index_and_features(
            mx.stack([edge_index[1], edge_index[0]]), edge_features
        )
        if mx.array_equal(src_dst_sort, dst_src_sort) and mx.array_equal(
            src_dst_feat, dst_src_feat
        ):
            return True

    return False


@validate_edge_index_and_features
def is_directed(edge_index: mx.array, edge_features: Optional[mx.array] = None) -> bool:
    """
    Determines whether a graph is directed based on the given edge index
    and optional edge features.

    Args:
        edge_index: The edge index of the graph.
        edge_features: Edge features associated
            with each edge. If provided, the function considers both edge indices
            and features for the check.

    Returns:
        True if the graph is directed, False otherwise.
    """
    return not is_undirected(edge_index, edge_features)


def get_num_hops(model: nn.Module) -> int:
    """
    Returns the number of hops the model is aggregating information
    from. This works only for networks based on `MessagePassing`.

    Args:
        model: The GNN Model.

    Returns:
        number of hops the model is aggregating information

    Example:

    .. code-block:: python

        class GNN(nn.Module):
             def __init__(self):
                 super().__init__()
                 self.conv1 = GCNConv(4, 16)
                 self.conv2 = GCNConv(16, 16)
                 self.lin = nn.linear(16, 2)

             def __call__(self, edge_index: mx.array, node_features: mx.array) -> mx.array:
                 x = nn.relu(self.conv1(node_features, edge_index))
                 x = self.conv2(node_features, edge_index)
                 return self.lin(x)
        get_num_hops(GNN())
        # 2
    """
    from mlx_graphs.nn import MessagePassing

    num_hops = 0
    for module in model.modules():
        if isinstance(module, MessagePassing):
            num_hops += 1
    return num_hops
