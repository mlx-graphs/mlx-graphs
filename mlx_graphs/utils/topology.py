from typing import Optional
import mlx.core as mx
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
        edge_index (mlx.core.array): The edge index of the graph of shape [num_edges, 2]
        edge_features (mlx.core.array, optional): Edge features associated
            with each edge. If provided, the function considers both edge indices
            and features for the check.

    Returns:
        bool: True if the graph is undirected, False otherwise.
    """
    # The function checks if the sorted order of source-destination pairs is equal
    # to the sorted order of destination-source pairs. If edge features are provided,
    # it also checks for equality in their order.
    if edge_features is None:
        src_dst_sort, _ = sort_edge_index(edge_index)
        dst_src_sort, _ = sort_edge_index(
            mx.stack([edge_index[:, 1], edge_index[:, 0]], axis=1)
        )
        if mx.array_equal(src_dst_sort, dst_src_sort):
            return True
    else:
        src_dst_sort, src_dst_feat = sort_edge_index_and_features(
            edge_index, edge_features
        )
        dst_src_sort, dst_src_feat = sort_edge_index_and_features(
            mx.stack([edge_index[:, 1], edge_index[:, 0]], axis=1), edge_features
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
        edge_index (mlx.core.array): The edge index of the graph.
        edge_features (mlx.core.array, optional): Edge features associated
            with each edge. If provided, the function considers both edge indices
            and features for the check.

    Returns:
        bool: True if the graph is directed, False otherwise.
    """
    return not is_undirected(edge_index, edge_features)
