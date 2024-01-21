import mlx.core as mx
from mlx_graphs.utils.validators import (
    validate_edge_index,
    validate_edge_index_and_features,
)


@validate_edge_index
def sort_edge_index(edge_index: mx.array) -> tuple[mx.array, mx.array]:
    """Sort the edge index.

    Args:
        edge_index (mlx.core.array): A [num_edges, 2] array representing edge indices,
            where each row contains the source and destination index of an edge.

    Returns:
        tuple[mlx.core.array, mlx.core.array]: A tuple containing the sorted edge index and the
            corresponding sorting indices.
    """
    sorted_target_indices = mx.argsort(edge_index[:, 1])
    target_sorted_index = edge_index[sorted_target_indices]
    sorted_source_indices = mx.argsort(target_sorted_index[:, 0])
    sorted_edge_index = target_sorted_index[sorted_source_indices]
    sorting_indices = sorted_target_indices[sorted_source_indices]
    return sorted_edge_index, sorting_indices


@validate_edge_index_and_features
def sort_edge_index_and_features(
    edge_index: mx.array, edge_features: mx.array
) -> tuple[mx.array, mx.array]:
    """Sorts the given edge_index and their corresponding features.

    Args:
        edge_index (mlx.core.array): A [num_edges, 2] array representing edge indices,
            where each row contains the source and destination index of an edge.
        edge_features (mlx.core.array): An array representing edge features, where each row
            corresponds to an edge.

    Returns:
        tuple[mlx.core.array, mlx.core.array]: A tuple containing the sorted edge index and the
            corresponding sorted edge features.
    """
    sorted_edge_index, sorting_indices = sort_edge_index(edge_index)
    sorted_edge_features = edge_features[sorting_indices]
    return sorted_edge_index, sorted_edge_features
