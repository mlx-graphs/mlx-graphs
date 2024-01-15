from typing import Optional
import mlx.core as mx


def sort_edge_index(edge_index: mx.array) -> tuple[mx.array, mx.array]:
    """
    Sort the edge index.

    Args:
        edge_index (mlx.core.array): A [2, num_edges] array representing edge indices,
            where the first row contains source indices and the second row contains target indices.

    Returns:
        tuple[mlx.core.array, mlx.core.array]: A tuple containing the sorted edge index and the
            corresponding sorting indices.
    """
    sorted_target_indices = mx.argsort(edge_index[1])
    target_sorted_index = edge_index[:, sorted_target_indices]
    sorted_source_indices = mx.argsort(target_sorted_index[0])
    sorted_edge_index = target_sorted_index[:, sorted_source_indices]
    sorting_indices = sorted_target_indices[sorted_source_indices]
    return sorted_edge_index, sorting_indices


def sort_edge_index_and_features(edge_index: mx.array, edge_features: mx.array):
    """
    Sorts the given edge_index and their corresponding features.

    Args:
        edge_index (mlx.core.array): A [2, num_edges] array representing edge indices,
            where the first row contains source indices and the second row contains target indices.
        edge_features (mlx.core.array): An array representing edge features, where each column
            corresponds to an edge.

    Returns:
        tuple[mlx.core.array, mlx.core.array]: A tuple containing the sorted edge index and the
            corresponding sorted edge features.
    """
    sorted_edge_index, sorting_indices = sort_edge_index(edge_index)
    if edge_features.ndim == 1:
        sorted_edge_features = edge_features[sorting_indices]
    else:
        sorted_edge_features = edge_features[:, sorting_indices]
    return sorted_edge_index, sorted_edge_features


def is_undirected(
    edge_index: mx.array, edge_features: Optional[mx.array] = None
) -> bool:
    pass
