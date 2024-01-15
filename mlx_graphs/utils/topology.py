from typing import Optional
import mlx.core as mx


def sort_edge_index(edge_index: mx.array) -> tuple[mx.array, mx.array]:
    sorted_target_indices = mx.argsort(edge_index[1])
    target_sorted_index = edge_index[:, sorted_target_indices]
    sorted_source_indices = mx.argsort(target_sorted_index[0])
    sorted_edge_index = target_sorted_index[:, sorted_source_indices]
    sorting_indices = sorted_target_indices[sorted_source_indices]
    return sorted_edge_index, sorting_indices


def sort_edge_index_and_features(edge_index: mx.array, edge_features: mx.array):
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
