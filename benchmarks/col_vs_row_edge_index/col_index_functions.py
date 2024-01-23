import mlx.core as mx
import numpy as np
from typing import Optional


def get_src_dst_features_COL(
    edge_index: mx.array,
    node_features: mx.array,
):
    src_idx, dst_idx = edge_index[:, 0], edge_index[:, 1]

    src_val = node_features[src_idx]
    dst_val = node_features[dst_idx]

    return src_val, dst_val


def sort_edge_index_COL(edge_index: mx.array) -> tuple[mx.array, mx.array]:  # win
    sorted_target_indices = mx.argsort(edge_index[:, 1])
    target_sorted_index = edge_index[sorted_target_indices]
    sorted_source_indices = mx.argsort(target_sorted_index[:, 0])
    sorted_edge_index = target_sorted_index[sorted_source_indices]
    sorting_indices = sorted_target_indices[sorted_source_indices]
    return sorted_edge_index, sorting_indices


def sort_edge_index_and_features_COL(
    edge_index: mx.array, edge_features: mx.array
) -> tuple[mx.array, mx.array]:
    sorted_edge_index, sorting_indices = sort_edge_index_COL(edge_index)
    sorted_edge_features = edge_features[sorting_indices]
    return sorted_edge_index, sorted_edge_features


def is_undirected_COL(
    edge_index: mx.array, edge_features: Optional[mx.array] = None
) -> bool:
    if edge_features is None:
        src_dst_sort, _ = sort_edge_index_COL(edge_index)
        dst_src_sort, _ = sort_edge_index_COL(
            mx.stack([edge_index[:, 1], edge_index[:, 0]], axis=1)
        )
        if mx.array_equal(src_dst_sort, dst_src_sort):
            return True
    else:
        src_dst_sort, src_dst_feat = sort_edge_index_and_features_COL(
            edge_index, edge_features
        )
        dst_src_sort, dst_src_feat = sort_edge_index_and_features_COL(
            mx.stack([edge_index[:, 1], edge_index[:, 0]], axis=1), edge_features
        )
        if mx.array_equal(src_dst_sort, dst_src_sort) and mx.array_equal(
            src_dst_feat, dst_src_feat
        ):
            return True

    return False


def to_edge_index_COL(
    adjacency_matrix: mx.array, *, dtype: mx.Dtype = mx.uint32
) -> mx.array:
    edge_index = mx.stack(
        [mx.array(x, dtype=dtype) for x in np.nonzero(adjacency_matrix)]
    )
    return edge_index


def to_sparse_adjacency_matrix_COL(
    adjacency_matrix: mx.array, *, dtype: mx.Dtype = mx.uint32
) -> tuple[mx.array, mx.array]:
    edge_index = to_edge_index_COL(adjacency_matrix)
    edge_features = adjacency_matrix[edge_index[0], edge_index[1]]
    return edge_index, edge_features


def to_adjacency_matrix_COL(
    edge_index: mx.array,
    edge_features: Optional[mx.array] = None,
    num_nodes: Optional[int] = None,
) -> mx.array:
    num_nodes = (mx.max(edge_index) + 1).item()
    adjacency_matrix = mx.zeros((num_nodes, num_nodes), dtype=edge_index.dtype)
    if edge_features is None:
        adjacency_matrix[edge_index[0], edge_index[1]] = 1
    else:
        adjacency_matrix[edge_index[0], edge_index[1]] = edge_features
    return adjacency_matrix
