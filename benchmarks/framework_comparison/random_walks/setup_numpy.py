import numpy as np
from torch_geometric.utils import sort_edge_index
from torch_geometric.utils.num_nodes import maybe_num_nodes


def random_walk_numpy_optimized(rowptr, col, start, walk_length, rand_data):
    num_walks = len(start)

    n_out = np.zeros((num_walks, walk_length + 1), dtype=np.int64)
    e_out = np.zeros((num_walks, walk_length), dtype=np.int64)

    n_out[:, 0] = start

    for index in range(walk_length):
        n_cur = n_out[:, index]
        row_start = rowptr[n_cur]
        row_end = rowptr[n_cur + 1]

        mask = (row_end - row_start) > 0
        num_neighbors = row_end - row_start

        rand_idx = (rand_data[:, index] * num_neighbors).astype(np.int64)
        e_cur = row_start + rand_idx
        n_cur = col[e_cur]

        n_cur[~mask] = n_out[~mask, index]
        e_cur[~mask] = -1

        n_out[:, index + 1] = n_cur
        e_out[:, index] = e_cur

    return n_out, e_out


def random_walks_numpy(edge_index, start_indices, walk_length, compile):
    num_nodes = maybe_num_nodes(edge_index=edge_index)
    row, col = sort_edge_index(edge_index=edge_index, num_nodes=num_nodes)
    row_numpy = row.numpy()
    unique_vals, counts = np.unique(row_numpy, return_counts=True)
    row_ptr_numpy = np.cumsum(counts)
    row_ptr_numpy = np.insert(row_ptr_numpy, 0, 0)
    rand_data = (
        np.random.rand(start_indices.shape[0] * walk_length)
        .astype(np.float32)
        .reshape(start_indices.shape[0], walk_length)
    )
    random_walk_numpy_optimized(
        row_ptr_numpy,
        col.numpy(),
        start_indices.numpy(),
        walk_length=walk_length,
        rand_data=rand_data,
    )
