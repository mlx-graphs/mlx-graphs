import mlx.core as mx
import numpy as np

from mlx_graphs.utils.sorting import sort_edge_index


@mx.compile
def random_walk_mlx(
    row_ptr: mx.array, col: mx.array, start: mx.array, walk_length: int, rand_data
):
    num_walks = len(start)
    n_out = mx.zeros((num_walks, walk_length + 1), dtype=col.dtype)
    e_out = mx.zeros((num_walks, walk_length), dtype=col.dtype)

    n_out[:, 0] = start

    for index in range(walk_length):
        n_cur = n_out[:, index]

        row_start = row_ptr[n_cur]
        row_end = row_ptr[n_cur + 1]

        mask = (row_end - row_start) > 0
        num_neighbors = row_end - row_start

        rand_idx = rand_data[:, index] * num_neighbors
        rand_idx = rand_idx.astype(mx.int64)
        e_cur = row_start + rand_idx
        n_cur = col[e_cur]
        n_cur = mx.where(~mask, n_out[:, index], n_cur)
        e_cur = mx.where(~mask, -1, e_cur)

        n_out[:, index + 1] = n_cur
        e_out[:, index] = e_cur

    return n_out, e_out


def mlx_random_walks(edge_index, start_indices, walk_length, compile=False):
    sorted_edge_index = sort_edge_index(edge_index=edge_index)
    row_mlx = sorted_edge_index[0][0]
    col_mlx = sorted_edge_index[0][1]
    _, counts_mlx = np.unique(np.array(row_mlx, copy=False), return_counts=True)
    cum_sum_mlx = counts_mlx.cumsum()
    row_ptr_mlx = mx.concatenate([mx.array([0]), mx.array(cum_sum_mlx)])
    rand_data = mx.random.uniform(shape=[start_indices.shape[0], walk_length])
    return random_walk_mlx(
        row_ptr_mlx,
        col_mlx,
        start=mx.array(start_indices.numpy()),
        walk_length=walk_length,
        rand_data=rand_data,
    )
