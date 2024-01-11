from typing import Tuple

import mlx.core as mx


def gather_src_dst(x: mx.array, edge_index: mx.array) -> Tuple[mx.array, mx.array]:
    src_idx, dst_idx = edge_index
    x_i = x[src_idx]
    x_j = x[dst_idx]

    return x_i, x_j

def max_nodes(indices: mx.array) -> int:
    return indices.max().item() + 1
