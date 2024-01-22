from typing import Tuple, Union

import mlx.core as mx


def get_src_dst_features(
    edge_index: mx.array,
    node_features: Union[mx.array, Tuple[mx.array, mx.array]],
) -> Tuple[mx.array, mx.array]:
    """
    Extracts source and destination node features based on the given edge indices.

    Args:
        edge_index (mx.array): An array of shape (2, number_of_edges), where each columns contains the source
                and destination nodes of an edge.
        node_features (mx.array): The input array of node features.

    Returns:
        Tuple[mx.array, mx.array]: A tuple containing source and destination features.
    """
    src_idx, dst_idx = edge_index

    if isinstance(node_features, tuple):
        src_val, dst_val = node_features
        src_val = src_val[src_idx]
        dst_val = dst_val[dst_idx]

    elif isinstance(node_features, mx.array):
        src_val = node_features[src_idx]
        dst_val = node_features[dst_idx]

    else:
        raise ValueError(
            "Invalid type for argument `array`, should be a `mx.array` or a `tuple`."
        )

    return src_val, dst_val


def broadcast(src: mx.array, other: mx.array, dim: int) -> mx.array:
    """
    Make the shape broadcastable between arrays src and other.
    May be required in some situations like index broadcasting.

    Args:
        src (mx.array): source array to broadcast.
        other (mx.array): other array to match the shape.

    Returns:
        mx.array: array with new broadcastable shape.
    """
    if dim < 0:
        dim = other.ndim + dim
    if src.ndim == 1:
        for _ in range(0, dim):
            src = mx.expand_dims(src, 0)
    for _ in range(src.ndim, other.ndim):
        src = mx.expand_dims(src, -1)
    src = expand(src, other.shape)
    return src


def expand(array: mx.array, new_shape: tuple) -> mx.array:
    """
    Expand the dimensions of an array to a new shape.

    Args:
        array (mx.array): The array to expand.
        new_shape (tuple): The new shape desired. The new dimensions must be compatible
                       with the original shape of the array.

    Returns:
        mx.array: A view of the array with the new shape.
    """
    orig_shape = array.shape

    if not all(new_dim >= orig_dim for new_dim, orig_dim in zip(new_shape, orig_shape)):
        raise ValueError("New shape must be greater than or equal to original shape")

    broadcast_shape = tuple(
        max(orig_dim, new_dim) for orig_dim, new_dim in zip(orig_shape, new_shape)
    )
    return mx.broadcast_to(array, broadcast_shape)
