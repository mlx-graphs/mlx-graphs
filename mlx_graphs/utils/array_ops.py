import mlx.core as mx


def broadcast(src: mx.array, other: mx.array, dim: int) -> mx.array:
    """
    Make the shape broadcastable between arrays src and other.
    May be required in some situations like index broadcasting.

    Args:
        src: source array to broadcast.
        other: other array to match the shape.

    Returns:
        Array with new broadcastable shape.
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
        array: The array to expand.
        new_shape: The new shape desired. The new dimensions must be compatible
            with the original shape of the array.

    Returns:
        A view of the array with the new shape.
    """
    orig_shape = array.shape

    if not all(new_dim >= orig_dim for new_dim, orig_dim in zip(new_shape, orig_shape)):
        raise ValueError("New shape must be greater than or equal to original shape")

    broadcast_shape = tuple(
        max(orig_dim, new_dim) for orig_dim, new_dim in zip(orig_shape, new_shape)
    )
    return mx.broadcast_to(array, broadcast_shape)
