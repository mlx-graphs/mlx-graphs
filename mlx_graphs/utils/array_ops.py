from typing import Optional

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


def one_hot(labels: mx.array, num_classes: Optional[int] = None) -> mx.array:
    """
    Creates one-hot encoded vectors for all elements provided in `labels`.

    Given an array of labels [num_elements,], returns an array with shape
    [num_elements, num_classes]where each column is an all-zero vector with
    a one at the index of the label.

    Args:
        labels: Array with the labels to transform to one-hot encoded vectors.
        num_classes: Number of labels for the one-hot encoding. By default,
            ``num_classes`` is set to `max_label + 1`.

    Returns:
        An array of shape [num_elements, num_classes] with one-hot encoded vectors.
    """
    if num_classes is None:
        num_classes = (labels.max() + 1).item()

    shape = (labels.shape[0], num_classes)
    one_hot = mx.zeros(shape)

    one_hot[mx.arange(shape[0]), labels.squeeze()] = 1
    return one_hot


def pairwise_distances(x: mx.array, y: mx.array) -> mx.array:
    """
    Compute pairwise distances between points in vectors x and y.

    Args:
        x: array of shape (N, D)
        y: array of shape (M, D)

    Returns:
        Array of shape (N, M)
    """
    assert x.shape[1] == y.shape[1], "Input vectors must have the same dimensionality"
    # Use broadcasting to compute pairwise differences
    expanded_x = mx.expand_dims(x, 1)
    distances = mx.linalg.norm(expanded_x - y, axis=-1)
    return mx.array(distances.tolist())


def index_to_mask(index: mx.array, size: Optional[int] = None) -> mx.array:
    """Converts indices to a mask representation.

    Args:
        index: Array of indices where the mask will be True.
        size: Length of the returned mask array. By default,
            the size is set to the maximum index in the indices + 1.

    Returns:
        A boolean array of length ``size``, with `True` at the given
        ``index`` indices and `False` elsewhere.

    Example:

    .. code-block:: python

        >>> index = mx.array([1, 2, 3])
        >>> index_to_mask(index, size=5)
        array([False, True, True, True, False], dtype=bool)
    """
    index = index.reshape(-1)
    size = index.max().item() + 1 if size is None else size
    mask = mx.zeros(size, dtype=mx.bool_)
    mask[index] = True
    return mask
