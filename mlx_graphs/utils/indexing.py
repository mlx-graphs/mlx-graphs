from typing import Any, Union

import mlx.core as mx
import numpy as np


def mask_to_index(mask: mx.array) -> mx.array:
    """
    Converts a boolean mask to the indices of its ``True`` entries (the
    inverse of :func:`~mlx_graphs.utils.array_ops.index_to_mask`).

    Args:
        mask: A boolean array of shape ``[size]``.

    Returns:
        The sorted int32 indices where ``mask`` is ``True``.

    Example:

    .. code-block:: python

        mask_to_index(mx.array([False, True, True, False]))
        # mx.array([1, 2])
    """
    return mx.array(np.nonzero(np.asarray(mask).reshape(-1))[0].astype(np.int32))


def cumsum(x: mx.array, axis: int = 0) -> mx.array:
    """
    Cumulative sum of ``x`` along ``axis``, with a leading zero: the output
    has one more element than ``x`` along ``axis`` (e.g. to turn counts into
    pointers/offsets).

    Args:
        x: The input array.
        axis: The axis along which to sum. Default: ``0``

    Returns:
        An array whose size along ``axis`` is ``x.shape[axis] + 1``.

    Example:

    .. code-block:: python

        cumsum(mx.array([2, 4, 1]))
        # mx.array([0, 2, 6, 7])
    """
    axis = axis + x.ndim if axis < 0 else axis
    zero_shape = list(x.shape)
    zero_shape[axis] = 1
    zeros = mx.zeros(zero_shape, dtype=x.dtype)
    return mx.concatenate([zeros, mx.cumsum(x, axis=axis)], axis=axis)


def select(
    src: Union[mx.array, list, tuple],
    index_or_mask: Union[mx.array, list],
    axis: int = 0,
) -> Union[mx.array, list]:
    """
    Selects the entries of ``src`` along ``axis`` given by an index array or
    by a boolean mask.

    Args:
        src: An array, or a list/tuple (selected along its only axis).
        index_or_mask: Indices of the entries to select, or a boolean mask of
            shape ``[src.shape[axis]]``.
        axis: The axis along which to select (must be ``0`` for lists).
            Default: ``0``

    Returns:
        The selected entries, as an array if ``src`` is an array and as a list
        otherwise.

    Example:

    .. code-block:: python

        src = mx.array([[1, 2], [3, 4], [5, 6]])
        select(src, mx.array([True, False, True]))
        # mx.array([[1, 2], [5, 6]])
        select(["a", "b", "c"], mx.array([2, 0]))
        # ["c", "a"]
    """
    index = np.asarray(index_or_mask)
    if index.dtype == np.bool_:
        index = np.nonzero(index)[0]
    index = index.reshape(-1).astype(np.int64)

    if isinstance(src, mx.array):
        if index.size == 0:
            shape = list(src.shape)
            shape[axis] = 0
            return mx.zeros(shape, dtype=src.dtype)
        return mx.take(src, mx.array(index), axis=axis)
    if isinstance(src, (list, tuple)):
        if axis != 0:
            raise ValueError("Cannot select along an axis other than 0 of a list")
        return [src[i] for i in index.tolist()]
    raise ValueError(f"Invalid input type (got '{type(src)}')")


def narrow(
    src: Union[mx.array, list], axis: int, start: int, length: int
) -> Union[mx.array, list, Any]:
    """
    Returns the ``length`` entries of ``src`` along ``axis`` starting at
    ``start``.

    Args:
        src: An array or a list.
        axis: The axis along which to slice (must be ``0`` for lists).
        start: Index of the first entry.
        length: Number of entries.

    Returns:
        The slice of ``src``.

    Example:

    .. code-block:: python

        narrow(mx.array([[1, 2, 3], [4, 5, 6]]), 1, 1, 2)
        # mx.array([[2, 3], [5, 6]])
    """
    if isinstance(src, mx.array):
        axis = axis + src.ndim if axis < 0 else axis
        slices = [slice(None)] * src.ndim
        slices[axis] = slice(start, start + length)
        return src[tuple(slices)]
    if isinstance(src, list):
        if axis != 0:
            raise ValueError("Cannot narrow along an axis other than 0 of a list")
        return src[start : start + length]
    raise ValueError(f"Invalid input type (got '{type(src)}')")
