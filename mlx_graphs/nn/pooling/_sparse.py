"""Sparse-sparse matrix products of the pooling operators (ASAPooling,
GraphUNet, PANConv).

The products of the entries are enumerated on the host with numpy, merged
by a sort on the device, and the values are computed with gathers and a
scatter-add, so they stay differentiable.
"""

from typing import Optional, Tuple

import mlx.core as mx
import numpy as np

from mlx_graphs.utils.transformations import _to_numpy


def spspmm(
    index_a: mx.array,
    value_a: Optional[mx.array],
    index_b: mx.array,
    value_b: Optional[mx.array],
    num_cols: int,
) -> Tuple[mx.array, mx.array]:
    """Product ``C = A @ B`` of two sparse matrices in COO form.

    Duplicate entries of ``A`` or ``B`` are summed. ``C`` is returned
    coalesced: sorted by row, then column, without duplicates (and without
    dropping the entries whose value happens to be 0).

    Args:
        index_a: ``[2, nnz_a]`` (row, column) indices of ``A``.
        value_a: Values of ``A``, of shape ``[nnz_a]`` (``None`` for ones).
        index_b: ``[2, nnz_b]`` (row, column) indices of ``B``.
        value_b: Values of ``B``, of shape ``[nnz_b]`` (``None`` for ones).
        num_cols: Number of columns of ``B``.

    Returns:
        The ``[2, nnz_c]`` indices and ``[nnz_c]`` values of ``C``.
    """
    a = _to_numpy(index_a).astype(np.int64).reshape(2, -1)
    b = _to_numpy(index_b).astype(np.int64).reshape(2, -1)
    dtype = (
        value_a.dtype
        if value_a is not None
        else value_b.dtype
        if value_b is not None
        else mx.float32
    )
    inner = int(max(a[1].max(initial=-1), b[0].max(initial=-1))) + 1

    # For every entry (i, j) of A, all the entries (j, k) of B.
    order_b = np.argsort(b[0], kind="stable")
    counts_b = np.bincount(b[0], minlength=inner)
    starts_b = np.cumsum(counts_b) - counts_b
    reps = counts_b[a[1]]
    total = int(reps.sum())
    if total == 0:
        return mx.zeros((2, 0), dtype=mx.int32), mx.zeros((0,), dtype=dtype)
    a_pos = np.repeat(np.arange(a.shape[1]), reps)
    offsets = np.arange(total) - np.repeat(np.cumsum(reps) - reps, reps)
    b_pos = order_b[np.repeat(starts_b[a[1]], reps) + offsets]

    values = None
    if value_a is not None:
        values = value_a.reshape(-1)[mx.array(a_pos.astype(np.int32))]
    if value_b is not None:
        vb = value_b.reshape(-1)[mx.array(b_pos.astype(np.int32))]
        values = vb if values is None else values * vb
    if values is None:
        values = mx.ones((total,), dtype=dtype)
    return coalesce_sum(a[0][a_pos], b[1][b_pos], values, num_cols)


def coalesce_sum(
    rows: np.ndarray, cols: np.ndarray, values: mx.array, num_cols: int
) -> Tuple[mx.array, mx.array]:
    """Sorts the entries ``(rows, cols, values)`` of a sparse matrix by row
    then column, and sums the duplicates (differentiably in ``values``).

    The keys ``row * num_cols + col`` are sorted on the device, which is much
    faster than on the host for millions of entries.
    """
    if rows.size == 0:
        return mx.zeros((2, 0), dtype=mx.int32), values
    num_rows = int(rows.max()) + 1
    key_dtype = np.int32 if num_rows * num_cols < 2**31 else np.int64
    keys = mx.array((rows * num_cols + cols).astype(key_dtype))
    order = mx.argsort(keys)
    sorted_keys = keys[order]
    is_first = mx.concatenate([mx.array([True]), sorted_keys[1:] != sorted_keys[:-1]])
    segment = mx.cumsum(is_first.astype(mx.int32)) - 1
    unique = np.array(sorted_keys)[np.array(is_first)].astype(np.int64)
    out_index = mx.array(
        np.stack([unique // num_cols, unique % num_cols]).astype(np.int32)
    )
    out_values = (
        mx.zeros((unique.size,), dtype=values.dtype).at[segment].add(values[order])
    )
    return out_index, out_values


def transpose(index: mx.array) -> mx.array:
    """Indices of the transposed matrix."""
    return mx.stack([index[1], index[0]])
