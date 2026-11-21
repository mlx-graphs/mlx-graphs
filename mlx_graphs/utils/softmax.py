from typing import Optional

import mlx.core as mx
import numpy as np

from mlx_graphs.utils.scatter import scatter_softmax


def softmax(
    src: mx.array,
    index: Optional[mx.array] = None,
    ptr: Optional[mx.array] = None,
    num_nodes: Optional[int] = None,
    axis: int = 0,
) -> mx.array:
    """
    Computes a sparsely evaluated softmax: the entries of ``src`` are grouped
    by ``index`` along ``axis`` and normalized within each group (e.g. the
    attention scores of the incoming edges of each node).

    PyG-named version of :func:`~mlx_graphs.utils.scatter.scatter_softmax`.

    Args:
        src: The values to normalize.
        index: The group of each entry of ``src`` along ``axis``.
        ptr: Alternatively to ``index``, the CSR pointers of the groups, of
            shape ``[num_groups + 1]``, for entries sorted by group.
        num_nodes: Number of groups. Inferred from ``index`` if not given.
        axis: The axis along which to group. Default: ``0``

    Returns:
        An array of the shape of ``src`` whose entries of each group sum to 1.

    Example:

    .. code-block:: python

        src = mx.array([1.0, 1.0, 1.0, 1.0])
        index = mx.array([0, 0, 1, 2])
        softmax(src, index)
        # mx.array([0.5, 0.5, 1.0, 1.0])
    """
    if index is None:
        if ptr is None:
            raise ValueError("`softmax` requires `index` or `ptr` to be specified")
        counts = np.diff(np.asarray(ptr).reshape(-1).astype(np.int64))
        num_nodes = counts.size if num_nodes is None else num_nodes
        index = mx.array(np.repeat(np.arange(counts.size), counts).astype(np.int32))

    axis = axis + src.ndim if axis < 0 else axis
    if src.shape[axis] == 0:
        return src  # no entries to normalize (scatters need a group)

    if num_nodes is None:
        num_nodes = int(mx.max(index).item()) + 1

    if axis == 0:
        return scatter_softmax(src, index, num_nodes)
    moved = mx.moveaxis(src, axis, 0)
    return mx.moveaxis(scatter_softmax(moved, index, num_nodes), 0, axis)
