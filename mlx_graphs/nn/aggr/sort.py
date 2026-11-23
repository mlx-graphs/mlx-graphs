from typing import Optional

import mlx.core as mx

from mlx_graphs.nn.aggr.base import Aggregation
from mlx_graphs.utils.dense import to_dense_batch


class SortAggregation(Aggregation):
    r"""Sort pooling from `"An End-to-End Deep Learning Architecture for Graph
    Classification" <https://www.cse.wustl.edu/~muhan/papers/AAAI_2018_DGCNN.pdf>`_
    paper (DGCNN).

    The elements of each set (e.g. the nodes of each graph) are sorted in
    descending order of their last feature, and the first :math:`k` are kept
    and flattened. Sets with fewer than :math:`k` elements are padded with
    zeros. Unlike PyG, ``index`` does not need to be sorted.

    This is also the ``global_sort_pool`` of earlier PyG versions, see
    :func:`global_sort_pool`.

    Args:
        k: Number of elements kept per set.

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn.aggr import SortAggregation

        pool = SortAggregation(k=3)
        node_features = mx.random.normal((7, 16))
        batch_indices = mx.array([0, 0, 0, 0, 1, 1, 1])
        >>> pool(node_features, batch_indices).shape
        (2, 48)
    """

    def __init__(self, k: int):
        super().__init__()
        if k < 1:
            raise ValueError(f"`k` must be positive, got {k}")
        self.k = k

    def aggregate(
        self,
        x: mx.array,
        index: mx.array,
        dim_size: int,
        max_num_elements: Optional[int] = None,
        **kwargs,
    ) -> mx.array:
        """Sort-pools ``x``.

        Args:
            x: Array of shape ``[num_elements, num_features]``.
            index: Set (e.g. graph) of each element.
            dim_size: Number of sets.
            max_num_elements: Only the first ``max_num_elements`` elements of
                each set (in input order) are considered. Default: ``None``

        Returns:
            An array of shape ``[dim_size, k * num_features]``.
        """
        if x.ndim != 2:
            raise ValueError(
                f"SortAggregation requires two-dimensional inputs (got {x.ndim})"
            )
        num_features = x.shape[1]
        if x.shape[0] == 0:
            return mx.zeros((dim_size, self.k * num_features), dtype=x.dtype)

        # [num_sets, max_set_size, num_features], with a mask of the real rows
        dense, mask = to_dense_batch(x, index, max_num_nodes=max_num_elements)
        num_sets, size, _ = dense.shape
        if num_sets < dim_size:
            dense = mx.concatenate(
                [dense, mx.zeros((dim_size - num_sets, size, num_features), x.dtype)]
            )
            mask = mx.concatenate(
                [mask, mx.zeros((dim_size - num_sets, size), mx.bool_)]
            )
        elif num_sets > dim_size:
            dense, mask = dense[:dim_size], mask[:dim_size]

        # Descending order of the last feature, with the padding last.
        key = mx.where(mask, -dense[:, :, -1], float("inf"))
        perm = mx.argsort(mx.stop_gradient(key), axis=1)
        dense = mx.take_along_axis(dense, perm[:, :, None], axis=1)
        mask = mx.take_along_axis(mask, perm, axis=1)
        dense = mx.where(mask[:, :, None], dense, mx.zeros_like(dense))

        if size >= self.k:
            dense = dense[:, : self.k]
        else:
            pad = mx.zeros((dim_size, self.k - size, num_features), dtype=x.dtype)
            dense = mx.concatenate([dense, pad], axis=1)

        return dense.reshape(dim_size, self.k * num_features)

    def _extra_repr(self) -> str:
        return f"k={self.k}"
