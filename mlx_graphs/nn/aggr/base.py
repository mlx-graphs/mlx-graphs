from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.utils import scatter
from mlx_graphs.utils.scatter import ScatterAggregations


class Aggregation(nn.Module):
    r"""Base class of the aggregation operators, mirroring
    ``torch_geometric.nn.aggr.Aggregation``.

    An aggregation reduces a set of elements (e.g. the messages received by
    each node, or the nodes of each graph) to a single vector per set. The sets
    are given by an ``index`` array mapping every element of ``x`` to its set,
    and the reduction is always done along the first axis of ``x``:

    .. code-block:: python

        x = mx.random.normal((10, 64))
        index = mx.array([0, 0, 1, 0, 2, 0, 2, 1, 0, 2])

        out = aggr(x, index)  # shape [3, 64]

    ``index`` does not need to be sorted. Unlike PyG, compressed ``ptr``
    inputs and a ``dim`` argument are not supported: elements are always
    stacked along axis 0, which is the mlx-graphs convention.

    Aggregations can be used on their own (e.g. as a global readout with
    ``index = batch_indices``) or passed as ``aggr`` to any
    :class:`~mlx_graphs.nn.MessagePassing` layer.

    Subclasses implement :meth:`aggregate`, which receives a validated
    ``index`` and ``dim_size``.
    """

    def __call__(
        self,
        x: mx.array,
        index: Optional[mx.array] = None,
        dim_size: Optional[int] = None,
        **kwargs: Any,
    ) -> mx.array:
        """Aggregates the elements of ``x`` according to ``index``.

        Args:
            x: Array of shape ``[num_elements, ...]`` with the elements to
                aggregate (e.g. messages or node features).
            index: Array of shape ``[num_elements]`` with the set of each
                element (e.g. the destination node of each message, or
                ``batch_indices``). If ``None``, all the elements form a
                single set.
            dim_size: Number of sets, i.e. size of the first axis of the
                output. Defaults to ``index.max() + 1`` (``1`` without
                ``index``).
            **kwargs: Additional arguments of the aggregation.

        Returns:
            An array of shape ``[dim_size, ...]``.
        """
        if x.ndim == 0:
            raise ValueError("Aggregations need at least a one-dimensional input")
        if index is None:
            index = mx.zeros((x.shape[0],), dtype=mx.int32)
            dim_size = 1 if dim_size is None else dim_size
        elif dim_size is None:
            dim_size = int(index.max().item()) + 1 if index.size > 0 else 0
        return self.aggregate(x, index, dim_size, **kwargs)

    def aggregate(
        self, x: mx.array, index: mx.array, dim_size: int, **kwargs: Any
    ) -> mx.array:
        """Computes the aggregation. Implemented by subclasses.

        Args:
            x: Array of shape ``[num_elements, ...]``.
            index: Array of shape ``[num_elements]`` with the set of each
                element.
            dim_size: Number of sets.
        """
        raise NotImplementedError

    def reduce(
        self,
        x: mx.array,
        index: mx.array,
        dim_size: int,
        reduce: ScatterAggregations = "add",
    ) -> mx.array:
        """Reduces ``x`` with :func:`~mlx_graphs.utils.scatter`.

        Args:
            x: Array of shape ``[num_elements, ...]``.
            index: Array of shape ``[num_elements]``.
            dim_size: Number of sets.
            reduce: Any reduction supported by
                :func:`~mlx_graphs.utils.scatter`. Default: ``"add"``
        """
        return scatter(x, index, dim_size, aggr=reduce)
