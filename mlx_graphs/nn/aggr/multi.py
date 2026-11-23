import copy
from typing import Any, Dict, List, Literal, Optional, Union, get_args

import mlx.core as mx
from mlx.nn import Linear  # PyTorch's default initialization, as in PyG

from mlx_graphs.nn.aggr.base import Aggregation
from mlx_graphs.nn.aggr.resolver import aggregation_resolver

CombineModes = Literal[
    "cat", "proj", "sum", "mean", "max", "min", "logsumexp", "std", "var"
]


class MultiAggregation(Aggregation):
    r"""Applies several aggregations and combines their results, as in
    `"Principal Neighbourhood Aggregation for Graph Nets"
    <https://arxiv.org/abs/2004.05718>`_ and `"Adaptive Filters and Aggregator
    Fusion for Efficient Graph Convolutions"
    <https://arxiv.org/abs/2104.01481>`_ papers.

    Args:
        aggrs: List of aggregations: :class:`Aggregation` instances or names
            (see :func:`aggregation_resolver`).
        aggrs_kwargs: Optional list (one dict per aggregation) of arguments
            used to build the aggregations given by name. Default: ``None``
        mode: How the results are combined:

            * ``"cat"`` (default): concatenation along the last axis
            * ``"proj"``: concatenation followed by a linear layer
            * ``"sum"``, ``"mean"``, ``"max"``, ``"min"``, ``"logsumexp"``,
              ``"std"``, ``"var"``: element-wise reduction over the
              aggregations (``std``/``var`` are unbiased, as ``torch.std``)
        mode_kwargs: Arguments of the combine mode. ``"proj"`` requires
            ``node_features_dim`` (input size of each aggregation, an int or a
            list) and ``out_features_dim`` (output size); the PyG names
            ``in_channels``/``out_channels`` are accepted too. Other entries
            are passed to the linear layer (e.g. ``bias``). Default: ``None``

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn.aggr import MultiAggregation

        aggr = MultiAggregation(["mean", "max", "std"])
        x = mx.random.normal((6, 16))
        index = mx.array([0, 0, 1, 1, 1, 2])
        >>> aggr(x, index).shape
        (3, 48)
    """

    def __init__(
        self,
        aggrs: List[Union[Aggregation, str]],
        aggrs_kwargs: Optional[List[Dict[str, Any]]] = None,
        mode: CombineModes = "cat",
        mode_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        if not isinstance(aggrs, (list, tuple)):
            raise ValueError(
                f"'aggrs' of MultiAggregation should be a list or tuple "
                f"(got '{type(aggrs).__name__}')"
            )
        if len(aggrs) == 0:
            raise ValueError("'aggrs' of MultiAggregation should not be empty")
        if aggrs_kwargs is None:
            aggrs_kwargs = [{}] * len(aggrs)
        elif len(aggrs) != len(aggrs_kwargs):
            raise ValueError(
                f"'aggrs_kwargs' with invalid length passed to MultiAggregation "
                f"(got {len(aggrs_kwargs)}, expected {len(aggrs)})"
            )
        if mode not in get_args(CombineModes):
            raise ValueError(f"Combine mode '{mode}' is not supported")

        self.aggrs = [
            aggregation_resolver(aggr, **(kwargs or {}))
            for aggr, kwargs in zip(aggrs, aggrs_kwargs)
        ]
        self.mode = mode

        mode_kwargs = copy.copy(mode_kwargs) or {}
        in_dim = mode_kwargs.pop("node_features_dim", None)
        in_dim = mode_kwargs.pop("in_channels", in_dim)
        out_dim = mode_kwargs.pop("out_features_dim", None)
        out_dim = mode_kwargs.pop("out_channels", out_dim)
        self.out_features_dim = out_dim

        if mode == "proj":
            if len(aggrs) == 1:
                raise ValueError("Multiple aggregations are required for 'proj' mode")
            if in_dim is None or out_dim is None:
                raise ValueError(
                    "Combine mode 'proj' needs `node_features_dim` and "
                    "`out_features_dim` in `mode_kwargs`"
                )
            if isinstance(in_dim, int):
                in_dim = [in_dim] * len(aggrs)
            self.lin = Linear(sum(in_dim), out_dim, **mode_kwargs)

    def get_out_features_dim(self, node_features_dim: int) -> int:
        """Output size for inputs with ``node_features_dim`` features."""
        if self.out_features_dim is not None:
            return self.out_features_dim
        if self.mode == "cat":
            return node_features_dim * len(self.aggrs)
        return node_features_dim

    def aggregate(self, x, index, dim_size, **kwargs):
        outs = [aggr(x, index, dim_size) for aggr in self.aggrs]
        return self.combine(outs)

    def combine(self, inputs: List[mx.array]) -> mx.array:
        """Combines the outputs of the aggregations according to ``mode``."""
        if len(inputs) == 1:
            return inputs[0]
        if self.mode == "cat":
            return mx.concatenate(inputs, axis=-1)
        if self.mode == "proj":
            return self.lin(mx.concatenate(inputs, axis=-1))

        stacked = mx.stack(inputs, axis=0)
        if self.mode == "sum":
            return stacked.sum(axis=0)
        if self.mode == "mean":
            return stacked.mean(axis=0)
        if self.mode == "max":
            return stacked.max(axis=0)
        if self.mode == "min":
            return stacked.min(axis=0)
        if self.mode == "logsumexp":
            return mx.logsumexp(stacked, axis=0)
        var = mx.var(stacked, axis=0, ddof=1)
        if self.mode == "std":
            # sqrt has an infinite gradient at 0 (e.g. a set with a single
            # element, where mean == max == sum): give it a zero gradient
            # there, as torch.std does.
            positive = var > 0
            return mx.where(positive, mx.sqrt(mx.where(positive, var, 1.0)), 0.0)
        return var

    def _extra_repr(self) -> str:
        return f"mode={self.mode}"
