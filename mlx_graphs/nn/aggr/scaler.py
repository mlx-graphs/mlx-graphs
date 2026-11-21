from typing import Any, Dict, List, Literal, Optional, Sequence, Union, get_args

import mlx.core as mx
import numpy as np

from mlx_graphs.nn.aggr.base import Aggregation
from mlx_graphs.nn.aggr.multi import MultiAggregation
from mlx_graphs.nn.aggr.resolver import aggregation_resolver
from mlx_graphs.utils.scatter import degree

Scalers = Literal[
    "identity", "amplification", "attenuation", "linear", "inverse_linear"
]


class DegreeScalerAggregation(Aggregation):
    r"""Combines one or more aggregations and rescales their output with one or
    more degree-based scalers, from `"Principal Neighbourhood Aggregation for
    Graph Nets" <https://arxiv.org/abs/2004.05718>`_ paper.

    With :math:`d` the number of elements of a set (the in-degree of a node in
    message passing), and :math:`\delta` the average of :math:`\log(d + 1)`
    (or of :math:`d`, for the linear scalers) over the training graphs, the
    scalers multiply the aggregated features by:

    * ``"identity"``: :math:`1`
    * ``"amplification"``: :math:`\log(d + 1) / \delta`
    * ``"attenuation"``: :math:`\delta / \log(\max(d, 1) + 1)`
    * ``"linear"``: :math:`d / \delta_{lin}`
    * ``"inverse_linear"``: :math:`\delta_{lin} / \max(d, 1)`

    The outputs of all the scalers are concatenated along the last axis.

    Args:
        aggr: The aggregation(s): a name, a list of names/aggregations
            (combined with :class:`MultiAggregation`, ``mode="cat"``) or an
            :class:`Aggregation`.
        scaler: A scaler name or a list of scaler names.
        deg: Histogram of the in-degrees of the training nodes:
            ``deg[i]`` is the number of nodes with in-degree ``i`` (see
            :meth:`~mlx_graphs.nn.PNAConv.get_degree_histogram`). An
            ``mx.array``, a numpy array or a list.
        train_norm: If ``True``, the normalization constants :math:`\delta`
            are learnable parameters. Default: ``False``
        aggr_kwargs: Arguments used to build the aggregation(s) given by name
            (a dict, or a list of dicts for a list of aggregations).
            Default: ``None``

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn.aggr import DegreeScalerAggregation

        deg = mx.array([0, 2, 3, 1])  # 2 nodes of degree 1, 3 of degree 2, ...
        aggr = DegreeScalerAggregation(
            ["mean", "max"], ["identity", "amplification"], deg
        )
        x = mx.random.normal((6, 16))
        index = mx.array([0, 0, 1, 1, 1, 2])
        >>> aggr(x, index).shape
        (3, 64)
    """

    def __init__(
        self,
        aggr: Union[str, List[Union[str, Aggregation]], Aggregation],
        scaler: Union[Scalers, List[Scalers]],
        deg: Union[mx.array, np.ndarray, Sequence[int]],
        train_norm: bool = False,
        aggr_kwargs: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    ):
        super().__init__()

        if isinstance(aggr, (str, Aggregation)):
            self.aggr = aggregation_resolver(aggr, **(aggr_kwargs or {}))
        elif isinstance(aggr, (list, tuple)):
            self.aggr = MultiAggregation(list(aggr), aggr_kwargs)
        else:
            raise ValueError(
                "Only strings, lists, tuples and Aggregation instances are valid "
                f"aggregation schemes (got '{type(aggr).__name__}')"
            )

        scalers = [scaler] if isinstance(scaler, str) else list(scaler)
        for name in scalers:
            if name not in get_args(Scalers):
                raise ValueError(
                    f"Unknown scaler '{name}'. Available: {get_args(Scalers)}"
                )
        self.scaler = scalers

        deg = np.asarray(deg, dtype=np.float64).reshape(-1)
        num_nodes = deg.sum()
        if num_nodes <= 0:
            raise ValueError("The degree histogram `deg` must count some nodes")
        bins = np.arange(deg.size)
        self._init_avg_deg_lin = float((bins * deg).sum() / num_nodes)
        self._init_avg_deg_log = float((np.log(bins + 1) * deg).sum() / num_nodes)
        self.train_norm = train_norm
        self.reset_parameters()

    def reset_parameters(self):
        """Resets the normalization constants to their values from ``deg``."""
        if self.train_norm:
            self.avg_deg_lin = mx.array([self._init_avg_deg_lin])
            self.avg_deg_log = mx.array([self._init_avg_deg_log])
        else:
            self.avg_deg_lin = self._init_avg_deg_lin
            self.avg_deg_log = self._init_avg_deg_log

    def aggregate(self, x, index, dim_size, **kwargs):
        out = self.aggr(x, index, dim_size)

        deg = degree(index, dim_size).astype(out.dtype)
        deg = deg.reshape((-1,) + (1,) * (out.ndim - 1))

        outs = []
        for scaler in self.scaler:
            if scaler == "identity":
                outs.append(out)
            elif scaler == "amplification":
                outs.append(out * (mx.log(deg + 1) / self.avg_deg_log))
            elif scaler == "attenuation":
                outs.append(out * (self.avg_deg_log / mx.log(mx.maximum(deg, 1) + 1)))
            elif scaler == "linear":
                outs.append(out * (deg / self.avg_deg_lin))
            else:  # inverse_linear
                outs.append(out * (self.avg_deg_lin / mx.maximum(deg, 1)))

        return mx.concatenate(outs, axis=-1) if len(outs) > 1 else outs[0]

    def _extra_repr(self) -> str:
        return f"scaler={self.scaler}"
