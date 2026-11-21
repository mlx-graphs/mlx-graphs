from typing import List, Union, get_args

from mlx_graphs.nn.aggr.base import Aggregation
from mlx_graphs.utils.scatter import QuantileInterpolations, scatter_quantile


class QuantileAggregation(Aggregation):
    r"""Feature-wise :math:`q`-th quantile of the elements of each set.

    For every feature :math:`d`, with :math:`x_{\pi_1,d} \le \dots \le
    x_{\pi_n,d}` the sorted values of a set of size :math:`n`, the quantile is
    read at position :math:`q \cdot (n - 1)`, interpolating between the two
    neighbouring values when this position is not an integer.

    Args:
        q: Quantile, or list of quantiles, in ``[0, 1]``. With several
            quantiles, the results are concatenated along the last axis.
        interpolation: Interpolation between two values ``a <= b``:
            ``"linear"`` (``a + (b - a) * frac``), ``"lower"`` (``a``),
            ``"higher"`` (``b``), ``"nearest"`` or ``"midpoint"``.
            Default: ``"linear"``
        fill_value: Output of empty sets. Default: ``0.0``

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn.aggr import QuantileAggregation

        aggr = QuantileAggregation(q=[0.25, 0.75])
        x = mx.random.normal((6, 16))
        index = mx.array([0, 0, 1, 1, 1, 2])
        >>> aggr(x, index).shape
        (3, 32)
    """

    def __init__(
        self,
        q: Union[float, List[float]],
        interpolation: QuantileInterpolations = "linear",
        fill_value: float = 0.0,
    ):
        super().__init__()
        qs = list(q) if isinstance(q, (list, tuple)) else [q]
        if len(qs) == 0:
            raise ValueError("Provide at least one quantile value for `q`.")
        if not all(0.0 <= quantile <= 1.0 for quantile in qs):
            raise ValueError("`q` must be in the range [0, 1].")
        if interpolation not in get_args(QuantileInterpolations):
            raise ValueError(f"Invalid interpolation method '{interpolation}'")
        self._q = q
        self.interpolation = interpolation
        self.fill_value = fill_value

    def aggregate(self, x, index, dim_size, **kwargs):
        return scatter_quantile(
            x,
            index,
            dim_size,
            q=self._q,
            interpolation=self.interpolation,
            fill_value=self.fill_value,
        )

    def _extra_repr(self) -> str:
        return f"q={self._q}"


class MedianAggregation(QuantileAggregation):
    r"""Feature-wise median of the elements of each set.

    For every feature :math:`d`, with :math:`x_{\pi_1,d} \le \dots \le
    x_{\pi_n,d}` the sorted values of a set of size :math:`n`, it returns
    :math:`x_{\pi_i,d}` with :math:`i = \lfloor (n - 1) / 2 \rfloor + 1`.

    .. note::
        If the median lies between two values, the lowest one is returned.
        Use :class:`QuantileAggregation` for other interpolations.

    Args:
        fill_value: Output of empty sets. Default: ``0.0``
    """

    def __init__(self, fill_value: float = 0.0):
        super().__init__(0.5, "lower", fill_value)

    def _extra_repr(self) -> str:
        return ""
