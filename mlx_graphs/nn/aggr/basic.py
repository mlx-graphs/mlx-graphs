from typing import Optional

import mlx.core as mx

from mlx_graphs.nn.aggr.base import Aggregation
from mlx_graphs.utils.scatter import scatter_softmax, scatter_std, scatter_var


class SumAggregation(Aggregation):
    r"""Sums the elements of each set.

    .. math::
        \mathrm{sum}(\mathcal{X}) = \sum_{\mathbf{x}_i \in \mathcal{X}}
        \mathbf{x}_i.

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn.aggr import SumAggregation

        x = mx.array([[1.0], [2.0], [4.0]])
        index = mx.array([0, 0, 1])
        >>> SumAggregation()(x, index)
        array([[3], [4]], dtype=float32)
    """

    def aggregate(self, x, index, dim_size, **kwargs):
        return self.reduce(x, index, dim_size, "add")


class MeanAggregation(Aggregation):
    r"""Averages the elements of each set.

    .. math::
        \mathrm{mean}(\mathcal{X}) = \frac{1}{|\mathcal{X}|}
        \sum_{\mathbf{x}_i \in \mathcal{X}} \mathbf{x}_i.
    """

    def aggregate(self, x, index, dim_size, **kwargs):
        return self.reduce(x, index, dim_size, "mean")


class MaxAggregation(Aggregation):
    r"""Feature-wise maximum of the elements of each set (0 for empty sets).

    .. math::
        \mathrm{max}(\mathcal{X}) = \max_{\mathbf{x}_i \in \mathcal{X}}
        \mathbf{x}_i.
    """

    def aggregate(self, x, index, dim_size, **kwargs):
        return self.reduce(x, index, dim_size, "max")


class MinAggregation(Aggregation):
    r"""Feature-wise minimum of the elements of each set (0 for empty sets).

    .. math::
        \mathrm{min}(\mathcal{X}) = \min_{\mathbf{x}_i \in \mathcal{X}}
        \mathbf{x}_i.
    """

    def aggregate(self, x, index, dim_size, **kwargs):
        return self.reduce(x, index, dim_size, "min")


class MulAggregation(Aggregation):
    r"""Feature-wise product of the elements of each set (1 for empty sets).

    .. math::
        \mathrm{mul}(\mathcal{X}) = \prod_{\mathbf{x}_i \in \mathcal{X}}
        \mathbf{x}_i.
    """

    def aggregate(self, x, index, dim_size, **kwargs):
        return self.reduce(x, index, dim_size, "mul")


class VarAggregation(Aggregation):
    r"""Feature-wise (population) variance of the elements of each set.

    .. math::
        \mathrm{var}(\mathcal{X}) = \mathrm{mean}(\{ \mathbf{x}_i^2 :
        \mathbf{x}_i \in \mathcal{X} \}) - \mathrm{mean}(\mathcal{X})^2.

    Args:
        semi_grad: If ``True``, no gradient flows through the computation of
            :math:`\mathrm{mean}(\mathcal{X}^2)` (semi-gradient), which saves
            memory and compute in the backward pass. Default: ``False``
    """

    def __init__(self, semi_grad: bool = False):
        super().__init__()
        self.semi_grad = semi_grad

    def aggregate(self, x, index, dim_size, **kwargs):
        return scatter_var(x, index, dim_size, semi_grad=self.semi_grad)

    def _extra_repr(self) -> str:
        return f"semi_grad={self.semi_grad}"


class StdAggregation(Aggregation):
    r"""Feature-wise (population) standard deviation of the elements of each
    set.

    .. math::
        \mathrm{std}(\mathcal{X}) = \sqrt{\mathrm{var}(\mathcal{X})}.

    As in PyG, the variance is clamped to :math:`10^{-5}` before the square
    root, and standard deviations up to :math:`\sqrt{10^{-5}}` are set to 0.

    Args:
        semi_grad: See :class:`VarAggregation`. Default: ``False``
    """

    def __init__(self, semi_grad: bool = False):
        super().__init__()
        self.semi_grad = semi_grad

    def aggregate(self, x, index, dim_size, **kwargs):
        return scatter_std(x, index, dim_size, semi_grad=self.semi_grad)

    def _extra_repr(self) -> str:
        return f"semi_grad={self.semi_grad}"


class SoftmaxAggregation(Aggregation):
    r"""Softmax aggregation with an (optionally learnable) inverse temperature,
    from `"DeeperGCN: All You Need to Train Deeper GCNs"
    <https://arxiv.org/abs/2006.07739>`_ paper.

    .. math::
        \mathrm{softmax}(\mathcal{X}|t) = \sum_{\mathbf{x}_i\in\mathcal{X}}
        \frac{\exp(t\cdot\mathbf{x}_i)}{\sum_{\mathbf{x}_j\in\mathcal{X}}
        \exp(t\cdot\mathbf{x}_j)}\cdot\mathbf{x}_{i},

    where :math:`t` controls the softness of the softmax: :math:`t \to 0` gives
    the mean and :math:`t \to \infty` the maximum.

    .. note::
        This is an aggregation (one output per set). The ``"softmax"``
        string of :func:`~mlx_graphs.utils.scatter` and of
        :class:`~mlx_graphs.nn.MessagePassing` instead normalizes the values
        within each set (one output per element), e.g. for attention scores.

    Args:
        t: Initial inverse temperature. Default: ``1.0``
        learn: If ``True``, :math:`t` is a learnable parameter.
            Default: ``False``
        semi_grad: If ``True``, no gradient flows through the softmax
            coefficients. Only possible when ``learn=False``.
            Default: ``False``
        channels: Number of learnable temperatures. With more than 1, one
            temperature is learned per feature channel (``x`` must then be of
            shape ``[num_elements, channels]``). Requires ``learn=True``.
            Default: ``1``

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn.aggr import SoftmaxAggregation

        aggr = SoftmaxAggregation(t=1.0, learn=True)
        x = mx.random.normal((6, 16))
        index = mx.array([0, 0, 1, 1, 1, 2])
        >>> aggr(x, index).shape
        (3, 16)
    """

    def __init__(
        self,
        t: float = 1.0,
        learn: bool = False,
        semi_grad: bool = False,
        channels: int = 1,
    ):
        super().__init__()
        if learn and semi_grad:
            raise ValueError(
                "Cannot enable 'semi_grad' in SoftmaxAggregation when the "
                "temperature 't' is learnable"
            )
        if not learn and channels != 1:
            raise ValueError(
                "Cannot set 'channels' greater than 1 in SoftmaxAggregation "
                "when 't' is not learnable"
            )
        self._init_t = t
        self.learn = learn
        self.semi_grad = semi_grad
        self.channels = channels
        if learn:
            self.t = mx.full((channels,), t)
        else:
            self._t = t
        self.reset_parameters()

    def reset_parameters(self):
        """Resets the temperature to its initial value."""
        if self.learn:
            self.t = mx.full((self.channels,), self._init_t)

    def aggregate(self, x, index, dim_size, **kwargs):
        t = self.t if self.learn else self._t
        if self.channels != 1 and x.ndim != 2:
            raise ValueError(
                f"SoftmaxAggregation with channels > 1 requires two-dimensional "
                f"inputs (got {x.ndim})"
            )

        alpha = x
        if self.learn or t != 1:
            alpha = x * t
        alpha = scatter_softmax(alpha, index, dim_size)
        if self.semi_grad:
            alpha = mx.stop_gradient(alpha)
        return self.reduce(x * alpha, index, dim_size, "add")

    def _extra_repr(self) -> str:
        return f"learn={self.learn}"


class PowerMeanAggregation(Aggregation):
    r"""Power mean aggregation with an (optionally learnable) power, from
    `"DeeperGCN: All You Need to Train Deeper GCNs"
    <https://arxiv.org/abs/2006.07739>`_ paper.

    .. math::
        \mathrm{powermean}(\mathcal{X}|p) = \left(\frac{1}{|\mathcal{X}|}
        \sum_{\mathbf{x}_i\in\mathcal{X}}\mathbf{x}_i^{p}\right)^{1/p}.

    As in PyG, unless :math:`p = 1`, the inputs and the mean are clamped to
    ``[clamp_min, clamp_max]`` (so the inputs are expected to be positive).

    Args:
        p: Initial power. Default: ``1.0``
        learn: If ``True``, :math:`p` is a learnable parameter.
            Default: ``False``
        channels: Number of learnable powers, one per feature channel when
            greater than 1. Requires ``learn=True``. Default: ``1``
        clamp_min: Lower bound of the clamping, or ``None``. Default: ``1e-4``
        clamp_max: Upper bound of the clamping, or ``None``. Default: ``100.``
    """

    def __init__(
        self,
        p: float = 1.0,
        learn: bool = False,
        channels: int = 1,
        clamp_min: Optional[float] = 1e-4,
        clamp_max: Optional[float] = 100.0,
    ):
        super().__init__()
        if not learn and channels != 1:
            raise ValueError(
                "Cannot set 'channels' greater than 1 in PowerMeanAggregation "
                "when 'p' is not learnable"
            )
        self._init_p = p
        self.learn = learn
        self.channels = channels
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        if learn:
            self.p = mx.full((channels,), p)
        else:
            self._p = p
        self.reset_parameters()

    def reset_parameters(self):
        """Resets the power to its initial value."""
        if self.learn:
            self.p = mx.full((self.channels,), self._init_p)

    def _clamp(self, x: mx.array) -> mx.array:
        if self.clamp_min is not None:
            x = mx.maximum(x, self.clamp_min)
        if self.clamp_max is not None:
            x = mx.minimum(x, self.clamp_max)
        return x

    def aggregate(self, x, index, dim_size, **kwargs):
        p = self.p if self.learn else self._p
        if self.channels != 1 and x.ndim != 2:
            raise ValueError(
                f"PowerMeanAggregation with channels > 1 requires "
                f"two-dimensional inputs (got {x.ndim})"
            )
        powered = self.learn or p != 1
        if powered:
            x = mx.power(self._clamp(x), p)
        out = self.reduce(x, index, dim_size, "mean")
        if powered:
            out = mx.power(self._clamp(out), 1.0 / p)
        return out

    def _extra_repr(self) -> str:
        return f"learn={self.learn}"
