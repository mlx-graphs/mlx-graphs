from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.utils.scatter import scatter


class BatchNormalization(nn.Module):
    r"""Applies batch normalization over a batch of features as described in
    the `Batch Normalization: Accelerating Deep Network Training by
    Reducing Internal Covariate Shift" <https://arxiv.org/abs/1502.03167>`_
    paper.

    .. math::
        \mathbf{x}^{\prime}_i = \frac{\mathbf{x} -
        \textrm{E}[\mathbf{x}]}{\sqrt{\textrm{Var}[\mathbf{x}] + \epsilon}}
        \odot \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over all nodes
    inside the mini-batch.

    Args:
        in_channels : Size of each input sample.
        eps : A value added to the denominator for numerical
            stability. (default: :obj:`1e-5`)
        momentum : The value used for the running mean and
            running variance computation. (default: :obj:`0.1`)
        affine : If set to :obj:`True`, this module has
            learnable affine parameters :math:`\gamma` and :math:`\beta`.
            (default: :obj:`True`)
        track_running_stats : If set to :obj:`True`, this
            module tracks the running mean and variance, and when set to
            :obj:`False`, this module does not track such statistics and always
            uses batch statistics in both training and eval modes.
            (default: :obj:`True`)
        allow_single_element : If set to :obj:`True`, batches
            with only a single element will work as during in evaluation.
            That is the running mean and variance will be used.
            Requires :obj:`track_running_stats=True`. (default: :obj:`False`)

    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        allow_single_element: bool = False,
    ):
        super().__init__()
        if allow_single_element and not track_running_stats:
            raise ValueError(
                "'allow_single_element' requires "
                "'track_running_stats' to be set to `True`"
            )

        self.module = nn.BatchNorm(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )
        self.num_features = num_features
        self.allow_single_element = allow_single_element

    def __call__(self, x: mx.array):
        if self.allow_single_element and x.shape[0] <= 1:
            x = (x - self.module.running_mean) * mx.rsqrt(
                self.module.running_var + self.module.eps
            )
            return self.module.weight * x + self.module.bias
        return self.module(x)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.module.num_features})"


class HeteroBatchNormalization(nn.Module):
    r"""Applies batch normalization over a batch of heterogeneous features,
    as described in the `"Batch Normalization: Accelerating Deep Network
    Training by Reducing Internal Covariate Shift"
    <https://arxiv.org/abs/1502.03167>`_ paper.

    Compared to a standard batch norm, :class:`HeteroBatchNorm` applies
    normalization individually for each node or edge type.

    Args:
        in_channels (int): Size of each input sample.
        num_types (int): The number of types.
        eps (float, optional): A value added to the denominator for
            numerical stability. (default: :obj:`1e-5`)
        momentum (float, optional): The value used for the running mean and
            running variance computation. If set to :obj:`None`, a
            cumulative moving average is used instead.
            (default: :obj:`0.1`)
        affine (bool, optional): If set to :obj:`True`, this module has
            learnable affine parameters :math:`\gamma` and :math:`\beta`.
            (default: :obj:`True`)
        track_running_stats (bool, optional): If set to :obj:`True`, this
            module tracks the running mean and variance, and when set to
            :obj:`False`, this module does not track such statistics and
            always uses batch statistics in both training and eval modes.
            (default: :obj:`True`)
    """

    def __init__(
        self,
        in_channels: int,
        num_types: int,
        eps: float = 1e-5,
        momentum: Optional[float] = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_types = num_types
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = mx.ones((num_types, in_channels))
            self.bias = mx.zeros((num_types, in_channels))

        if self.track_running_stats:
            self.running_mean = mx.zeros((num_types, in_channels))
            self.running_var = mx.ones((num_types, in_channels))
            self.num_batches_tracked = mx.array(0)
            # Running stats are not learnable: exclude them from gradients.
            self.freeze(
                keys=["running_mean", "running_var", "num_batches_tracked"],
                recurse=False,
            )

        self.reset_parameters()

    def reset_running_stats(self):
        r"""Resets all running statistics of the module."""
        if self.track_running_stats:
            self.running_mean = mx.zeros_like(self.running_mean)
            self.running_var = mx.ones_like(self.running_var)
            self.num_batches_tracked = mx.array(0)

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.reset_running_stats()
        if self.affine:
            self.weight = mx.ones_like(self.weight)
            self.bias = mx.zeros_like(self.bias)

    def __call__(self, x: mx.array, type_vec: mx.array) -> mx.array:
        r"""Forward pass.

        Args:
            x (mx.array): The input features of shape
                :obj:`[num_items, in_channels]`.
            type_vec (mx.array): An integer vector of shape
                :obj:`[num_items]` that maps each entry to a type in
                :obj:`[0, num_types)`.
        """
        if not self.training and self.track_running_stats:
            mean, var = self.running_mean, self.running_var
        else:
            mean = scatter(x, type_vec, out_size=self.num_types, aggr="mean")
            mean_sq = scatter(x * x, type_vec, out_size=self.num_types, aggr="mean")
            var = mx.maximum(mean_sq - mean * mean, 0.0)

        if self.training and self.track_running_stats:
            if self.momentum is None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                exp_avg_factor = 1.0 / float(self.num_batches_tracked.item())
            else:
                exp_avg_factor = self.momentum

            # Only update stats for types actually present in this batch,
            # leaving stats for absent types untouched.
            ones = mx.ones((type_vec.shape[0], 1))
            counts = scatter(ones, type_vec, out_size=self.num_types, aggr="add")
            mask = counts > 0

            new_running_mean = (
                1.0 - exp_avg_factor
            ) * self.running_mean + exp_avg_factor * mean
            new_running_var = (
                1.0 - exp_avg_factor
            ) * self.running_var + exp_avg_factor * var

            self.running_mean = mx.where(mask, new_running_mean, self.running_mean)
            self.running_var = mx.where(mask, new_running_var, self.running_var)

        out = (x - mean[type_vec]) / mx.sqrt(mx.maximum(var, self.eps))[type_vec]

        if self.affine:
            out = out * self.weight[type_vec] + self.bias[type_vec]

        return out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"num_types={self.num_types})"
        )
