import mlx.core as mx
import mlx.nn as nn


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
