from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.utils import degree, scatter


class InstanceNormalization(nn.Module):
    r"""Instance normalization over each individual example in batch of node features
    as described in the paper `Instance Normalization: The Missing \
    Ingredient for Fast Stylization <https://arxiv.org/abs/1607.08022>`_
    paper.

    .. math::

        \mathbf{x}^{\prime}_i = \frac{\mathbf{x} -
        \textrm{E}[\mathbf{x}]}{\sqrt{\textrm{Var}[\mathbf{x}] + \epsilon}}
        \odot \gamma + \beta

    The mean and standard-deviation are calculated per-dimension separately for
    each object in a mini-batch.

    Args:
        in_channels : Size of each input sample.
        eps : A value added to the denominator for numerical
            stability. (default: :obj:`1e-5`)
        momentum : The value used for the running mean and
            running variance computation. (default: :obj:`0.1`)
        affine : If set to :obj:`True`, this module has
            learnable affine parameters :math:`\gamma` and :math:`\beta`.
            (default: :obj:`False`)
        track_running_stats : If set to :obj:`True`, this
            module tracks the running mean and variance, and when set to
            :obj:`False`, this module does not track such statistics and always
            uses instance statistics in both training and eval modes.
            (default: :obj:`False`)
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = False,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.instance_norm = nn.InstanceNorm(num_features, self.eps, self.affine)
        if affine:
            self.weight = mx.ones((num_features,))
            self.bias = mx.zeros((num_features,))

        if self.track_running_stats:
            self.running_mean = mx.zeros((num_features,))
            self.running_var = mx.ones((num_features,))
            self.freeze(keys=["running_mean", "running_var"], recurse=False)

    def __call__(
        self,
        node_features: mx.array,
        batch: Optional[mx.array] = None,
        batch_size: Optional[int] = None,
    ):
        if batch is None:
            return self.instance_norm(mx.expand_dims(node_features, axis=0)).squeeze(
                axis=0
            )

        if batch_size is None:
            batch_size = batch.max().item() + 1

        mean = var = unbiased_var = node_features
        if self.training or not self.track_running_stats:
            norm = mx.clip(degree(batch), a_min=1, a_max=None)
            norm = norm.reshape(-1, 1)
            unbiased_norm = mx.clip(norm - 1, a_min=1, a_max=None)

            mean = (
                scatter(
                    node_features,
                    batch,
                    batch.max().item() + 1,
                    aggr="add",
                    axis=0,
                )
                / norm
            )
            node_features = node_features - mx.take(mean, batch, axis=0)

            var = scatter(
                node_features * node_features,
                batch,
                batch.max().item() + 1,
                aggr="add",
                axis=0,
            )
            unbiased_var = var / unbiased_norm
            var = var / norm

            momentum = self.momentum

            if self.track_running_stats and self.running_mean is not None:
                self.running_mean = (
                    1 - momentum
                ) * self.running_mean + momentum * mean.mean(0)
            if self.track_running_stats and self.running_var is not None:
                self.running_var = (
                    1 - momentum
                ) * self.running_var + momentum * unbiased_var.mean(0)
        else:
            if self.track_running_stats and self.running_mean is not None:
                mean = mx.repeat(self.running_mean.reshape(1, -1), batch_size, axis=0)
            if self.track_running_stats and self.running_var is not None:
                var = mx.repeat(self.running_var.reshape(1, -1), batch_size, axis=0)

            node_features = node_features - mx.take(mean, batch, axis=0)

        out = node_features / mx.take((var + self.eps).sqrt(), batch, axis=0)

        if self.affine and self.weight is not None and self.bias is not None:
            out = out * self.weight + self.bias
        return out
