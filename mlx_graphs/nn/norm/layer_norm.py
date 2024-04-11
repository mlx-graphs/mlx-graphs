from typing import Any, Literal, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.utils import degree, scatter


class LayerNormalization(nn.Module):
    r"""Applies layer normalization over each individual example in a batch
    of features as described in the `Layer Normalization <https://arxiv.org/abs/1607.06450>`_
    paper.

    .. math::

        \mathbf{x}^{\prime}_i = \frac{\mathbf{x} -
        \textrm{E}[\mathbf{x}]}{\sqrt{\textrm{Var}[\mathbf{x}] + \epsilon}}
        \odot \gamma + \beta

    The mean and standard-deviation are calculated across all nodes and all
    node channels separately for each object in a mini-batch.

    Args:
        in_channels : Size of each input sample.
        eps : A value added to the denominator for numerical
            stability. (default: :obj:`1e-5`)
        affine : If set to :obj:`True`, this module has
            learnable affine parameters :math:`\gamma` and :math:`\beta`.
            (default: :obj:`True`)
        mode : The normalization mode to use for layer
            normalization (:obj:`"graph"` or :obj:`"node"`). If :obj:`"graph"`
            is used, each graph will be considered as an element to be
            normalized. If `"node"` is used, each node will be considered as
            an element to be normalized. (default: :obj:`"graph"`)
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = True,
        mode: Literal["graph", "node"] = "graph",
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.mode = mode

        if affine:
            self.weight = mx.ones((num_features,))
            self.bias = mx.zeros((num_features,))

        self.layernorm = nn.LayerNorm(
            dims=self.num_features, eps=self.eps, affine=self.affine
        )

    def __call__(
        self,
        features: mx.array,
        batch: Optional[mx.array] = None,
        batch_size: Optional[int] = None,
    ) -> Any:
        if self.mode == "graph":
            # perform graph level normalization
            if batch is None:
                features = features - features.mean()
                features = features / (features.var().sqrt() + self.eps)
            else:
                if batch_size is None:
                    batch_size = batch.max().item() + 1
                batch_index = batch
                # try getting degrees of nodes in a batch
                norm = mx.clip(degree(batch_index), a_min=1, a_max=None)
                norm = norm * (features.shape[-1])
                norm = norm.T
                mean = (
                    scatter(
                        features,
                        batch_index,
                        batch_index.max().item() + 1,
                        aggr="add",
                        axis=0,
                    ).sum(axis=-1, keepdims=True)
                    / norm
                )

                node_features = features - mx.take(mean, batch_index, axis=0)
                variance = scatter(
                    node_features * node_features,
                    batch_index,
                    batch_index.max().item() + 1,
                    aggr="add",
                    axis=0,
                ).sum(axis=-1, keepdims=True)
                variance = variance / norm
                out = node_features / mx.take(
                    (variance + self.eps).sqrt(), batch_index, axis=0
                )
            if self.affine and self.weight is not None and self.bias is not None:
                out = out * self.weight + self.bias

            return out

        elif self.mode == "node":
            return self.layernorm(features)
        else:
            raise ValueError(f"Unknow normalization mode: {self.mode}")
