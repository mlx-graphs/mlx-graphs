from typing import Literal, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.backend import on_cuda
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
        num_features: Size of each input sample.
        eps: A value added to the denominator for numerical
            stability. (default: :obj:`1e-5`)
        affine: If set to :obj:`True`, this module has
            learnable affine parameters :math:`\gamma` and :math:`\beta`
            (``weight`` and ``bias``, used by both modes, as in PyG).
            (default: :obj:`True`)
        mode: The normalization mode to use for layer
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
        if mode not in ("graph", "node"):
            raise ValueError(f"Unknown normalization mode: {mode}")
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.mode = mode

        # Used by both modes, as in PyG.
        if affine:
            self.weight = mx.ones((num_features,))
            self.bias = mx.zeros((num_features,))

    def __call__(
        self,
        features: mx.array,
        batch: Optional[mx.array] = None,
        batch_size: Optional[int] = None,
    ) -> mx.array:
        """Normalizes the features.

        Args:
            features: Features of shape ``[num_nodes, num_features]``.
            batch: Graph of each node, of shape ``[num_nodes]`` (``"graph"``
                mode). Default: ``None`` (all the nodes form one graph).
            batch_size: Number of graphs. Default: ``None`` (inferred from
                ``batch``).

        Returns:
            The normalized features.
        """
        weight = self.weight if self.affine else None
        bias = self.bias if self.affine else None
        if self.mode == "node":
            # MLX's CUDA backend (0.32) gets the gradient of mx.fast.layer_norm
            # wrong when the feature size is not a multiple of 4.
            if not (features.shape[-1] % 4 and on_cuda()):
                return mx.fast.layer_norm(features, weight, bias, self.eps)
            features = features - features.mean(axis=-1, keepdims=True)
            var = (features * features).mean(axis=-1, keepdims=True)
            out = features * mx.rsqrt(var + self.eps)
            if self.affine:
                out = out * weight + bias
            return out

        # "graph" mode: statistics over all the nodes and channels of a graph
        if batch is None:
            features = features - features.mean()
            out = features / (mx.sqrt(features.var()) + self.eps)
        else:
            if batch_size is None:
                batch_size = int(batch.max().item()) + 1 if batch.size > 0 else 1
            count = mx.maximum(degree(batch, batch_size), 1) * features.shape[-1]
            count = count.astype(features.dtype).reshape(-1, 1)
            mean = scatter(features, batch, batch_size, "add").sum(
                axis=-1, keepdims=True
            )
            features = features - (mean / count)[batch]
            var = scatter(features * features, batch, batch_size, "add").sum(
                axis=-1, keepdims=True
            )
            out = features / mx.sqrt(var / count + self.eps)[batch]
        if self.affine:
            out = out * weight + bias
        return out


class HeteroLayerNormalization(nn.Module):
    r"""Applies layer normalization over each individual example in a batch
    of heterogeneous features, as described in the `"Layer Normalization"
    <https://arxiv.org/abs/1607.06450>`_ paper.

    Compared to a standard layer norm, :class:`HeteroLayerNorm` applies
    normalization individually for each node or edge type. Note that the
    normalization itself (per-node mean/variance across channels) does not
    depend on the type: only the learnable affine parameters do.

    Args:
        in_channels (int): Size of each input sample.
        num_types (int): The number of types.
        eps (float, optional): A value added to the denominator for
            numerical stability. (default: :obj:`1e-5`)
        affine (bool, optional): If set to :obj:`True`, this module has
            learnable affine parameters :math:`\gamma` and :math:`\beta`.
            (default: :obj:`True`)
    """

    def __init__(
        self,
        in_channels: int,
        num_types: int,
        eps: float = 1e-5,
        affine: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_types = num_types
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = mx.ones((num_types, in_channels))
            self.bias = mx.zeros((num_types, in_channels))

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
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
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        out = (x - mean) / mx.sqrt(var + self.eps)

        if self.affine:
            out = out * self.weight[type_vec] + self.bias[type_vec]

        return out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"num_types={self.num_types})"
        )
