import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.utils.spatial import knn


class _DepthwiseConv(nn.Module):
    """``torch.nn.Conv1d(channels, channels * multiplier, kernel_size,
    groups=channels)`` applied to inputs of length ``kernel_size`` (one output
    position), on inputs of shape ``[N, channels, kernel_size]``. The weight has
    PyTorch's layout ``[channels * multiplier, 1, kernel_size]``."""

    def __init__(
        self, channels: int, multiplier: int, kernel_size: int, bias: bool = True
    ):
        super().__init__()
        self.channels = channels
        self.multiplier = multiplier
        bound = 1.0 / math.sqrt(kernel_size)
        self.weight = mx.random.uniform(
            -bound, bound, (channels * multiplier, 1, kernel_size)
        )
        if bias:
            self.bias = mx.random.uniform(-bound, bound, (channels * multiplier,))

    def __call__(self, x: mx.array) -> mx.array:
        weight = self.weight.reshape(self.channels, self.multiplier, -1)
        out = mx.einsum("nck,cmk->ncm", x, weight).reshape(x.shape[0], -1)
        if "bias" in self:
            out = out + self.bias
        return out


class XConv(nn.Module):
    r"""The convolutional operator on :math:`\mathcal{X}`-transformed points
    from the `"PointCNN: Convolution On X-Transformed Points"
    <https://arxiv.org/abs/1801.07791>`_ paper.

    .. math::
        \mathbf{x}^{\prime}_i = \mathrm{Conv}\left(\mathbf{K},
        \gamma_{\mathbf{\Theta}}(\mathbf{P}_i - \mathbf{p}_i) \times
        \left( h_\mathbf{\Theta}(\mathbf{P}_i - \mathbf{p}_i) \, \Vert \,
        \mathbf{x}_i \right) \right),

    where :math:`\mathbf{K}` and :math:`\mathbf{P}_i` denote the trainable
    filter and the neighboring point positions of :math:`\mathbf{x}_i`, and
    :math:`\gamma_{\mathbf{\Theta}}` and :math:`h_{\mathbf{\Theta}}` are
    neural networks. The neighbors are the ``kernel_size * dilation`` nearest
    points (including the point itself), of which every ``dilation``-th one is
    kept, as in PyG.

    Each graph must have at least ``kernel_size * dilation`` points.

    Args:
        node_features_dim: Size of the input features (``0`` without
            features).
        out_features_dim: Size of the output features.
        dim: Point cloud dimensionality.
        kernel_size: Size of the convolving kernel, i.e. the number of
            neighbors.
        hidden_features_dim: Size of the lifted point positions. Default:
            ``None`` (``node_features_dim // 4``).
        dilation: The factor by which the neighborhood is extended. Default:
            ``1``
        bias: Whether the final linear layer has a bias. Default: ``True``

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import XConv

        pos = mx.random.normal((100, 3))
        x = mx.random.normal((100, 16))
        conv = XConv(16, 32, dim=3, kernel_size=8)
        >>> conv(x, pos).shape
        (100, 32)
    """

    def __init__(
        self,
        node_features_dim: int,
        out_features_dim: int,
        dim: int,
        kernel_size: int,
        hidden_features_dim: Optional[int] = None,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        if hidden_features_dim is None:
            hidden_features_dim = node_features_dim // 4
        if hidden_features_dim <= 0:
            raise ValueError("hidden_features_dim must be positive")
        self.node_features_dim = node_features_dim
        self.hidden_features_dim = hidden_features_dim
        self.out_features_dim = out_features_dim
        self.dim = dim
        self.kernel_size = kernel_size
        self.dilation = dilation

        C_in, C_delta, C_out = node_features_dim, hidden_features_dim, out_features_dim
        D, K = dim, kernel_size
        self.mlp1 = nn.Sequential(
            nn.Linear(D, C_delta),
            nn.ELU(),
            nn.BatchNorm(C_delta),
            nn.Linear(C_delta, C_delta),
            nn.ELU(),
            nn.BatchNorm(C_delta),
        )
        self.mlp2_lin = nn.Linear(D * K, K**2)
        self.mlp2_bn1 = nn.BatchNorm(K**2)
        self.mlp2_conv1 = _DepthwiseConv(K, K, K)
        self.mlp2_bn2 = nn.BatchNorm(K**2)
        self.mlp2_conv2 = _DepthwiseConv(K, K, K)
        self.mlp2_bn3 = nn.BatchNorm(K**2)

        C_in = C_in + C_delta
        depth_multiplier = int(math.ceil(C_out / C_in))
        self.conv = _DepthwiseConv(C_in, depth_multiplier, K)
        self.lin = nn.Linear(C_in * depth_multiplier, C_out, bias=bias)

    def _mlp2(self, pos: mx.array) -> mx.array:
        K = self.kernel_size
        t = self.mlp2_bn1(nn.elu(self.mlp2_lin(pos)))
        t = self.mlp2_bn2(nn.elu(self.mlp2_conv1(t.reshape(-1, K, K))))
        t = self.mlp2_bn3(self.mlp2_conv2(t.reshape(-1, K, K)))
        return t.reshape(-1, K, K)

    def __call__(
        self,
        node_features: Optional[mx.array],
        pos: mx.array,
        batch_indices: Optional[mx.array] = None,
    ) -> mx.array:
        """Computes the forward pass of XConv.

        Args:
            node_features: Point features of shape ``[N, node_features_dim]``,
                or ``None``.
            pos: Point positions of shape ``[N, dim]``.
            batch_indices: Graph of each point, of shape ``[N]``. Default:
                ``None`` (a single point cloud).

        Returns:
            The features of the points, of shape ``[N, out_features_dim]``.
        """
        pos = pos.reshape(-1, 1) if pos.ndim == 1 else pos
        (N, D), K = pos.shape, self.kernel_size

        edge_index = knn(pos, pos, K * self.dilation, batch_indices, batch_indices)
        if edge_index.shape[1] != N * K * self.dilation:
            raise ValueError(
                f"XConv needs at least kernel_size * dilation = "
                f"{K * self.dilation} points in each graph"
            )
        if self.dilation > 1:
            edge_index = edge_index[:, :: self.dilation]
        row, col = edge_index[0], edge_index[1]

        rel = pos[col] - pos[row]  # [N * K, D]
        x_star = self.mlp1(rel).reshape(N, K, self.hidden_features_dim)
        if node_features is not None:
            x = (
                node_features.reshape(-1, 1)
                if node_features.ndim == 1
                else node_features
            )
            x_star = mx.concatenate(
                [x_star, x[col].reshape(N, K, self.node_features_dim)], axis=-1
            )
        x_star = x_star.transpose(0, 2, 1)  # [N, C, K]

        transform = self._mlp2(rel.reshape(N, K * D))  # [N, K, K]
        x_transformed = x_star @ transform  # [N, C, K]
        return self.lin(self.conv(x_transformed))
