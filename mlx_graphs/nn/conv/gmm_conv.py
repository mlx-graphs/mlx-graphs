from typing import Tuple, Union

import mlx.core as mx

from mlx_graphs.nn.linear import Linear, glorot
from mlx_graphs.nn.message_passing import MessagePassing

_EPS = 1e-15


class GMMConv(MessagePassing):
    r"""The Gaussian mixture model convolutional operator from the `"Geometric
    Deep Learning on Graphs and Manifolds using Mixture Model CNNs"
    <https://arxiv.org/abs/1611.08402>`_ paper.

    .. math::
        \mathbf{x}^{\prime}_i = \frac{1}{|\mathcal{N}(i)|}
        \sum_{j \in \mathcal{N}(i)} \frac{1}{K} \sum_{k=1}^K
        \mathbf{w}_k(\mathbf{e}_{i,j}) \odot \mathbf{\Theta}_k \mathbf{x}_j,

    where

    .. math::
        \mathbf{w}_k(\mathbf{e}) = \exp \left( -\frac{1}{2} {\left(
        \mathbf{e} - \mathbf{\mu}_k \right)}^{\top} \Sigma_k^{-1}
        \left( \mathbf{e} - \mathbf{\mu}_k \right) \right)

    denotes a weighting function based on trainable mean vectors
    :math:`\mathbf{\mu}_k` and diagonal covariance matrices
    :math:`\mathbf{\Sigma}_k` (``sigma``). As in PyG, the sum over the kernels
    is not divided by :math:`K`.

    Args:
        node_features_dim: Size of the input features, or a ``(src_dim,
            dst_dim)`` tuple for bipartite graphs.
        out_features_dim: Size of the output features.
        dim: Pseudo-coordinate dimensionality.
        kernel_size: Number of kernels :math:`K`.
        separate_gaussians: If ``True``, learns separate Gaussians for each
            pair of input and output channels. Default: ``False``
        aggr: Aggregation of the messages. Default: ``"mean"``
        root_weight: Whether to add the transformed root node features.
            Default: ``True``
        bias: Whether to add a bias. Default: ``True``
        **kwargs: Arguments of :class:`MessagePassing`

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import GMMConv

        edge_index = mx.array([[0, 1, 2, 3], [1, 0, 3, 2]])
        x = mx.random.normal((4, 16))
        pseudo = mx.random.uniform(shape=(4, 2))
        conv = GMMConv(16, 32, dim=2, kernel_size=4)
        >>> conv(edge_index, x, pseudo).shape
        (4, 32)
    """

    supports_concat_aggr = False

    def __init__(
        self,
        node_features_dim: Union[int, Tuple[int, int]],
        out_features_dim: int,
        dim: int,
        kernel_size: int,
        separate_gaussians: bool = False,
        aggr: str = "mean",
        root_weight: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(aggr=aggr, **kwargs)
        self.node_features_dim = node_features_dim
        self.out_features_dim = out_features_dim
        self.dim = dim
        self.kernel_size = kernel_size
        self.separate_gaussians = separate_gaussians
        self.root_weight = root_weight
        src_dim, dst_dim = (
            (node_features_dim, node_features_dim)
            if isinstance(node_features_dim, int)
            else node_features_dim
        )
        self.src_features_dim = src_dim
        self.g = glorot((src_dim, out_features_dim * kernel_size))
        if separate_gaussians:
            shape = (src_dim, out_features_dim, kernel_size, dim)
        else:
            shape = (kernel_size, dim)
        self.mu = glorot(shape)
        self.sigma = glorot(shape)
        if root_weight:
            self.root = Linear(dst_dim, out_features_dim, bias=False)
        if bias:
            self.bias = mx.zeros((out_features_dim,))

    def __call__(
        self,
        edge_index: mx.array,
        node_features: Union[mx.array, Tuple[mx.array, mx.array]],
        edge_features: mx.array,
    ) -> mx.array:
        """Computes the forward pass of GMMConv.

        Args:
            edge_index: Edges of shape ``[2, num_edges]``.
            node_features: Node features, or a ``(src_features,
                dst_features)`` tuple for bipartite graphs.
            edge_features: Pseudo-coordinates of shape ``[num_edges, dim]``.

        Returns:
            The features of the destination nodes.
        """
        if isinstance(node_features, tuple):
            x_src, x_dst = node_features
        else:
            x_src = x_dst = node_features
        if not self.separate_gaussians:
            x_src = x_src @ self.g
        out = self.propagate(
            edge_index=edge_index,
            node_features=(x_src, x_dst),
            message_kwargs={"edge_features": edge_features},
        )
        if self.root_weight:
            out = out + self.root(x_dst)
        if "bias" in self:
            out = out + self.bias
        return out

    def message(
        self,
        src_features: mx.array,
        dst_features: mx.array,
        edge_features: mx.array,
        **kwargs,
    ) -> mx.array:
        F, M = self.src_features_dim, self.out_features_dim
        K, D = self.kernel_size, self.dim
        E = edge_features.shape[0]
        if not self.separate_gaussians:
            gaussian = (
                -0.5 * (edge_features.reshape(E, 1, D) - self.mu.reshape(1, K, D)) ** 2
            )
            gaussian = gaussian / (_EPS + self.sigma.reshape(1, K, D) ** 2)
            gaussian = mx.exp(gaussian.sum(axis=-1))  # [E, K]
            return (src_features.reshape(E, K, M) * gaussian.reshape(E, K, 1)).sum(
                axis=-2
            )
        gaussian = (
            -0.5
            * (edge_features.reshape(E, 1, 1, 1, D) - self.mu.reshape(1, F, M, K, D))
            ** 2
        )
        gaussian = gaussian / (_EPS + self.sigma.reshape(1, F, M, K, D) ** 2)
        gaussian = mx.exp(gaussian.sum(axis=-1))  # [E, F, M, K]
        gaussian = (gaussian * self.g.reshape(1, F, M, K)).sum(axis=-1)  # [E, F, M]
        return (src_features.reshape(E, F, 1) * gaussian).sum(axis=-2)  # [E, M]
