from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.utils.dense import to_dense_batch

EPS = 1e-15


class MemPooling(nn.Module):
    r"""Memory based pooling layer from the `"Memory-Based Graph Networks"
    <https://arxiv.org/abs/2002.09518>`_ paper, which learns a coarsened
    graph representation based on soft cluster assignments.

    .. math::
        S_{i,j}^{(h)} &= \frac{
        (1+{\| \mathbf{x}_i-\mathbf{k}^{(h)}_j \|}^2 / \tau)^{
        -\frac{1+\tau}{2}}}{
        \sum_{k=1}^K (1 + {\| \mathbf{x}_i-\mathbf{k}^{(h)}_k \|}^2 / \tau)^{
        -\frac{1+\tau}{2}}}

        \mathbf{S} &= \textrm{softmax}(\textrm{Conv2d}
        (\Vert_{h=1}^H \mathbf{S}^{(h)})) \in \mathbb{R}^{N \times K}

        \mathbf{X}^{\prime} &= \mathbf{S}^{\top} \mathbf{X} \mathbf{W} \in
        \mathbb{R}^{K \times F^{\prime}}

    where :math:`H` denotes the number of heads, and :math:`K` the number of
    clusters (the keys :math:`\mathbf{k}^{(h)}_j` are learned). The
    auxiliary KL divergence loss of the paper is given by :meth:`kl_loss`.

    Args:
        node_features_dim: Size of the input node features.
        out_features_dim: Size of the output (pooled) features.
        heads: Number of heads :math:`H`.
        num_clusters: Number of clusters :math:`K` per head.
        tau: The temperature :math:`\tau`. Default: ``1.0``

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import MemPooling

        pool = MemPooling(16, 32, heads=4, num_clusters=3)
        node_features = mx.random.normal((7, 16))
        batch_indices = mx.array([0, 0, 0, 1, 1, 1, 1])
        x, S = pool(node_features, batch_indices)
        >>> x.shape, S.shape
        ((2, 3, 32), (2, 4, 3))
        loss = MemPooling.kl_loss(S)
    """

    def __init__(
        self,
        node_features_dim: int,
        out_features_dim: int,
        heads: int,
        num_clusters: int,
        tau: float = 1.0,
    ):
        super().__init__()
        self.node_features_dim = node_features_dim
        self.out_features_dim = out_features_dim
        self.heads = heads
        self.num_clusters = num_clusters
        self.tau = tau

        self.k = mx.random.uniform(-1.0, 1.0, (heads, num_clusters, node_features_dim))
        # 1x1 convolution over the heads, with PyTorch's initialization
        self.conv = nn.Conv2d(heads, 1, kernel_size=1, padding=0, bias=False)
        self.lin = nn.Linear(node_features_dim, out_features_dim, bias=False)

    @staticmethod
    def kl_loss(S: mx.array) -> mx.array:
        r"""The additional KL divergence-based loss.

        .. math::
            P_{i,j} &= \frac{S_{i,j}^2 / \sum_{n=1}^N S_{n,j}}{\sum_{k=1}^K
            S_{i,k}^2 / \sum_{n=1}^N S_{n,k}}

            \mathcal{L}_{\textrm{KL}} &= \textrm{KLDiv}(\mathbf{P} \Vert
            \mathbf{S})

        Args:
            S: The soft assignments of shape ``[batch_size, num_nodes,
                num_clusters]`` returned by the layer.

        Returns:
            The loss, averaged over the graphs of the batch.
        """
        P = (S * S) / S.sum(axis=1, keepdims=True)
        denom = P.sum(axis=2, keepdims=True)
        denom = mx.where(S.sum(axis=2, keepdims=True) == 0.0, 1.0, denom)
        P = mx.maximum(P / denom, EPS)
        log_S = mx.log(mx.maximum(S, EPS))
        return (P * (mx.log(P) - log_S)).sum() / S.shape[0]

    def __call__(
        self,
        node_features: mx.array,
        batch_indices: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        max_num_nodes: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> Tuple[mx.array, mx.array]:
        """Pools the graphs.

        Args:
            node_features: Node features, either of shape ``[num_nodes,
                node_features_dim]`` (with ``batch_indices``) or already dense,
                of shape ``[batch_size, num_nodes, node_features_dim]``.
            batch_indices: Graph of each node, of shape ``[num_nodes]``, for
                2D ``node_features``. Default: ``None`` (a single graph).
            mask: Boolean mask of the real nodes, of shape ``[batch_size,
                num_nodes]``, for dense ``node_features``. Default: ``None``
                (all the nodes are real).
            max_num_nodes: Number of nodes per graph of the dense batch built
                from 2D ``node_features``. Default: ``None`` (the size of the
                largest graph).
            batch_size: Number of graphs, for 2D ``node_features``.
                Default: ``None`` (inferred from ``batch_indices``).

        Returns:
            A tuple ``(node_features, S)`` with the pooled features of shape
            ``[batch_size, num_clusters, out_features_dim]`` and the soft
            assignments ``S`` of shape ``[batch_size, num_nodes,
            num_clusters]``.
        """
        x = node_features
        if x.ndim <= 2:
            x, mask = to_dense_batch(x, batch_indices, max_num_nodes=max_num_nodes)
            if batch_size is not None and batch_size > x.shape[0]:
                extra = batch_size - x.shape[0]
                x = mx.concatenate([x, mx.zeros((extra, *x.shape[1:]), x.dtype)])
                mask = mx.concatenate(
                    [mask, mx.zeros((extra, mask.shape[1]), dtype=mx.bool_)]
                )
        elif mask is None:
            mask = mx.ones(x.shape[:2], dtype=mx.bool_)

        (B, N, _), H, K = x.shape, self.heads, self.num_clusters

        keys = self.k.reshape(H * K, -1)
        flat = x.reshape(B * N, -1)
        dist = (
            (keys * keys).sum(axis=-1)[:, None]
            + (flat * flat).sum(axis=-1)[None, :]
            - 2 * (keys @ flat.T)
        )
        dist = mx.maximum(dist, 0.0)
        dist = (1.0 + dist / self.tau) ** (-(self.tau + 1.0) / 2.0)

        dist = dist.reshape(H, K, B, N).transpose(2, 3, 1, 0)  # [B, N, K, H]
        S = dist / dist.sum(axis=2, keepdims=True)
        S = mx.softmax(self.conv(S).squeeze(-1), axis=-1)  # [B, N, K]
        S = S * mask.reshape(B, N, 1)

        x = self.lin(S.swapaxes(1, 2) @ x)
        return x, S

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.node_features_dim}, "
            f"{self.out_features_dim}, heads={self.heads}, "
            f"num_clusters={self.num_clusters})"
        )
