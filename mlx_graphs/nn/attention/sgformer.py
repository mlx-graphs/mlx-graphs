from typing import Optional

import mlx.core as mx
import mlx.nn as nn


class SGFormerAttention(nn.Module):
    r"""The simple global attention mechanism from the `"SGFormer:
    Simplifying and Empowering Transformers for Large-Graph Representations"
    <https://arxiv.org/abs/2306.10759>`_ paper, as in PyG.

    Queries and keys are :math:`\ell_2`-normalized and the all-pair attention
    is computed in linear time:

    .. math::
        \mathbf{Z} = \mathrm{diag}^{-1}\left(N + \mathbf{Q}
        (\mathbf{K}^{\top} \mathbf{1})\right) \left(N \mathbf{V} + \mathbf{Q}
        (\mathbf{K}^{\top} \mathbf{V})\right)

    The outputs of the heads are averaged.

    Args:
        channels: Size of each input sample.
        heads: Number of attention heads. Default: ``1``
        head_channels: Size of each attention head. Default: ``64``
        qkv_bias: Whether the query, key and value projections have a bias.
            Default: ``False``

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn.attention import SGFormerAttention

        attn = SGFormerAttention(channels=16, heads=2, head_channels=8)
        x = mx.random.normal((2, 5, 16))
        >>> attn(x).shape
        (2, 5, 8)

    References:
        - `Wu et al., "SGFormer: Simplifying and Empowering Transformers for
          Large-Graph Representations", NeurIPS 2023
          <https://arxiv.org/abs/2306.10759>`_
    """

    def __init__(
        self,
        channels: int,
        heads: int = 1,
        head_channels: Optional[int] = 64,
        qkv_bias: bool = False,
    ):
        super().__init__()
        if channels % heads != 0:
            raise ValueError("`channels` must be divisible by `heads`")
        if head_channels is None:
            head_channels = channels // heads
        self.heads = heads
        self.head_channels = head_channels
        inner_channels = head_channels * heads
        self.q = nn.Linear(channels, inner_channels, bias=qkv_bias)
        self.k = nn.Linear(channels, inner_channels, bias=qkv_bias)
        self.v = nn.Linear(channels, inner_channels, bias=qkv_bias)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        """Computes the attention.

        Args:
            x: Node features of shape ``[batch_size, num_nodes, channels]``.
            mask: Optional boolean mask of shape ``[batch_size, num_nodes]``,
                ``True`` for real nodes.

        Returns:
            An array of shape ``[batch_size, num_nodes, head_channels]``.
        """
        B, N = x.shape[:2]
        shape = (B, N, self.heads, self.head_channels)
        qs, ks, vs = (lin(x).reshape(shape) for lin in (self.q, self.k, self.v))
        if mask is not None:
            vs = mx.where(mask[:, :, None, None], vs, 0.0)

        epsilon = 1e-6
        qs = mx.where(qs == 0, epsilon, qs)
        ks = mx.where(ks == 0, epsilon, ks)
        qs = qs / mx.linalg.norm(qs, axis=-1, keepdims=True)
        ks = ks / mx.linalg.norm(ks, axis=-1, keepdims=True)

        # [B, H, M, D]
        kvs = mx.einsum("blhm,blhd->bhmd", ks, vs)
        numerator = mx.einsum("bnhm,bhmd->bnhd", qs, kvs) + N * vs
        ks_sum = ks.sum(axis=1)  # [B, H, M]
        normalizer = mx.einsum("bnhm,bhm->bnh", qs, ks_sum)[..., None] + N
        return (numerator / normalizer).mean(axis=2)
