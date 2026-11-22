from typing import Optional

import mlx.core as mx
import mlx.nn as nn


class PolynormerAttention(nn.Module):
    r"""The polynomial-expressive linear global attention from the
    `"Polynormer: Polynomial-Expressive Graph Transformer in Linear Time"
    <https://arxiv.org/abs/2403.01232>`_ paper, as in PyG.

    Sigmoid-kernelized attention is computed in linear time, normalized with
    a layer norm and gated by :math:`\mathbf{H} + \beta` where
    :math:`\mathbf{H}` is a linear transformation of the input.

    Args:
        channels: Size of each input sample.
        heads: Number of attention heads.
        head_channels: Size of each attention head. Default: ``64``
        beta: Polynormer :math:`\beta`. Default: ``0.9``
        qkv_bias: Whether the query, key and value projections have a bias.
            Default: ``False``
        qk_shared: Whether queries and keys share their weights.
            Default: ``True``
        dropout: Dropout probability of the attention output. Default: ``0.0``

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn.attention import PolynormerAttention

        attn = PolynormerAttention(channels=16, heads=2, head_channels=8)
        x = mx.random.normal((2, 5, 16))
        >>> attn(x).shape
        (2, 5, 16)

    References:
        - `Deng et al., "Polynormer: Polynomial-Expressive Graph Transformer
          in Linear Time", ICLR 2024 <https://arxiv.org/abs/2403.01232>`_
    """

    def __init__(
        self,
        channels: int,
        heads: int,
        head_channels: int = 64,
        beta: float = 0.9,
        qkv_bias: bool = False,
        qk_shared: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.head_channels = head_channels
        self.heads = heads
        self.beta = beta
        self.qk_shared = qk_shared
        inner_channels = heads * head_channels
        self.h_lins = nn.Linear(channels, inner_channels)
        if not qk_shared:
            self.q = nn.Linear(channels, inner_channels, bias=qkv_bias)
        self.k = nn.Linear(channels, inner_channels, bias=qkv_bias)
        self.v = nn.Linear(channels, inner_channels, bias=qkv_bias)
        self.lns = nn.LayerNorm(inner_channels)
        self.lin_out = nn.Linear(inner_channels, inner_channels)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        """Computes the attention.

        Args:
            x: Node features of shape ``[batch_size, num_nodes, channels]``.
            mask: Optional boolean mask of shape ``[batch_size, num_nodes]``,
                ``True`` for real nodes.

        Returns:
            An array of shape ``[batch_size, num_nodes, heads * head_channels]``.
        """
        B, N = x.shape[:2]
        shape = (B, N, self.head_channels, self.heads)
        h = self.h_lins(x)
        k = mx.sigmoid(self.k(x)).reshape(shape)
        q = k if self.qk_shared else mx.sigmoid(self.q(x)).reshape(shape)
        v = self.v(x).reshape(shape)
        if mask is not None:
            v = mx.where(mask[:, :, None, None], v, 0.0)

        kv = mx.einsum("bndh,bnmh->bdmh", k, v)
        num = mx.einsum("bndh,bdmh->bnmh", q, kv)
        den = mx.einsum("bndh,bdh->bnh", q, k.sum(axis=1))[:, :, None, :]
        out = (num / (den + 1e-6)).reshape(B, N, -1)
        out = self.lns(out) * (h + self.beta)
        out = nn.relu(self.lin_out(out))
        return self.dropout(out)
