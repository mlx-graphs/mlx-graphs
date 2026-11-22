import math
from typing import Callable, Optional

import mlx.core as mx
import mlx.nn as nn


def _orthogonal_block(dim: int) -> mx.array:
    q, _ = mx.linalg.qr(mx.random.normal((dim, dim)), stream=mx.cpu)
    return q.T


def orthogonal_matrix(num_rows: int, num_cols: int) -> mx.array:
    """Random matrix of shape ``[num_rows, num_cols]`` made of stacked blocks
    with orthonormal rows (from QR decompositions of Gaussian matrices), as in
    PyG's ``torch_geometric.nn.attention.performer.orthogonal_matrix``.

    Args:
        num_rows: Number of rows.
        num_cols: Number of columns.
    """
    num_full_blocks = num_rows // num_cols
    blocks = [_orthogonal_block(num_cols) for _ in range(num_full_blocks)]
    remain_rows = num_rows - num_full_blocks * num_cols
    if remain_rows > 0:
        blocks.append(_orthogonal_block(num_cols)[:remain_rows])
    return mx.concatenate(blocks, axis=0)


def linear_attention(q: mx.array, k: mx.array, v: mx.array) -> mx.array:
    r"""Linear attention :math:`\mathbf{\hat{D}}^{-1}(\mathbf{Q}'
    ((\mathbf{K}')^{\top} \mathbf{V}))` from the `"Rethinking Attention with
    Performers" <https://arxiv.org/abs/2009.14794>`_ paper, on arrays of shape
    ``[..., length, channels]``."""
    d_inv = 1.0 / (q @ k.sum(axis=-2)[..., None])
    return d_inv * (q @ (k.swapaxes(-2, -1) @ v))


class PerformerAttention(nn.Module):
    r"""The linear scaled attention mechanism from the `"Rethinking Attention
    with Performers" <https://arxiv.org/abs/2009.14794>`_ paper, as in PyG.

    Queries and keys are mapped by a fixed random orthogonal projection
    followed by ``kernel`` (plus :math:`10^{-3}`), which makes the attention
    linear in the number of nodes. The projection matrix is a frozen
    parameter that can be redrawn with :meth:`redraw_projection_matrix`.

    Args:
        channels: Size of each input sample.
        heads: Number of attention heads.
        head_channels: Size of each attention head. Default: ``64``
        kernel: Kernel of the generalized attention. Default: ``nn.relu``
        qkv_bias: Whether the query, key and value projections have a bias.
            Default: ``False``
        attn_out_bias: Whether the output projection has a bias.
            Default: ``True``
        dropout: Dropout probability of the attention output. Default: ``0.0``

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn.attention import PerformerAttention

        attn = PerformerAttention(channels=16, heads=2, head_channels=8)
        x = mx.random.normal((2, 5, 16))
        mask = mx.ones((2, 5), dtype=mx.bool_)
        >>> attn(x, mask).shape
        (2, 5, 16)

    References:
        - `Choromanski et al., "Rethinking Attention with Performers", ICLR
          2021 <https://arxiv.org/abs/2009.14794>`_
    """

    def __init__(
        self,
        channels: int,
        heads: int,
        head_channels: Optional[int] = 64,
        kernel: Callable[[mx.array], mx.array] = nn.relu,
        qkv_bias: bool = False,
        attn_out_bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        if channels % heads != 0:
            raise ValueError("`channels` must be divisible by `heads`")
        if head_channels is None:
            head_channels = channels // heads
        self.heads = heads
        self.head_channels = head_channels
        self.kernel = kernel
        self.num_rows = int(head_channels * math.log(head_channels))
        inner_channels = head_channels * heads
        self.q = nn.Linear(channels, inner_channels, bias=qkv_bias)
        self.k = nn.Linear(channels, inner_channels, bias=qkv_bias)
        self.v = nn.Linear(channels, inner_channels, bias=qkv_bias)
        self.attn_out = nn.Linear(inner_channels, channels, bias=attn_out_bias)
        self.dropout = nn.Dropout(dropout)
        self.projection_matrix = orthogonal_matrix(self.num_rows, head_channels)
        self.freeze(keys=["projection_matrix"], recurse=False)

    def redraw_projection_matrix(self):
        """Draws a new random projection matrix, as recommended in the paper
        to improve the approximation of the attention."""
        self.projection_matrix = orthogonal_matrix(self.num_rows, self.head_channels)

    def _feature_map(self, x: mx.array) -> mx.array:
        return self.kernel(x @ self.projection_matrix.T) + 0.001

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        """Computes the attention.

        Args:
            x: Node features of shape ``[batch_size, num_nodes, channels]``
                (e.g. from :func:`~mlx_graphs.utils.to_dense_batch`).
            mask: Optional boolean mask of shape ``[batch_size, num_nodes]``,
                ``True`` for real nodes.

        Returns:
            An array of shape ``[batch_size, num_nodes, channels]``.
        """
        B, N = x.shape[:2]

        def split(t):
            return t.reshape(B, N, self.heads, self.head_channels).transpose(0, 2, 1, 3)

        q, k, v = split(self.q(x)), split(self.k(x)), split(self.v(x))
        if mask is not None:
            v = mx.where(mask[:, None, :, None], v, 0.0)
        out = linear_attention(self._feature_map(q), self._feature_map(k), v)
        out = out.transpose(0, 2, 1, 3).reshape(B, N, -1)
        return self.dropout(self.attn_out(out))
