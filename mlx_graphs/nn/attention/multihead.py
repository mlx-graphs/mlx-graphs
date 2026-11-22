import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


class MultiheadAttention(nn.Module):
    r"""Multi-head scaled dot-product attention with the semantics of
    ``torch.nn.MultiheadAttention(..., batch_first=True)``.

    Unlike :class:`mlx.nn.MultiHeadAttention`, it has biases in all the
    projections (as in PyTorch), a ``key_padding_mask`` and dropout on the
    attention weights, which PyG's graph transformers (e.g.
    :class:`~mlx_graphs.nn.GPSConv`, the set transformer aggregations) rely on.

    .. math::
        \mathrm{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) =
        \mathrm{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^{\top}}{\sqrt{d}}
        \right) \mathbf{V}

    PyTorch packs the query, key and value projections in a single
    ``in_proj_weight`` of shape ``[3 * dims, dims]``; here they are the
    separate ``query_proj``, ``key_proj`` and ``value_proj`` layers (the
    first, second and third block of ``in_proj_weight``). The weights are
    initialized as in PyTorch: Xavier uniform over the packed projection,
    zero biases, and a PyTorch default ``out_proj`` weight.

    Args:
        dims: Size of the inputs and outputs (``embed_dim`` in PyTorch).
        num_heads: Number of attention heads. ``dims`` must be divisible by
            ``num_heads``.
        dropout: Dropout probability on the attention weights. Default: ``0.0``
        bias: Whether the projections have a bias. Default: ``True``

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn.attention import MultiheadAttention

        attn = MultiheadAttention(16, num_heads=4)
        x = mx.random.normal((2, 5, 16))
        mask = mx.array([[True] * 5, [True] * 3 + [False] * 2])
        >>> attn(x, x, x, key_mask=mask).shape
        (2, 5, 16)
    """

    def __init__(
        self, dims: int, num_heads: int, dropout: float = 0.0, bias: bool = True
    ):
        super().__init__()
        if dims % num_heads != 0:
            raise ValueError(
                f"`dims` ({dims}) must be divisible by `num_heads` ({num_heads})"
            )
        self.dims = dims
        self.num_heads = num_heads
        self.dropout = dropout

        limit = math.sqrt(6.0 / (dims + 3 * dims))  # xavier over [3 * dims, dims]
        self.query_proj = nn.Linear(dims, dims, bias=bias)
        self.key_proj = nn.Linear(dims, dims, bias=bias)
        self.value_proj = nn.Linear(dims, dims, bias=bias)
        self.out_proj = nn.Linear(dims, dims, bias=bias)
        for proj in (self.query_proj, self.key_proj, self.value_proj):
            proj.weight = mx.random.uniform(-limit, limit, (dims, dims))
        if bias:
            for proj in (self.query_proj, self.key_proj, self.value_proj):
                proj.bias = mx.zeros((dims,))
            self.out_proj.bias = mx.zeros((dims,))

    def __call__(
        self,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        key_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Computes the attention.

        Args:
            queries: Array of shape ``[batch_size, num_queries, dims]``.
            keys: Array of shape ``[batch_size, num_keys, dims]``.
            values: Array of shape ``[batch_size, num_keys, dims]``.
            key_mask: Optional boolean array of shape
                ``[batch_size, num_keys]``, ``True`` for the keys that can be
                attended to (the opposite of PyTorch's ``key_padding_mask``).
                Queries without any valid key get a zero attention output
                (so the result is the bias of ``out_proj``), as PyTorch with
                ``need_weights=False``.

        Returns:
            An array of shape ``[batch_size, num_queries, dims]``.
        """
        batch_size, num_queries, _ = queries.shape
        num_keys = keys.shape[1]
        heads = self.num_heads
        head_dim = self.dims // heads

        q = self.query_proj(queries).reshape(batch_size, num_queries, heads, -1)
        k = self.key_proj(keys).reshape(batch_size, num_keys, heads, -1)
        v = self.value_proj(values).reshape(batch_size, num_keys, heads, -1)
        q, k, v = (t.transpose(0, 2, 1, 3) for t in (q, k, v))

        scores = (q * (1.0 / math.sqrt(head_dim))) @ k.transpose(0, 1, 3, 2)
        if key_mask is not None:
            # Queries without any valid key attend to nothing (zero output
            # before ``out_proj``), as PyTorch's fused attention path.
            has_key = key_mask.any(axis=-1)[:, None, None, None]
            keep = key_mask[:, None, None, :] | ~has_key
            scores = mx.where(keep, scores, mx.array(-mx.inf, scores.dtype))
        weights = mx.softmax(scores, axis=-1, precise=True)
        if key_mask is not None:
            weights = mx.where(has_key, weights, 0.0)
        if self.training and self.dropout > 0.0:
            keep = mx.random.bernoulli(1.0 - self.dropout, weights.shape)
            weights = mx.where(keep, weights / (1.0 - self.dropout), 0.0)

        out = (weights @ v).transpose(0, 2, 1, 3).reshape(batch_size, num_queries, -1)
        return self.out_proj(out)
