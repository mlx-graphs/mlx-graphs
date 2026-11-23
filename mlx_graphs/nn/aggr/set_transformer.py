from typing import Optional

import mlx.core as mx

from mlx_graphs.nn.aggr.base import Aggregation
from mlx_graphs.nn.aggr.utils import (
    DenseLayout,
    PoolingByMultiheadAttention,
    SetAttentionBlock,
)


def _check_2d(module: Aggregation, x: mx.array):
    if x.ndim != 2:
        raise ValueError(
            f"{type(module).__name__} requires two-dimensional inputs (got {x.ndim})"
        )


class SetTransformerAggregation(Aggregation):
    r"""Aggregation with a Set Transformer, from the `"Set Transformer: A
    Framework for Attention-based Permutation-Invariant Neural Networks"
    <https://arxiv.org/abs/1810.00825>`_ paper, as PyG's
    ``SetTransformerAggregation``.

    The elements of each set are encoded with ``num_encoder_blocks`` Set
    Attention Blocks (SAB), pooled onto ``num_seed_points`` learnable seeds by
    Pooling by Multihead Attention (PMA), and decoded with
    ``num_decoder_blocks`` SABs. The seeds of empty sets attend to nothing.

    Args:
        channels: Size of each input sample.
        num_seed_points: Number of seed points. Default: ``1``
        num_encoder_blocks: Number of SABs in the encoder. Default: ``1``
        num_decoder_blocks: Number of SABs in the decoder. Default: ``1``
        heads: Number of attention heads. Default: ``1``
        concat: If ``True``, the outputs of the seed points are concatenated
            (output size ``num_seed_points * channels``); otherwise they are
            averaged. Default: ``True``
        layer_norm: Whether to apply layer normalization. Default: ``False``
        dropout: Dropout probability of the attention weights. Default: ``0.0``

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn.aggr import SetTransformerAggregation

        aggr = SetTransformerAggregation(16, num_seed_points=2, heads=4)
        x = mx.random.normal((6, 16))
        index = mx.array([0, 0, 1, 1, 1, 2])
        >>> aggr(x, index).shape
        (3, 32)

    References:
        - `Lee et al., "Set Transformer: A Framework for Attention-based
          Permutation-Invariant Neural Networks", ICML 2019
          <https://arxiv.org/abs/1810.00825>`_
    """

    def __init__(
        self,
        channels: int,
        num_seed_points: int = 1,
        num_encoder_blocks: int = 1,
        num_decoder_blocks: int = 1,
        heads: int = 1,
        concat: bool = True,
        layer_norm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.channels = channels
        self.num_seed_points = num_seed_points
        self.heads = heads
        self.concat = concat
        self.encoders = [
            SetAttentionBlock(channels, heads, layer_norm, dropout)
            for _ in range(num_encoder_blocks)
        ]
        self.pma = PoolingByMultiheadAttention(
            channels, num_seed_points, heads, layer_norm, dropout
        )
        self.decoders = [
            SetAttentionBlock(channels, heads, layer_norm, dropout)
            for _ in range(num_decoder_blocks)
        ]

    def aggregate(
        self,
        x: mx.array,
        index: mx.array,
        dim_size: int,
        max_num_elements: Optional[int] = None,
        **kwargs,
    ) -> mx.array:
        """Aggregates ``x``.

        Args:
            x: Array of shape ``[num_elements, channels]``.
            index: Set of each element.
            dim_size: Number of sets.
            max_num_elements: Only the first ``max_num_elements`` elements of
                each set (in input order) are used. Default: ``None``
        """
        _check_2d(self, x)
        layout = DenseLayout(index, x.shape[0], dim_size, max_num_elements)
        x, mask = layout.to_dense(x), layout.mask
        for encoder in self.encoders:
            x = encoder(x, mask)
        x = self.pma(x, mask)
        for decoder in self.decoders:
            x = decoder(x)
        x = mx.nan_to_num(x)
        return x.reshape(x.shape[0], -1) if self.concat else x.mean(axis=1)


class GraphMultisetTransformer(Aggregation):
    r"""The Graph Multiset Transformer pooling operator from the `"Accurate
    Learning of Graph Representations with Graph Multiset Pooling"
    <https://arxiv.org/abs/2102.11533>`_ paper, as PyG's
    ``GraphMultisetTransformer``.

    The elements of each set (e.g. node embeddings computed by a GNN) are
    pooled onto ``k`` seed points by Pooling by Multihead Attention (PMA),
    processed by ``num_encoder_blocks`` Set Attention Blocks (SAB), and
    pooled again onto a single seed point.

    Args:
        channels: Size of each input sample.
        k: Number of seed points of the first PMA.
        num_encoder_blocks: Number of SABs between the two PMAs. Default: ``1``
        heads: Number of attention heads. Default: ``1``
        layer_norm: Whether to apply layer normalization. Default: ``False``
        dropout: Dropout probability of the attention weights. Default: ``0.0``

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn.aggr import GraphMultisetTransformer

        aggr = GraphMultisetTransformer(16, k=4, heads=2)
        x = mx.random.normal((6, 16))
        batch_indices = mx.array([0, 0, 1, 1, 1, 2])
        >>> aggr(x, batch_indices).shape
        (3, 16)

    References:
        - `Baek et al., "Accurate Learning of Graph Representations with Graph
          Multiset Pooling", ICLR 2021 <https://arxiv.org/abs/2102.11533>`_
    """

    def __init__(
        self,
        channels: int,
        k: int,
        num_encoder_blocks: int = 1,
        heads: int = 1,
        layer_norm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.channels = channels
        self.k = k
        self.heads = heads
        self.pma1 = PoolingByMultiheadAttention(channels, k, heads, layer_norm, dropout)
        self.encoders = [
            SetAttentionBlock(channels, heads, layer_norm, dropout)
            for _ in range(num_encoder_blocks)
        ]
        self.pma2 = PoolingByMultiheadAttention(channels, 1, heads, layer_norm, dropout)

    def aggregate(
        self,
        x: mx.array,
        index: mx.array,
        dim_size: int,
        max_num_elements: Optional[int] = None,
        **kwargs,
    ) -> mx.array:
        """Aggregates ``x``.

        Args:
            x: Array of shape ``[num_elements, channels]``.
            index: Set of each element.
            dim_size: Number of sets.
            max_num_elements: Only the first ``max_num_elements`` elements of
                each set (in input order) are used. Default: ``None``

        Returns:
            An array of shape ``[dim_size, channels]``.
        """
        _check_2d(self, x)
        layout = DenseLayout(index, x.shape[0], dim_size, max_num_elements)
        x = self.pma1(layout.to_dense(x), layout.mask)
        for encoder in self.encoders:
            x = encoder(x)
        return self.pma2(x)[:, 0]
