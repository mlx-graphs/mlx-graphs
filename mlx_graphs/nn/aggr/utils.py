from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_graphs.nn.attention.multihead import MultiheadAttention
from mlx_graphs.utils.transformations import _to_numpy


class DenseLayout:
    """Host-side layout of a set of elements grouped by ``index`` into a dense
    ``[num_sets, max_set_size, ...]`` array, as PyG's ``to_dense_batch``.

    Elements keep their input order within their set and ``index`` does not
    need to be sorted. Unlike :func:`~mlx_graphs.utils.to_dense_batch`, the
    layout can be reused to go back from the dense array to the elements
    (:meth:`from_dense`), with gathers only, so both directions are
    differentiable.

    Args:
        index: Set of each element, of shape ``[num_elements]``. ``None``
            means a single set.
        num_elements: Number of elements (needed when ``index`` is ``None``).
        dim_size: Number of sets. Default: ``index.max() + 1``
        max_num_elements: Maximum number of elements per set. Elements beyond
            it are dropped. Default: the size of the largest set.
    """

    def __init__(
        self,
        index: Optional[mx.array],
        num_elements: int,
        dim_size: Optional[int] = None,
        max_num_elements: Optional[int] = None,
    ):
        if index is None:
            batch = np.zeros(num_elements, dtype=np.int64)
        else:
            batch = _to_numpy(index).reshape(-1).astype(np.int64)
        if dim_size is None:
            dim_size = int(batch.max()) + 1 if batch.size > 0 else 1
        counts = np.bincount(batch, minlength=dim_size)[:dim_size]
        order = np.argsort(batch, kind="stable")
        starts = np.cumsum(counts) - counts
        local = np.empty(batch.size, dtype=np.int64)
        valid = batch[order] < dim_size
        local[order[valid]] = np.arange(int(valid.sum())) - starts[batch[order[valid]]]
        local[order[~valid]] = np.iinfo(np.int64).max  # dropped
        if max_num_elements is None:
            size = int(counts.max(initial=0))
        else:
            size = int(max_num_elements)

        keep = np.nonzero(local < size)[0]
        flat = batch[keep] * size + local[keep]
        slots = np.full(dim_size * size, batch.size, dtype=np.int64)
        slots[flat] = keep
        mask = np.zeros(dim_size * size, dtype=bool)
        mask[flat] = True

        self.num_sets = dim_size
        self.size = size
        self.num_elements = batch.size
        self.all_kept = keep.size == batch.size
        self._slots = mx.array(slots.astype(np.int32))
        self._flat = mx.array(flat.astype(np.int32))
        self.mask = mx.array(mask.reshape(dim_size, size))

    @classmethod
    def from_sizes(
        cls,
        index: Optional[mx.array],
        num_elements: int,
        num_sets: int,
        max_num_elements: int,
    ) -> "DenseLayout":
        """Builds the layout on the device, without reading ``index`` on the
        host: no GPU synchronization, and usable inside :func:`mx.compile`.

        The number of sets and the size of the largest set are not computed
        from ``index`` but given, e.g. the number of graphs of a batch and an
        upper bound of their number of nodes (``max_num_elements`` is the
        padded size of the dense array, so a tight bound saves memory).
        Elements of sets larger than ``max_num_elements`` (beyond the first
        ``max_num_elements`` of their set), or with an ``index`` outside
        ``[0, num_sets)``, are dropped: :meth:`to_dense` leaves them out and
        :meth:`from_dense` returns zeros for them.

        With ``max_num_elements`` equal to the size of the largest set, the
        layout is the one of ``DenseLayout(index, num_elements)``. A larger
        bound only adds padding, which masked attentions (e.g.
        :class:`~mlx_graphs.nn.GPSConv` with ``attn_type="multihead"``)
        ignore; the linear attentions of :class:`~mlx_graphs.nn.SGFormer`,
        :class:`~mlx_graphs.nn.Polynormer` and the performer attention count
        the padding rows in their normalizers (as in PyG, where the padding is
        set by the largest graph of the batch), so their outputs then change
        slightly.

        Args:
            index: Set of each element, of shape ``[num_elements]``. ``None``
                means a single set.
            num_elements: Number of elements.
            num_sets: Number of sets.
            max_num_elements: Size of the sets in the dense array.

        Example:

        .. code-block:: python

            batch_indices = mx.array([0, 0, 1, 1, 1])
            layout = DenseLayout.from_sizes(
                batch_indices, 5, num_sets=2, max_num_elements=3
            )
            >>> layout.mask
            array([[True, True, False],
                   [True, True, True]], dtype=bool)
        """
        n, size = int(num_elements), int(max_num_elements)
        num_sets = int(num_sets)
        if index is None:
            index = mx.zeros((n,), dtype=mx.int32)
        index = index.reshape(-1).astype(mx.int32)
        positions = mx.arange(n, dtype=mx.int32)
        # Rank of every element within its set (in input order): sort by
        # (set, position), then subtract the start of the set.
        valid = (index >= 0) & (index < num_sets)
        set_key = mx.where(valid, index, num_sets)  # invalid elements last
        key = set_key.astype(mx.int64) * max(n, 1) + positions.astype(mx.int64)
        order = mx.argsort(key).astype(mx.int32)
        counts = mx.zeros((num_sets + 1,), dtype=mx.int32).at[set_key].add(1)
        counts = counts[:num_sets]
        starts = mx.cumsum(counts) - counts
        sorted_set = mx.clip(index[order], 0, max(num_sets - 1, 0))
        local_sorted = positions - starts[sorted_set]
        local = mx.zeros((n,), dtype=mx.int32).at[order].add(local_sorted)

        keep = valid & (local < size)
        pad = num_sets * size  # one extra slot collecting the dropped elements
        flat = mx.where(keep, index * size + local, pad)
        slots = mx.zeros((pad + 1,), dtype=mx.int32).at[flat].add(positions - n) + n
        mask = mx.zeros((pad + 1,), dtype=mx.int32).at[flat].add(1)[:pad] > 0

        layout = cls.__new__(cls)
        layout.num_sets = num_sets
        layout.size = size
        layout.num_elements = n
        layout.all_kept = True  # dropped elements read the padding slot
        layout._slots = slots[:pad]
        layout._flat = flat
        layout._pad_slot = True
        layout.mask = mask.reshape(num_sets, size)
        return layout

    def to_dense(self, x: mx.array, fill_value: float = 0.0) -> mx.array:
        """Converts ``x`` of shape ``[num_elements, ...]`` to a dense array of
        shape ``[num_sets, size, ...]`` padded with ``fill_value``."""
        pad = mx.full((1, *x.shape[1:]), fill_value, dtype=x.dtype)
        dense = mx.concatenate([x, pad], axis=0)[self._slots]
        return dense.reshape(self.num_sets, self.size, *x.shape[1:])

    def from_dense(self, dense: mx.array) -> mx.array:
        """Gathers the rows of the (kept) elements of a dense array of shape
        ``[num_sets, size, ...]``, in input order."""
        if not self.all_kept:
            raise ValueError("Some elements were dropped by `max_num_elements`")
        flat = dense.reshape(self.num_sets * self.size, *dense.shape[2:])
        if getattr(self, "_pad_slot", False):
            # dropped elements of `from_sizes` read a zero row
            pad = mx.zeros((1, *flat.shape[1:]), dtype=flat.dtype)
            flat = mx.concatenate([flat, pad], axis=0)
        return flat[self._flat]


class MultiheadAttentionBlock(nn.Module):
    r"""The Multihead Attention Block (MAB) from the `"Set Transformer: A
    Framework for Attention-based Permutation-Invariant Neural Networks"
    <https://arxiv.org/abs/1810.00825>`_ paper, as in PyG.

    .. math::
        \mathrm{MAB}(\mathbf{x}, \mathbf{y}) &= \mathrm{LayerNorm}(\mathbf{h} +
        \mathrm{ReLU}(\mathbf{h}\mathbf{W})) \\
        \mathbf{h} &= \mathrm{LayerNorm}(\mathbf{x} +
        \mathrm{Multihead}(\mathbf{x}, \mathbf{y}, \mathbf{y}))

    Args:
        channels: Size of each input sample.
        heads: Number of attention heads. Default: ``1``
        layer_norm: Whether to apply layer normalization. Default: ``True``
        dropout: Dropout probability of the attention weights. Default: ``0.0``
    """

    def __init__(
        self,
        channels: int,
        heads: int = 1,
        layer_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.channels = channels
        self.heads = heads
        self.dropout = dropout
        self.attn = MultiheadAttention(channels, heads, dropout=dropout)
        self.lin = nn.Linear(channels, channels)
        self.layer_norm = layer_norm
        if layer_norm:
            self.layer_norm1 = nn.LayerNorm(channels)
            self.layer_norm2 = nn.LayerNorm(channels)

    def __call__(
        self,
        x: mx.array,
        y: mx.array,
        x_mask: Optional[mx.array] = None,
        y_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Attends from ``x`` (``[B, N, channels]``) to ``y`` (``[B, M,
        channels]``); ``x_mask`` and ``y_mask`` are ``True`` for real rows."""
        out = self.attn(x, y, y, key_mask=y_mask)
        if x_mask is not None:
            out = mx.where(x_mask[..., None], out, 0.0)
        out = out + x
        if self.layer_norm:
            out = self.layer_norm1(out)
        out = out + nn.relu(self.lin(out))
        if self.layer_norm:
            out = self.layer_norm2(out)
        return out


class SetAttentionBlock(nn.Module):
    r"""The Set Attention Block (SAB) from the `"Set Transformer"
    <https://arxiv.org/abs/1810.00825>`_ paper:
    :math:`\mathrm{SAB}(\mathbf{X}) = \mathrm{MAB}(\mathbf{x}, \mathbf{y})`.

    Args:
        channels: Size of each input sample.
        heads: Number of attention heads. Default: ``1``
        layer_norm: Whether to apply layer normalization. Default: ``True``
        dropout: Dropout probability of the attention weights. Default: ``0.0``
    """

    def __init__(
        self,
        channels: int,
        heads: int = 1,
        layer_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.mab = MultiheadAttentionBlock(channels, heads, layer_norm, dropout)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        return self.mab(x, x, mask, mask)


class InducedSetAttentionBlock(nn.Module):
    r"""The Induced Set Attention Block (ISAB) from the `"Set Transformer"
    <https://arxiv.org/abs/1810.00825>`_ paper:
    :math:`\mathrm{ISAB}(\mathbf{X}) = \mathrm{MAB}(\mathbf{x}, \mathbf{h})`
    with :math:`\mathbf{h} = \mathrm{MAB}(\mathbf{I}, \mathbf{x})` for
    learnable induced points :math:`\mathbf{I}`.

    Args:
        channels: Size of each input sample.
        num_induced_points: Number of induced points.
        heads: Number of attention heads. Default: ``1``
        layer_norm: Whether to apply layer normalization. Default: ``True``
        dropout: Dropout probability of the attention weights. Default: ``0.0``
    """

    def __init__(
        self,
        channels: int,
        num_induced_points: int,
        heads: int = 1,
        layer_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.ind = _xavier((1, num_induced_points, channels))
        self.mab1 = MultiheadAttentionBlock(channels, heads, layer_norm, dropout)
        self.mab2 = MultiheadAttentionBlock(channels, heads, layer_norm, dropout)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        ind = mx.broadcast_to(self.ind, (x.shape[0], *self.ind.shape[1:]))
        h = self.mab1(ind, x, y_mask=mask)
        return self.mab2(x, h, x_mask=mask)


class PoolingByMultiheadAttention(nn.Module):
    r"""The Pooling by Multihead Attention (PMA) layer from the `"Set
    Transformer" <https://arxiv.org/abs/1810.00825>`_ paper:
    :math:`\mathrm{PMA}(\mathbf{X}) = \mathrm{MAB}(\mathbf{S},
    \mathrm{ReLU}(\mathbf{X}\mathbf{W}))` for learnable seed points
    :math:`\mathbf{S}`.

    Args:
        channels: Size of each input sample.
        num_seed_points: Number of seed points. Default: ``1``
        heads: Number of attention heads. Default: ``1``
        layer_norm: Whether to apply layer normalization. Default: ``True``
        dropout: Dropout probability of the attention weights. Default: ``0.0``
    """

    def __init__(
        self,
        channels: int,
        num_seed_points: int = 1,
        heads: int = 1,
        layer_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.lin = nn.Linear(channels, channels)
        self.seed = _xavier((1, num_seed_points, channels))
        self.mab = MultiheadAttentionBlock(channels, heads, layer_norm, dropout)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        x = nn.relu(self.lin(x))
        seed = mx.broadcast_to(self.seed, (x.shape[0], *self.seed.shape[1:]))
        return self.mab(seed, x, y_mask=mask)


def _xavier(shape: tuple) -> mx.array:
    # torch.nn.init.xavier_uniform_ on a [1, n, c] tensor: fan_in = n * c,
    # fan_out = c (receptive field n).
    fan_in, fan_out = shape[1] * shape[2], shape[0] * shape[2]
    limit = float(np.sqrt(6.0 / (fan_in + fan_out)))
    return mx.random.uniform(-limit, limit, shape)
