from typing import Iterator

import mlx.core as mx
import numpy as np

from mlx_graphs.data.temporal import TemporalData


class TemporalDataLoader:
    r"""Iterates over the events of a :class:`~mlx_graphs.data.TemporalData`
    in their stored (time) order, merging ``batch_size`` successive events into
    a mini-batch, as PyG's ``torch_geometric.loader.TemporalDataLoader``.

    Every mini-batch is a :class:`~mlx_graphs.data.TemporalData` with an extra
    ``n_id`` attribute: the sorted unique nodes involved in the batch
    (sources, destinations and, if any, negative destinations).

    With ``neg_sampling_ratio > 0``, every mini-batch also gets
    ``round(neg_sampling_ratio * batch_size)`` negative destinations
    ``neg_dst``, drawn uniformly (with ``mx.random``, so reproducible with
    ``mx.random.seed``) between the smallest and the largest destination node
    of ``data``.

    Args:
        data: The events to iterate over
        batch_size: Number of events per mini-batch. Default: ``1``
        neg_sampling_ratio: Ratio of sampled negative destinations to positive
            destinations. Default: ``0.0``
        drop_last: Whether to drop the last mini-batch if it has fewer than
            ``batch_size`` events. Default: ``False``

    Example:

    .. code-block:: python

        from mlx_graphs.datasets import JODIEDataset
        from mlx_graphs.loaders import TemporalDataLoader

        data = JODIEDataset("wikipedia")[0]
        train_data, val_data, test_data = data.train_val_test_split()
        loader = TemporalDataLoader(train_data, batch_size=200, neg_sampling_ratio=1.0)
        for batch in loader:
            batch.src, batch.dst, batch.neg_dst, batch.n_id

    References:
        `PyG TemporalDataLoader <https://pytorch-geometric.readthedocs.io/en/
        latest/generated/torch_geometric.loader.TemporalDataLoader.html>`_
    """

    def __init__(
        self,
        data: TemporalData,
        batch_size: int = 1,
        neg_sampling_ratio: float = 0.0,
        drop_last: bool = False,
    ):
        if batch_size < 1:
            raise ValueError(f"`batch_size` must be positive (got {batch_size}).")
        self.data = data
        self.batch_size = batch_size
        self.events_per_batch = batch_size
        self.neg_sampling_ratio = neg_sampling_ratio
        self.drop_last = drop_last

        if neg_sampling_ratio > 0:
            self.min_dst = int(data.dst.min().item())
            self.max_dst = int(data.dst.max().item())

        num_events = len(data)
        if drop_last and num_events % batch_size != 0:
            self._starts = range(0, num_events - batch_size, batch_size)
        else:
            self._starts = range(0, num_events, batch_size)

    def __len__(self) -> int:
        return len(self._starts)

    def __iter__(self) -> Iterator[TemporalData]:
        for start in self._starts:
            yield self._make_batch(start)

    def _make_batch(self, start: int) -> TemporalData:
        batch = self.data[start : start + self.events_per_batch]
        n_ids = [np.array(batch.src), np.array(batch.dst)]

        if self.neg_sampling_ratio > 0:
            num_neg = round(self.neg_sampling_ratio * batch.dst.shape[0])
            batch.neg_dst = mx.random.randint(
                self.min_dst, self.max_dst + 1, (num_neg,), dtype=batch.dst.dtype
            )
            n_ids.append(np.array(batch.neg_dst))

        n_id = np.unique(np.concatenate(n_ids))
        batch.n_id = mx.array(n_id.astype(np.int32))
        return batch
