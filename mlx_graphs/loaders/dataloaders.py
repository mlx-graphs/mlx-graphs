from typing import Sequence, Union

import numpy as np

from mlx_graphs.data import GraphData, GraphDataBatch, batch
from mlx_graphs.datasets.dataset import Dataset


class Dataloader:
    """
    Default data loader to batch and iterate over multiple graphs.

    Args:
        dataset: `Dataset` or list of `GraphData` to batch and iterate over
        batch_size: Number of graphs to load per batch. Defaults to 1
        shuffle: Whether to reshuffle the order of graphs within each batch.
            Defaults to False
    """

    def __init__(
        self,
        dataset: Union[Dataset, Sequence[GraphData]],
        batch_size: int = 1,
        shuffle: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        self._indices = list(range(len(dataset)))
        self._current_index = 0

    def _shuffle_indices(self):
        np.random.shuffle(self._indices)

    def __iter__(self):
        return self

    def __next__(self) -> GraphDataBatch:
        """
        Get the next batch of data.

        Returns:
            A batch of graphs.
        """
        if self._current_index >= len(self.dataset):
            self._current_index = 0
            if self.shuffle:
                self._shuffle_indices()
            raise StopIteration

        batch_indices = self._indices[
            self._current_index : self._current_index + self.batch_size
        ]
        batched_data = batch(
            [
                self.dataset[i]  # type: ignore - this is a list[GraphData]
                for i in batch_indices
            ]
        )
        self._current_index += self.batch_size
        return batched_data
