from typing import Union

import mlx.core as mx
import numpy as np

from mlx_graphs.data.data import GraphData
from mlx_graphs.data.collate import collate


class GraphDataBatch(GraphData):
    """Concatenates multiple `GraphData` into a single unified `GraphDataBatch`
    for efficient computation and parallelization over multiple graphs.

    All graphs remain disconnected in the batch, meaning that any pairs of graphs
    have no nodes or edges in common.
    `GraphDataBatch` can be especially used to speed up graph classification tasks, where
    multiple graphs can easily fit into memory and be processed in parallel.

    Args:
        graphs: List of `GraphData` objects to batch together

    Example:

    .. code-block:: python

        graphs = [
            GraphData(
                edge_index=mx.array([[0, 0, 0], [1, 1, 1]]),
                node_features=mx.zeros((3, 1)),
            ),
            GraphData(
                edge_index=mx.array([[1, 1, 1], [2, 2, 2]]),
                node_features=mx.ones((3, 1)),
            ),
            GraphData(
                edge_index=mx.array([[3, 3, 3], [4, 4, 4]]),
                node_features=mx.ones((3, 1)) * 2,
            )
        ]
        batch = GraphDataBatch(graphs)

        >>> batch.num_graphs
        3

        >>> batch
        GraphDataBatch(
            edge_index=array([[0, 0, 0, 4, 4, 4, 9, 9, 9], [1, 1, 1, 5, 5, 5, 10, 10, 10]], dtype=int32),
            node_features=array([[0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [2.0], [2.0], [2.0]], dtype=float32))

        >>> batch[1]
        GraphData(
            edge_index=array([[1, 1, 1], [2, 2, 2]], dtype=int32),
            node_features=array([[1], [1], [1]], dtype=float32))

        >>> batch[1:]
        [
            GraphData(
                edge_index=array([[1, 1, 1], [2, 2, 2]], dtype=int32),
                node_features=array([[1], [1], [1]], dtype=float32)),
            GraphData(
                edge_index=array([[3, 3, 3], [4, 4, 4]], dtype=int32),
                node_features=array([[2], [2], [2]], dtype=float32))
        ]
    """

    def __init__(self, graphs: list[GraphData], **kwargs) -> None:
        batch_kwargs = collate(graphs)
        super().__init__(_num_graphs=len(graphs), **batch_kwargs, **kwargs)

    @property
    def num_graphs(self):
        """Number of graphs in the batch."""
        return self._num_graphs  # provided via collate

    @property
    def batch_indices(self):
        """Mask indicating for each node its corresponding batch index."""
        return self._batch_indices  # provided via collate

    def __getitem__(self, idx: Union[int, slice]) -> Union[GraphData, list[GraphData]]:
        if isinstance(idx, int):
            idx = self._handle_neg_index_if_needed(idx)
            return self._get_graph(idx)

        elif isinstance(idx, slice):
            start, stop, step = (
                idx.start or 0,
                idx.stop or self._num_graphs,
                idx.step or 1,
            )
            start = self._handle_neg_index_if_needed(start)
            stop = self._handle_neg_index_if_needed(stop)
            if stop < start:
                raise IndexError(
                    "Batch slicing out of range (stop is greater than start)."
                )

            return [self._get_graph(i) for i in range(start, stop, step)]

        elif isinstance(idx, (mx.array, list)):
            if isinstance(idx, list):
                idx = mx.array(idx)

            if idx.ndim != 1:
                raise ValueError(
                    "Batch indexing with mx.array only supports 1D index array."
                )

            idx = self._handle_neg_index_if_needed(idx)
            return [self._get_graph(i) for i in idx]

        raise TypeError("GraphDataBatch indices should be int or slice.")

    def _get_graph(self, idx: int) -> GraphData:
        """Returns a `GraphData` from the batch with its original attributes and edge index."""
        large_graph_dict = {
            attr: getattr(self, attr)
            for attr in self.to_dict()
            if not attr.startswith("_")
        }

        single_graph_dict = {}
        for attr in large_graph_dict:
            from_idx, upto_idx = self._get_attr_slice_at_index(attr, idx)
            attr_array = getattr(self, attr)

            # Using mx.take_along_axis required indices with same number of dimensions as the values
            # Singleton dimensions are thus added to all dimensions except the __concat__ dimension
            gather_dim = self.to_dict()[f"_cat_dim_{attr}"]
            broadcast_dim = [
                -1 if dim == gather_dim else 1 for dim in range(attr_array.ndim)
            ]
            range_at_dim = mx.arange(from_idx, upto_idx).reshape(broadcast_dim)

            # Gather values at the current range on the __concat__ dimension
            original_value = mx.take_along_axis(attr_array, range_at_dim, gather_dim)

            # If __inc__ is True, we get back the original values by subtracting
            # the cumulative sum at the batch index
            if f"_inc_{attr}" in self.to_dict():
                original_value -= self.to_dict()[f"_cumsum_{attr}"][idx]

            single_graph_dict[attr] = original_value

        return GraphData(**single_graph_dict)

    def _get_attr_slice_at_index(self, attr: str, idx: int):
        """Returns the starting and ending index to retrieve the original attribute `attr`
        from the batch, at the given index `idx`.
        """
        attr_sizes = self.to_dict()[f"_size_{attr}"]
        cum_attr_counts = mx.cumsum(mx.concatenate([mx.array([0]), attr_sizes]))

        from_idx = cum_attr_counts[idx].item()
        upto_idx = cum_attr_counts[idx + 1].item()

        return from_idx, upto_idx

    def _handle_neg_index_if_needed(self, index: Union[mx.array, list, int]):
        """Returns the corresponsing positive index or indices of negative batch indices.
        Raises an index error if indices are incorrect. This method accepts either an mx.array,
        a list or a single int index.
        """
        if isinstance(index, int):
            index = mx.array([index])
        elif isinstance(index, mx.array):
            index = index.tolist()

        # Should be changed once boolean indexing exists in mlx
        index = np.asarray(index)
        index[index < 0] += self.num_graphs

        if np.any((index < 0) | (index > self.num_graphs)):
            raise IndexError("Batch indexing out of range.")

        index = index.tolist()
        return index[0] if len(index) == 1 else index


def batch(graphs: list[GraphData]) -> GraphDataBatch:
    """Constructs a `GraphDataBatch` object from a list of `GraphData`.

    Args:
        batch: List of `GraphData` to merge into a single batch

    Returns:
        `GraphDataBatch` storing a large batched graph
    """
    return GraphDataBatch(graphs)


def unbatch(batch: GraphDataBatch) -> list[GraphData]:
    """Reconstructs a list of `GraphData` objects from a `GraphDataBatch`.

    Args:
        batch: `GraphDataBatch` to unbatch

    Returns:
        List of unbatched `GraphData`
    """
    return [batch[idx] for idx in range(batch.num_graphs)]
