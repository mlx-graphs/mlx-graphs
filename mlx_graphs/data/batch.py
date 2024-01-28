from typing import List, Union
import mlx.core as mx

from mlx_graphs.data.data import GraphData
from mlx_graphs.data.collate import collate


class GraphDataBatch(GraphData):
    def __init__(self, graphs, **kwargs) -> None:
        if not isinstance(graphs, (list, tuple)):
            graphs = list(graphs)

        batch_kwargs = collate(graphs)
        super().__init__(_num_graphs=len(graphs), **batch_kwargs, **kwargs)

    @property
    def num_graphs(self):
        return self._num_graphs

    def __getitem__(self, idx: Union[int, slice]) -> Union[GraphData, list[GraphData]]:
        if isinstance(idx, int):
            idx = self._handle_neg_index_if_needed(idx)
            return self._get_graph(idx)

        elif isinstance(idx, slice):
            start, stop, step = (
                idx.start or 0,
                idx.stop or self._num_graphs - 1,
                idx.step or 1,
            )
            start = self._handle_neg_index_if_needed(start)
            stop = self._handle_neg_index_if_needed(stop)
            if stop < start:
                raise IndexError(
                    "Batch slicing out of range (stop is greater than start)."
                )

            return [self._get_graph(i) for i in range(start, stop, step)]

        raise TypeError("GraphDataBatch indices should be int or slice.")

    def _get_graph(self, idx: int):
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
            if self.to_dict()[f"_inc_{attr}"]:
                original_value -= self._cumsum[idx]

            # NOTE: `self._cumsum` is the cumsum of number of nodes per graph. For custom features
            # with __inc__, each feature should have its own cumsum

            single_graph_dict[attr] = original_value

        return GraphData(**single_graph_dict)

    def _get_attr_slice_at_index(self, attr: str, idx: int):
        attr_sizes = self.to_dict()[f"_size_{attr}"]
        cum_attr_counts = mx.cumsum(mx.concatenate([mx.array([0]), attr_sizes]))

        from_idx = cum_attr_counts[idx].item()
        upto_idx = cum_attr_counts[idx + 1].item()

        return from_idx, upto_idx

    def _handle_neg_index_if_needed(self, index: int):
        if index < 0:
            index = self.num_graphs + index
        if index < 0 or index >= self.num_graphs:
            raise IndexError("Batch indexing out of range.")
        return index


def batch(graphs: List[GraphData]) -> GraphDataBatch:
    """
    Constructs a :class:`mlx_graphs.batch.Batch` object from a
    list of :class:`~mlx_graphs.data.GraphData`
    """
    return GraphDataBatch(graphs)


def unbatch(batch: GraphDataBatch) -> List[GraphData]:
    """Reconstruct the list of :class:`~mlx_graphs.data.GraphData`
    objects from the :class:`~mlx_graphs.data.GraphDataBatch` object.
    """

    return [batch[idx] for idx in range(batch.num_graphs)]
