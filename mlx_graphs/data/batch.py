from typing import List, Callable
import mlx.core as mx

from mlx_graphs.data.data import GraphData
from mlx_graphs.data.collate import collate


class GraphDataBatch(GraphData):
    """
    Represents a batch data object describing a batch of graphs as one big (disconnected)
    graph.

    """

    def __init__(self, node_features, edge_index, cumsum, num_graphs, **kwargs) -> None:
        super().__init__(node_features=node_features, edge_index=edge_index, **kwargs)
        self.cumsum = cumsum
        self.num_graphs = num_graphs

    def __getitem__(self, idx):
        """Indexing to retrieve a specific graph from the batch

        Args:
            idx (int): the index of the graph to retrieve, must be in the range [0, num_graphs]

        Returns:
            GraphData: the graph associated to the specified index
        """

        lower_last_bound = self.cumsum[idx].item()
        upper_bound = self.cumsum[idx + 1].item()
        large_graph_dict = {
            attr: getattr(self, attr)
            for attr in self.to_dict()
            if attr not in {"cumsum", "num_graphs", "slices"}
        }
        single_graph_dict = {}
        for attr in large_graph_dict:
            if self.__inc__(attr):
                mask = (getattr(self, attr) >= lower_last_bound) & (
                    getattr(self, attr) <= upper_bound
                )
                mask = mask[0] & mask[1]
                indices = mx.array([i for i, e in enumerate(mask) if e])
                value = getattr(self, attr)[:, indices] - lower_last_bound
                single_graph_dict[attr] = value
            else:
                single_graph_dict[attr] = getattr(self, attr)[
                    lower_last_bound:upper_bound
                ]

        return GraphData(**single_graph_dict)


def batch(graphs: List[GraphData], collate_fn: Callable = collate) -> GraphDataBatch:
    """
    Constructs a :class:`mlx_graphs.batch.Batch` object from a
    list of :class:`~mlx_graphs.data.GraphData`
    """

    if not isinstance(graphs, (list, tuple)):
        graphs = list(graphs)

    global_dict = collate_fn(graphs)

    return GraphDataBatch(**global_dict)


def unbatch(batch: GraphDataBatch) -> List[GraphData]:
    """Reconstruct the list of :class:`~mlx_graphs.data.GraphData`
    objects from the :class:`~mlx_graphs.data.GraphDataBatch` object.
    """

    return [batch[idx] for idx in range(batch.num_graphs)]
