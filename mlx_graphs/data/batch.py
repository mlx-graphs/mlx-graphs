from typing import List
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
        node_features = self.node_features[lower_last_bound:upper_bound]

        mask = (self.edge_index >= lower_last_bound) & (self.edge_index <= upper_bound)

        mask = mask[0] & mask[1]

        # NOTE : since boolean indexing isn't yet available we need to deduce the indices
        indices = mx.array([i for i, e in enumerate(mask) if e])

        # undo the increment induced by the batching
        edge_index = self.edge_index[:, indices] - lower_last_bound

        return GraphData(
            node_features=node_features,
            edge_index=edge_index,
        )


def batch(graphs: List[GraphData]) -> GraphDataBatch:
    """
    Constructs a :class:`mlx_graphs.batch.Batch` object from a
    list of :class:`~mlx_graphs.data.GraphData`
    """

    if not isinstance(graphs, (list, tuple)):
        graphs = list(graphs)

    global_dict = collate(graphs)

    return GraphDataBatch(**global_dict)


def unbatch(batch: GraphDataBatch) -> List[GraphData]:
    """Reconstruct the list of :class:`~mlx_graphs.data.GraphData`
    objects from the :class:`~mlx_graphs.data.GraphDataBatch` object.
    """

    return [batch[idx] for idx in range(batch.num_graphs)]
