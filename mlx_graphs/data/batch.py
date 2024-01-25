from typing import List
import mlx.core as mx

from mlx_graphs.data.data import GraphData


class GraphDataBatch(GraphData):
    """
    Represents a batch data object describing a batch of graphs as one big (disconnected)
    graph.

    """

    def __init__(self, node_features, edge_index, cumsum) -> None:
        self.node_features = node_features
        self.edge_index = edge_index
        self.cumsum = cumsum

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

    num_nodes = mx.array([0] + [graph.num_nodes() for graph in graphs])
    cumsum = mx.cumsum(num_nodes)

    global_node_features = [graph.node_features for graph in graphs]
    global_node_features = mx.concatenate(global_node_features, axis=0)

    global_edge_index = [
        graph.edge_index + num_nodes[i] for i, graph in enumerate(graphs)
    ]
    global_edge_index = mx.concatenate(global_edge_index, axis=1)

    return GraphDataBatch(global_node_features, global_edge_index, cumsum=cumsum)
