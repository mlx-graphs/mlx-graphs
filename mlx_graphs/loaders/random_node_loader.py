import math

import numpy as np

from mlx_graphs.data import GraphData

from ._utils import Adjacency, index_array, slice_graph


class RandomNodeLoader:
    """Splits the nodes of a graph into ``num_parts`` random partitions and
    yields the subgraph induced by each partition, following PyG's
    ``RandomNodeLoader``.

    Useful to train on large graphs that do not fit in memory, when the loss of
    the edges between partitions is acceptable. With ``shuffle=True`` new
    random partitions are drawn at each epoch, otherwise the partitions are
    contiguous ranges of node ids.

    Each batch is a :class:`GraphData` with node- and edge-level attributes
    sliced to the subgraph, plus ``n_id`` (global id of every node) and
    ``e_id`` (original id of every edge).

    Args:
        data: The full graph.
        num_parts: Number of partitions (batches per epoch).
        shuffle: Whether to draw new random partitions at each epoch.
            Defaults to False

    Example:

    .. code-block:: python

        from mlx_graphs.loaders import RandomNodeLoader

        loader = RandomNodeLoader(data, num_parts=10, shuffle=True)
        for batch in loader:
            out = model(batch.node_features, batch.edge_index)
    """

    def __init__(self, data: GraphData, num_parts: int, shuffle: bool = False):
        if num_parts < 1:
            raise ValueError(f"`num_parts` must be positive (got {num_parts})")
        self.data = data
        self.num_parts = num_parts
        self.shuffle = shuffle
        self.num_nodes = data.num_nodes
        self.num_edges = data.num_edges
        self.batch_size = max(1, math.ceil(self.num_nodes / num_parts))
        self._adj = Adjacency(data.edge_index, self.num_nodes)
        self._perm = np.arange(self.num_nodes)
        self._current_index = 0

    def __len__(self) -> int:
        """Number of mini-batches per epoch."""
        return math.ceil(self.num_nodes / self.batch_size)

    def __iter__(self) -> "RandomNodeLoader":
        self._current_index = 0
        self._perm = (
            np.random.permutation(self.num_nodes)
            if self.shuffle
            else np.arange(self.num_nodes)
        )
        return self

    def __next__(self) -> GraphData:
        if self._current_index >= self.num_nodes:
            raise StopIteration
        end = self._current_index + self.batch_size
        n_id = self._perm[self._current_index : end].astype(np.int64)
        self._current_index = end

        edge_index, e_id = self._adj.induced_subgraph(n_id)
        return slice_graph(
            self.data,
            n_id,
            e_id,
            edge_index,
            num_nodes=self.num_nodes,
            num_edges=self.num_edges,
            n_id=index_array(n_id, self.num_nodes),
            e_id=index_array(e_id, self.num_edges),
        )
