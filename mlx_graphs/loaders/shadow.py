from typing import Optional, Union

import mlx.core as mx
import numpy as np

from mlx_graphs.data import GraphData

from ._utils import (
    Adjacency,
    index_array,
    sample_neighbors_grouped,
    segment_arange,
    slice_graph,
    to_index,
)


class ShaDowKHopSampler:
    """The ShaDow :math:`k`-hop sampler from the `"Decoupling the Depth and
    Scope of Graph Neural Networks" <https://arxiv.org/abs/2201.07858>`_
    paper, following PyG's ``ShaDowKHopSampler``.

    For every seed (root) node, ``num_neighbors`` incoming neighbors are
    sampled per node and per hop, for ``depth`` hops, and the subgraph
    *induced* by the sampled nodes is extracted. The subgraphs of the
    ``batch_size`` roots of a mini-batch are kept separate (a node may appear
    in several of them) and returned as one block-diagonal
    :class:`GraphData`:

    - ``edge_index``, node- and edge-level attributes of all subgraphs, with
      the nodes of each subgraph sorted by global id;
    - ``batch_indices``: subgraph of every node (PyG's ``batch``), to pool one
      representation per subgraph;
    - ``ptr``: start of every subgraph in the node list, ``[batch_size + 1]``;
    - ``root_n_id``: position of every root node in the node list;
    - ``n_id`` / ``e_id``: global node ids and original edge ids;
    - ``node_labels``: like PyG's ``y``, the labels of the **roots** only
      (``[batch_size, ...]``), when ``data.node_labels`` is node-level.

    Args:
        data: The full graph.
        depth: Number of hops of each subgraph.
        num_neighbors: Number of neighbors sampled per node and per hop.
            ``-1`` keeps all neighbors.
        node_idx: Root nodes to iterate over, as indices or a boolean mask.
            Defaults to None (all nodes)
        replace: Whether to sample neighbors with replacement.
            Defaults to False
        batch_size: Number of roots (subgraphs) per mini-batch. Defaults to 1
        shuffle: Whether to shuffle the roots at each epoch. Defaults to False

    Example:

    .. code-block:: python

        from mlx_graphs.loaders import ShaDowKHopSampler

        loader = ShaDowKHopSampler(
            data, depth=2, num_neighbors=5, node_idx=data.train_mask, batch_size=64
        )
        for batch in loader:
            h = model(batch.node_features, batch.edge_index)
            root_h = h[batch.root_n_id]
            loss = loss_fn(classifier(root_h), batch.node_labels)
    """

    def __init__(
        self,
        data: GraphData,
        depth: int,
        num_neighbors: int,
        node_idx: Optional[Union[mx.array, np.ndarray]] = None,
        replace: bool = False,
        batch_size: int = 1,
        shuffle: bool = False,
    ):
        if batch_size < 1:
            raise ValueError(f"`batch_size` must be positive (got {batch_size})")
        self.data = data
        self.depth = depth
        self.num_neighbors = num_neighbors
        self.replace = replace
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.num_nodes = data.num_nodes
        self.num_edges = data.num_edges
        self.node_idx = to_index(node_idx, self.num_nodes)
        self._adj = Adjacency(data.edge_index, self.num_nodes)
        self._indices = np.arange(self.node_idx.size)
        self._current_index = 0

    def __len__(self) -> int:
        """Number of mini-batches per epoch."""
        return (self.node_idx.size + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> "ShaDowKHopSampler":
        self._current_index = 0
        if self.shuffle:
            np.random.shuffle(self._indices)
        return self

    def __next__(self) -> GraphData:
        if self._current_index >= self._indices.size:
            raise StopIteration
        end = self._current_index + self.batch_size
        roots = self.node_idx[self._indices[self._current_index : end]]
        self._current_index = end
        return self.sample(roots)

    def sample(self, roots: np.ndarray) -> GraphData:
        """Builds the batch of the subgraphs around ``roots`` (global ids)."""
        adj, N = self._adj, self.num_nodes
        num_roots = roots.size

        # Every (subgraph, node) pair is encoded as `subgraph * N + node`.
        keys = np.arange(num_roots, dtype=np.int64) * N + roots
        visited = np.unique(keys)
        frontier = keys
        for _ in range(self.depth):
            if frontier.size == 0:
                break
            owner, nbr = sample_neighbors_grouped(
                adj.colptr,
                adj.csc_row,
                frontier % N,
                self.num_neighbors,
                self.replace,
            )
            candidates = np.unique((frontier[owner] // N) * N + nbr)
            frontier = np.setdiff1d(candidates, visited, assume_unique=True)
            visited = np.union1d(visited, frontier)

        # `visited` is sorted: subgraph by subgraph, nodes by global id.
        n_id = visited % N
        batch = visited // N
        ptr = np.zeros(num_roots + 1, dtype=np.int64)
        np.cumsum(np.bincount(batch, minlength=num_roots), out=ptr[1:])
        root_n_id = np.searchsorted(visited, keys)

        # Induced subgraph of every subgraph: out-edges whose target is in it.
        starts = adj.rowptr[n_id]
        counts = adj.rowptr[n_id + 1] - starts
        pos = segment_arange(starts, counts)
        src_local = np.repeat(np.arange(n_id.size), counts)
        dst_keys = np.repeat(batch, counts) * N + adj.csr_col[pos]
        dst_local = np.searchsorted(visited, dst_keys)
        dst_local_clipped = np.minimum(dst_local, max(visited.size - 1, 0))
        keep = visited[dst_local_clipped] == dst_keys if visited.size else dst_keys[:0]
        edge_index = np.stack([src_local[keep], dst_local[keep]])
        e_id = adj.csr_perm[pos[keep]]

        out = slice_graph(
            self.data,
            n_id,
            e_id,
            edge_index,
            num_nodes=self.num_nodes,
            num_edges=self.num_edges,
            n_id=index_array(n_id, N),
            e_id=index_array(e_id, self.num_edges),
            # positions in the batch, at most num_roots * N
            batch_indices=index_array(batch, num_roots),
            ptr=index_array(ptr, visited.size + 1),
            root_n_id=index_array(root_n_id, visited.size),
        )
        labels = self.data.node_labels
        if labels is not None and labels.ndim >= 1 and labels.shape[0] == N:
            out.node_labels = labels[mx.array(roots.astype(np.int64))]
        return out
