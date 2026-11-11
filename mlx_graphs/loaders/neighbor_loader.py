from typing import Any, Optional, Union

import mlx.core as mx
import numpy as np
from mlx_cluster import neighbor_sample

from mlx_graphs.data import GraphData, HeteroGraphData

from .hetero_neighbor_loader import EdgeType, _HeteroNeighborLoader


class _HomogeneousNeighborLoader:
    """Homogeneous implementation behind :class:`NeighborLoader`.

    Mini-batch loader for node-level tasks on a single large graph.

    Uses neighbor sampling to extract K-hop subgraphs around seed nodes.
    Each iteration yields a :class:`GraphData` representing the sampled
    subgraph with locally re-indexed edges.

    The first ``batch_size`` nodes in the returned subgraph are always the
    seed nodes.  During training, compute loss only on these nodes::

        out = model(batch.node_features, batch.edge_index)
        loss = loss_fn(out[: batch.batch_size], batch.node_labels[: batch.batch_size])

    Args:
        data: A single :class:`GraphData` representing the full graph.
        num_neighbors: Number of neighbors to sample per hop.
            Length determines the number of hops.
            Use ``-1`` to sample all neighbors at a given hop.
            E.g. ``[10, 5]`` samples 10 neighbors at hop-1, 5 at hop-2.
        input_nodes: 1-D array of seed node indices to iterate over.
            Typically the indices where ``train_mask`` is ``True``.
            If ``None``, all nodes are used.
        batch_size: Number of seed nodes per mini-batch.
        shuffle: Whether to shuffle ``input_nodes`` at the start of each epoch.
        replace: Whether to sample neighbors with replacement.
        directed: Whether to only include edges in the sampling direction.
    """

    def __init__(
        self,
        data: GraphData,
        num_neighbors: list[int],
        input_nodes: Optional[mx.array] = None,
        batch_size: int = 1,
        shuffle: bool = False,
        replace: bool = False,
        directed: bool = True,
    ):
        self.data = data
        self.num_neighbors = num_neighbors
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.replace = replace
        self.directed = directed

        if input_nodes is None:
            self.input_nodes = mx.arange(data.num_nodes, dtype=mx.int64)
        else:
            self.input_nodes = input_nodes.astype(mx.int64)

        # Pre-compute CSC format once and evaluate for C++ kernel access
        self._colptr, self._row, self._edge_perm = self._to_csc(
            data.edge_index, data.num_nodes
        )
        mx.eval(self._colptr, self._row)

        # Iteration state
        self._indices = np.arange(len(self.input_nodes))
        self._current_index = 0

    @staticmethod
    def _to_csc(
        edge_index: mx.array, num_nodes: int
    ) -> tuple[mx.array, mx.array, np.ndarray]:
        """Convert edge_index ``[2, E]`` to CSC format.

        Returns:
            Tuple of ``(colptr, row, edge_perm)`` where ``edge_perm`` is
            the argsort permutation used to sort edges by target, needed
            to map sampled edge indices back to the original ordering.
        """
        sources = np.array(edge_index[0], copy=False).astype(np.int64)
        targets = np.array(edge_index[1], copy=False).astype(np.int64)

        order = np.argsort(targets, kind="stable")
        sorted_sources = sources[order]

        counts = np.bincount(targets, minlength=num_nodes)
        colptr = np.zeros(num_nodes + 1, dtype=np.int64)
        np.cumsum(counts, out=colptr[1:])

        return (
            mx.array(colptr),
            mx.array(sorted_sources, dtype=mx.int64),
            order,
        )

    def __len__(self) -> int:
        """Number of mini-batches per epoch."""
        return (len(self.input_nodes) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> "_HomogeneousNeighborLoader":
        self._current_index = 0
        if self.shuffle:
            np.random.shuffle(self._indices)
        return self

    def __next__(self) -> GraphData:
        if self._current_index >= len(self.input_nodes):
            raise StopIteration

        # 1. Select seed nodes
        end = min(self._current_index + self.batch_size, len(self.input_nodes))
        batch_indices = self._indices[self._current_index : end]
        seed_nodes = self.input_nodes[mx.array(batch_indices)]
        num_seeds = len(batch_indices)
        self._current_index = end

        # 2. Sample K-hop neighborhood
        #    Arrays must be evaluated before passing to C++ kernel
        seed_nodes_i64 = seed_nodes.astype(mx.int64)
        mx.eval(seed_nodes_i64)
        samples, rows, cols, edges = neighbor_sample(
            self._colptr,
            self._row,
            seed_nodes_i64,
            self.num_neighbors,
            self.replace,
            self.directed,
        )

        # 3. Build re-indexed edge_index (rows/cols are already local)
        if len(rows) > 0:
            edge_index = mx.stack([rows, cols], axis=0)
        else:
            edge_index = mx.zeros((2, 0), dtype=mx.int64)

        # 4. Slice standard attributes using global node IDs
        n_id = samples

        node_features = None
        if self.data.node_features is not None:
            node_features = self.data.node_features[n_id]

        node_labels = None
        if self.data.node_labels is not None:
            node_labels = self.data.node_labels[n_id]

        edge_features = None
        if self.data.edge_features is not None and len(edges) > 0:
            original_edge_ids = mx.array(self._edge_perm[np.array(edges, copy=False)])
            edge_features = self.data.edge_features[original_edge_ids]

        # 5. Collect custom per-node attributes (e.g. train_mask)
        kwargs = {
            "batch_size": num_seeds,
            "n_id": n_id,
        }
        _skip = {
            "edge_index",
            "node_features",
            "edge_features",
            "graph_features",
            "node_labels",
            "edge_labels",
            "graph_labels",
        }
        num_nodes = self.data.num_nodes
        for attr_name, val in vars(self.data).items():
            if attr_name.startswith("_") or attr_name in _skip:
                continue
            if (
                isinstance(val, mx.array)
                and val.ndim >= 1
                and val.shape[0] == num_nodes
            ):
                kwargs[attr_name] = val[n_id]

        return GraphData(
            edge_index=edge_index,
            node_features=node_features,
            edge_features=edge_features,
            graph_features=self.data.graph_features,
            node_labels=node_labels,
            **kwargs,
        )


class NeighborLoader:
    """Mini-batch loader for node-level tasks on a single large graph.

    Uses neighbor sampling to extract K-hop subgraphs around seed nodes.
    Works on both homogeneous and heterogeneous graphs, following PyG's API:
    the implementation is picked from the type of ``data``.

    - :class:`GraphData`: each iteration yields a :class:`GraphData`
      holding the sampled subgraph with locally re-indexed edges.
    - :class:`HeteroGraphData`: each iteration yields a
      :class:`HeteroGraphData` with edges re-indexed locally per node type
      (sampled with ``mlx_cluster.hetero_neighbor_sample``).

    The first ``batch_size`` nodes (of the seed node type, for heterogeneous
    graphs) are always the seed nodes. During training, compute the loss
    only on these nodes::

        # homogeneous
        out = model(batch.node_features, batch.edge_index)
        loss = loss_fn(out[: batch.batch_size], batch.node_labels[: batch.batch_size])

        # heterogeneous
        out = model(batch.node_features_dict, batch.edge_index_dict)
        n = batch.batch_size
        loss = loss_fn(out["author"][:n], batch.node_labels_dict["author"][:n])

    Args:
        data: The full graph, a :class:`GraphData` or a
            :class:`HeteroGraphData`.
        num_neighbors: Number of neighbors to sample per hop. Its length
            determines the number of hops; ``-1`` samples all neighbors.
            E.g. ``[10, 5]`` samples 10 neighbors at hop-1, 5 at hop-2.
            For heterogeneous graphs this can also be a dict mapping each
            edge type to its own list.
        input_nodes: The seed nodes to iterate over.

            - Homogeneous: a 1-D array of node indices, or ``None`` for all
              nodes.
            - Heterogeneous (required): the seed node type and which of its
              nodes to use: ``"author"`` (all authors), ``("author", indices)``
              or ``("author", boolean_mask)``.
        batch_size: Number of seed nodes per mini-batch.
        shuffle: Whether to shuffle the seed nodes at the start of each epoch.
        replace: Whether to sample neighbors with replacement.
        directed: Whether to only include edges in the sampling direction.
            Only ``True`` is supported for heterogeneous graphs.

    Example:

    .. code-block:: python

        from mlx_graphs.datasets import DBLP
        from mlx_graphs.loaders import NeighborLoader

        data = DBLP()[0]
        loader = NeighborLoader(
            data,
            num_neighbors=[10, 5],
            input_nodes=("author", data.author_train_mask),
            batch_size=128,
            shuffle=True,
        )
        batch = next(iter(loader))
    """

    def __init__(
        self,
        data: Union[GraphData, HeteroGraphData],
        num_neighbors: Union[list[int], dict[EdgeType, list[int]]],
        input_nodes: Union[None, mx.array, str, tuple[str, Optional[mx.array]]] = None,
        batch_size: int = 1,
        shuffle: bool = False,
        replace: bool = False,
        directed: bool = True,
    ):
        if isinstance(data, HeteroGraphData):
            if input_nodes is None or isinstance(input_nodes, mx.array):
                raise ValueError(
                    "For a HeteroGraphData, `input_nodes` must name the seed "
                    "node type: 'author', ('author', indices) or "
                    "('author', boolean_mask)."
                )
            if not directed:
                raise ValueError(
                    "`directed=False` is not supported for heterogeneous graphs."
                )
            self._loader = _HeteroNeighborLoader(
                data,
                num_neighbors,
                input_nodes,
                batch_size=batch_size,
                shuffle=shuffle,
                replace=replace,
            )
        elif isinstance(data, GraphData):
            if isinstance(num_neighbors, dict):
                raise ValueError(
                    "A dict of `num_neighbors` per edge type is only supported "
                    "for heterogeneous graphs; pass a list for a GraphData."
                )
            if isinstance(input_nodes, (str, tuple)):
                raise ValueError(
                    "Node types in `input_nodes` are only supported for "
                    "heterogeneous graphs; pass an array of node indices "
                    "for a GraphData."
                )
            self._loader = _HomogeneousNeighborLoader(
                data,
                num_neighbors,
                input_nodes,
                batch_size=batch_size,
                shuffle=shuffle,
                replace=replace,
                directed=directed,
            )
        else:
            raise TypeError(
                "`data` must be a GraphData or a HeteroGraphData, "
                f"got {type(data).__name__}."
            )

    def __len__(self) -> int:
        """Number of mini-batches per epoch."""
        return len(self._loader)

    def __iter__(self) -> "NeighborLoader":
        iter(self._loader)
        return self

    def __next__(self) -> Union[GraphData, HeteroGraphData]:
        return next(self._loader)

    def __getattr__(self, name: str) -> Any:
        # Expose the implementation's attributes (data, input_nodes,
        # input_type, ...). Only called when normal lookup fails.
        if name == "_loader":
            raise AttributeError(name)
        return getattr(self._loader, name)
