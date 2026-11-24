from typing import Any, Iterable, Optional, Union

import mlx.core as mx
import numpy as np
from mlx_cluster import neighbor_sample

from mlx_graphs.data import GraphData, HeteroGraphData

from ._disjoint_sampler import build_csc, disjoint_neighbor_sample, graph_time
from ._store_utils import gather_from_store, is_store_pair, store_time
from ._utils import index_array, to_index
from .hetero_neighbor_loader import EdgeType, _HeteroNeighborLoader


def _sampler_array(array: np.ndarray) -> mx.array:
    if (
        isinstance(array, np.memmap)
        and array.dtype == np.int64
        and array.flags.c_contiguous
    ):
        try:
            # The native sampler reads these buffers on the host. Sharing
            # them lets the OS page in just the adjacency it touches.
            return mx.asarray(memoryview(array), copy=False)
        except (TypeError, ValueError):
            # MLX builds without zero-copy host buffer support can still sample.
            pass
    return mx.array(np.ascontiguousarray(array, dtype=np.int64))


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
        data: A single :class:`GraphData` representing the full graph, or a
            ``(FeatureStore, GraphStore)`` pair.
        num_neighbors: Number of neighbors to sample per hop.
            Length determines the number of hops.
            Use ``-1`` to sample all neighbors at a given hop.
            E.g. ``[10, 5]`` samples 10 neighbors at hop-1, 5 at hop-2.
        input_nodes: Seed nodes to iterate over: a 1-D array of node
            indices or a boolean mask (e.g. ``train_mask``). If ``None``, all
            nodes are used.
        batch_size: Number of seed nodes per mini-batch.
        shuffle: Whether to shuffle ``input_nodes`` at the start of each epoch.
        replace: Whether to sample neighbors with replacement.
        directed: Whether to only include edges in the sampling direction.
        sampler: Iterable yielding positions in ``input_nodes`` for each epoch
            (e.g. an :class:`ImbalancedSampler`). Overrides ``shuffle``.
            Duplicate seeds (e.g. drawn with replacement) are kept: each copy
            is a separate seed among the first ``batch_size`` nodes.
        time_attr: Name of a node-level or edge-level timestamp attribute
            for temporal sampling (see :class:`NeighborLoader`).
        input_time: Timestamp of every seed in ``input_nodes``.
        temporal_strategy: ``"uniform"`` or ``"last"``.
        disjoint: Whether every seed gets its own subgraph.
    """

    def __init__(
        self,
        data: Union[GraphData, tuple],
        num_neighbors: list[int],
        input_nodes: Optional[mx.array] = None,
        batch_size: int = 1,
        shuffle: bool = False,
        replace: bool = False,
        directed: bool = True,
        sampler: Optional[Iterable[int]] = None,
        time_attr: Optional[str] = None,
        input_time: Optional[Any] = None,
        temporal_strategy: str = "uniform",
        disjoint: bool = False,
    ):
        self.data = data
        self.num_neighbors = num_neighbors
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.replace = replace
        self.directed = directed
        self.sampler = sampler
        self.time_attr = time_attr
        self.temporal_strategy = temporal_strategy

        self._stores = is_store_pair(data)
        if self._stores:
            self.feature_store, self.graph_store = data  # type: ignore
            self.num_nodes = int(self.graph_store.num_nodes()[1])
            self.num_edges = int(self.graph_store.num_edges())
        else:
            self.num_nodes = data.num_nodes  # type: ignore
            self.num_edges = data.num_edges  # type: ignore

        # Node indices, or a boolean mask (as in PyG): `to_index` turns a mask
        # into the indices of its `True` entries.
        self.input_nodes = mx.array(to_index(input_nodes, self.num_nodes))

        if temporal_strategy not in ("uniform", "last"):
            raise ValueError(
                "`temporal_strategy` must be 'uniform' or 'last' "
                f"(got '{temporal_strategy}')"
            )
        if input_time is not None and time_attr is None:
            raise ValueError(
                "Received 'input_time' but no 'time_attr': temporal sampling "
                "needs the name of the timestamp attribute of the graph."
            )
        self._node_time, self._edge_time = self._resolve_time(time_attr)
        self.input_time = None
        if input_time is not None:
            self.input_time = np.asarray(input_time).reshape(-1)
            if self.input_time.size != self.input_nodes.size:
                raise ValueError(
                    f"`input_time` has {self.input_time.size} entries but "
                    f"there are {self.input_nodes.size} input nodes."
                )
        if self._edge_time is not None and self.input_time is None:
            raise ValueError(
                f"'{time_attr}' is an edge-level time: temporal sampling "
                "needs the time of every seed in `input_time`."
            )
        # As in PyG, temporal sampling always samples one subgraph per seed.
        self.disjoint = disjoint or time_attr is not None
        if self.disjoint and not directed:
            raise ValueError(
                "`directed=False` is not supported with disjoint sampling."
            )

        if self.disjoint:
            src, dst = self._coo()
            (
                self._np_colptr,
                self._np_row,
                self._edge_perm,
                self._csc_time,
            ) = build_csc(src, dst, self.num_nodes, self._node_time, self._edge_time)
        elif self._stores:
            colptr, row, self._edge_perm = self.graph_store.csc()
            self._colptr = _sampler_array(colptr)
            self._row = _sampler_array(row)
            mx.eval(self._colptr, self._row)
        else:
            # Pre-compute CSC format once and evaluate for C++ kernel access
            self._colptr, self._row, self._edge_perm = self._to_csc(
                data.edge_index,
                self.num_nodes,  # type: ignore
            )
            mx.eval(self._colptr, self._row)

        # Iteration state
        self._indices = np.arange(len(self.input_nodes))
        self._current_index = 0

    def _coo(self) -> tuple[np.ndarray, np.ndarray]:
        if self._stores:
            row, col = self.graph_store.coo()
            return np.asarray(row), np.asarray(col)
        edge_index = np.asarray(self.data.edge_index)  # type: ignore
        return edge_index[0], edge_index[1]

    def _resolve_time(
        self, time_attr: Optional[str]
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """``(node_time, edge_time)`` from the ``time_attr`` attribute."""
        if time_attr is None or not self._stores:
            return graph_time(self.data, time_attr)
        time, is_node = store_time(
            self.feature_store, time_attr, self.num_nodes, self.num_edges
        )
        time = time.reshape(time.shape[0], -1)
        if time.shape[1] != 1:
            raise ValueError(f"The time attribute '{time_attr}' must be 1-D.")
        time = time.reshape(-1)
        return (time, None) if is_node else (None, time)

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
        num = len(self.sampler) if self.sampler is not None else len(self.input_nodes)  # type: ignore
        return (num + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> "_HomogeneousNeighborLoader":
        self._current_index = 0
        if self.sampler is not None:
            self._indices = np.asarray(list(self.sampler), dtype=np.int64)
        elif self.shuffle:
            np.random.shuffle(self._indices)
        return self

    def _seed_time(
        self, batch_indices: np.ndarray, seeds: np.ndarray
    ) -> Optional[np.ndarray]:
        if self.input_time is not None:
            return self.input_time[batch_indices]
        if self._node_time is not None:
            return self._node_time[seeds]
        return None

    def __next__(self) -> GraphData:
        if self._current_index >= len(self._indices):
            raise StopIteration

        # 1. Select seed nodes
        end = min(self._current_index + self.batch_size, len(self._indices))
        batch_indices = self._indices[self._current_index : end]
        seed_nodes = self.input_nodes[mx.array(batch_indices)]
        num_seeds = len(batch_indices)
        self._current_index = end

        extra: dict[str, Any] = {}
        if self.disjoint:
            # 2a. One subgraph per seed (temporal sampling), on the host.
            seeds = np.asarray(seed_nodes).astype(np.int64)
            seed_time = self._seed_time(batch_indices, seeds)
            out = disjoint_neighbor_sample(
                self._np_colptr,
                self._np_row,
                seeds,
                self.num_neighbors,
                seed_time=seed_time,
                csc_time=self._csc_time,
                replace=self.replace,
                temporal_strategy=self.temporal_strategy,
            )
            samples = out.node
            edge_index_np = np.stack([out.row, out.col])
            edges = out.edge
            extra["batch"] = index_array(out.batch, max(num_seeds, 1))
            extra["num_sampled_nodes"] = out.num_sampled_nodes
            extra["num_sampled_edges"] = out.num_sampled_edges
            if self.input_time is not None:  # as in PyG
                extra["seed_time"] = mx.array(seed_time)
            edge_dtype = self._edge_dtype()
            edge_index = mx.array(edge_index_np, dtype=edge_dtype)
        else:
            # 2b. Sample K-hop neighborhood
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
            #    (in the dtype of the graph's edge_index)
            edge_dtype = self._edge_dtype()
            if len(rows) > 0:
                edge_index = mx.stack([rows, cols], axis=0).astype(edge_dtype)
            else:
                edge_index = mx.zeros((2, 0), dtype=edge_dtype)

        # 4. Slice standard attributes using global node IDs (int32 ids)
        n_id = index_array(samples, self.num_nodes)
        if self._edge_perm is not None:
            edges = self._edge_perm[np.asarray(edges, dtype=np.int64)]
        original_edge_ids = index_array(edges, self.num_edges)
        if self._stores:
            return self._gather_from_stores(
                edge_index, n_id, original_edge_ids, num_seeds, extra
            )

        node_features = None
        if self.data.node_features is not None:
            node_features = self.data.node_features[n_id]

        node_labels = None
        if self.data.node_labels is not None:
            node_labels = self.data.node_labels[n_id]

        edge_features = None
        if self.data.edge_features is not None and len(edges) > 0:
            edge_features = self.data.edge_features[original_edge_ids]

        # 5. Collect custom per-node attributes (e.g. train_mask)
        kwargs = {
            "batch_size": num_seeds,
            "n_id": n_id,
            "e_id": original_edge_ids,
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
        num_nodes = self.num_nodes
        for attr_name, val in vars(self.data).items():
            if attr_name.startswith("_") or attr_name in _skip:
                continue
            if (
                isinstance(val, mx.array)
                and val.ndim >= 1
                and val.shape[0] == num_nodes
            ):
                kwargs[attr_name] = val[n_id]
            elif (
                attr_name == self.time_attr
                and self._edge_time is not None
                and isinstance(val, mx.array)
            ):
                kwargs[attr_name] = val[original_edge_ids]
        kwargs.update(extra)

        return GraphData(
            edge_index=edge_index,
            node_features=node_features,
            edge_features=edge_features,
            graph_features=self.data.graph_features,
            node_labels=node_labels,
            **kwargs,
        )

    def _edge_dtype(self) -> mx.Dtype:
        if self._stores:
            return mx.int32 if self.num_nodes <= np.iinfo(np.int32).max else mx.int64
        return self.data.edge_index.dtype  # type: ignore

    def _gather_from_stores(
        self,
        edge_index: mx.array,
        n_id: mx.array,
        e_id: mx.array,
        num_seeds: int,
        extra: dict[str, Any],
    ) -> GraphData:
        """Builds the mini-batch from the feature store: only the rows of
        the sampled nodes (and edges) are read."""
        node_index = np.asarray(n_id).astype(np.int64)
        edge_ids = np.asarray(e_id).astype(np.int64)[: edge_index.shape[1]]
        kwargs = gather_from_store(
            self.feature_store, self.num_nodes, self.num_edges, node_index, edge_ids
        )
        kwargs.update(batch_size=num_seeds, n_id=n_id, e_id=e_id, **extra)
        out = GraphData(edge_index=edge_index, **kwargs)
        out.num_nodes = int(node_index.size)
        return out


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
    - A ``(FeatureStore, GraphStore)`` pair (PyG's remote backend, see
      :class:`~mlx_graphs.data.FeatureStore`): the graph structure is read
      from the graph store, and only the rows of the sampled nodes (and
      edges) are gathered from the feature store at every iteration, so
      node features can stay on disk
      (:class:`~mlx_graphs.data.MemmapFeatureStore`). Store arrays named
      ``x`` / ``y`` / ``edge_attr`` become ``node_features`` /
      ``node_labels`` / ``edge_features`` of the mini-batch.
      For static, non-disjoint sampling, CSC adjacency stored in a
      :class:`~mlx_graphs.data.MemmapGraphStore` is shared with the native
      sampler on MLX builds that support zero-copy host buffers.

    **Temporal sampling** (homogeneous graphs): with ``time_attr``, only the
    neighbors whose timestamp is earlier than or equal to the time of their
    seed are sampled, as in PyG. ``time_attr`` names a node-level timestamp
    (the time of a neighbor is then its node's) or an edge-level one (the
    time of the edge, which requires ``input_time``). The time of a seed is
    ``input_time`` if given, else its own node timestamp. As in PyG, every
    seed then gets its own subgraph (``disjoint=True``): a node sampled for
    two seeds appears twice, and ``batch.batch`` holds the position of the
    seed of every node. These mini-batches also hold the per-hop counts
    ``num_sampled_nodes`` / ``num_sampled_edges``, and ``seed_time`` when
    ``input_time`` is given.

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
            :class:`HeteroGraphData`, or a homogeneous graph as a
            ``(FeatureStore, GraphStore)`` pair.
        num_neighbors: Number of neighbors to sample per hop. Its length
            determines the number of hops; ``-1`` samples all neighbors.
            E.g. ``[10, 5]`` samples 10 neighbors at hop-1, 5 at hop-2.
            For heterogeneous graphs this can also be a dict mapping each
            edge type to its own list.
        input_nodes: The seed nodes to iterate over.

            - Homogeneous: a 1-D array of node indices, a boolean mask
              (e.g. ``data.train_mask``), or ``None`` for all nodes.
            - Heterogeneous (required): the seed node type and which of its
              nodes to use: ``"author"`` (all authors), ``("author", indices)``
              or ``("author", boolean_mask)``.
        batch_size: Number of seed nodes per mini-batch.
        shuffle: Whether to shuffle the seed nodes at the start of each epoch.
        replace: Whether to sample neighbors with replacement.
        directed: Whether to only include edges in the sampling direction.
            Only ``True`` is supported for heterogeneous graphs.
        sampler: Iterable yielding, for each epoch, the positions in
            ``input_nodes`` of the seeds to use, e.g. an
            :class:`ImbalancedSampler` built with the same ``input_nodes``.
            Overrides ``shuffle``. Seeds drawn several times in a mini-batch
            are kept as separate seeds for a :class:`GraphData` (so
            ``batch_size`` counts them all), but merged for a
            :class:`HeteroGraphData` (``batch_size`` is then the number of
            distinct seeds).
        time_attr: Name of the node-level or edge-level timestamp attribute
            of ``data`` (or array of the feature store) used for temporal
            sampling. Homogeneous graphs only.
        input_time: Optional time of every seed in ``input_nodes``
            (overrides the node timestamps; required for an edge-level
            ``time_attr``).
        temporal_strategy: ``"uniform"`` samples uniformly among the
            neighbors valid in time, ``"last"`` keeps the most recent ones.
        disjoint: Whether every seed gets its own subgraph (always the case
            with ``time_attr``). Homogeneous graphs only.

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
        data: Union[GraphData, HeteroGraphData, tuple],
        num_neighbors: Union[list[int], dict[EdgeType, list[int]]],
        input_nodes: Union[None, mx.array, str, tuple[str, Optional[mx.array]]] = None,
        batch_size: int = 1,
        shuffle: bool = False,
        replace: bool = False,
        directed: bool = True,
        sampler: Optional[Iterable[int]] = None,
        time_attr: Optional[str] = None,
        input_time: Optional[mx.array] = None,
        temporal_strategy: str = "uniform",
        disjoint: bool = False,
    ):
        if isinstance(data, HeteroGraphData):
            if time_attr is not None or input_time is not None or disjoint:
                raise NotImplementedError(
                    "Temporal and disjoint sampling are only supported for "
                    "homogeneous graphs."
                )
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
                sampler=sampler,
            )
        elif isinstance(data, GraphData) or is_store_pair(data):
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
                sampler=sampler,
                time_attr=time_attr,
                input_time=input_time,
                temporal_strategy=temporal_strategy,
                disjoint=disjoint,
            )
        else:
            raise TypeError(
                "`data` must be a GraphData or a HeteroGraphData (or a "
                f"(FeatureStore, GraphStore) pair), got {type(data).__name__}."
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
