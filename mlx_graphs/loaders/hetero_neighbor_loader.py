from typing import Any, Optional, Union

import mlx.core as mx
import numpy as np
from mlx_cluster import hetero_neighbor_sample

from mlx_graphs.data import HeteroGraphData

EdgeType = tuple[str, str, str]


class _HeteroNeighborLoader:
    """Heterogeneous implementation behind :class:`NeighborLoader`.

    Not meant to be used directly: pass a :class:`HeteroGraphData` to
    :class:`NeighborLoader` instead.

    Mini-batch loader for node-level tasks on a single large heterogeneous graph.

    Uses heterogeneous neighbor sampling (``mlx_cluster.hetero_neighbor_sample``)
    to extract K-hop subgraphs around seed nodes of one node type. Each
    iteration yields a :class:`HeteroGraphData` holding the sampled subgraph,
    with edges re-indexed locally per node type.

    The first ``batch_size`` nodes of the seed node type are always the seed
    nodes. During training, compute the loss only on these nodes::

        for batch in loader:
            out = model(batch.node_features_dict, batch.edge_index_dict)
            n = batch.batch_size
            loss = loss_fn(out["author"][:n], batch.node_labels_dict["author"][:n])

    Args:
        data: The full graph.
        num_neighbors: Number of neighbors to sample per hop. Either one list
            used for every edge type (e.g. ``[10, 5]``), or a dict mapping each
            edge type to its own list. ``-1`` samples all neighbors.
        input_nodes: The seed node type and which of its nodes to iterate over:
            ``"author"`` (all authors), ``("author", indices)`` or
            ``("author", boolean_mask)``.
        batch_size: Number of seed nodes per mini-batch.
        shuffle: Whether to shuffle the seed nodes at the start of each epoch.
        replace: Whether to sample neighbors with replacement.

    Each batch contains:

    - ``edge_index_dict``: sampled edges per edge type, as local ids. Edge
      types with no sampled edges have an empty ``[2, 0]`` array.
    - ``node_features_dict``, ``node_labels_dict``: sliced per node type.
    - ``edge_features_dict``: sliced per edge type.
    - ``n_id_dict``: global id of every sampled node, per node type.
    - ``e_id_dict``: original edge position (column in ``data.edge_index_dict``)
      of every sampled edge, per edge type.
    - ``batch_size`` and ``input_type`` (the seed node type).
    - ``num_sampled_nodes_dict`` / ``num_sampled_edges_dict``: per-hop counts.
    - Any other per-node attribute of ``data``, sliced to the sampled nodes:
      dicts keyed by node type, and arrays named ``<node_type>_...`` (e.g.
      ``author_train_mask``) whose length is that node type's node count.

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
        data: HeteroGraphData,
        num_neighbors: Union[list[int], dict[EdgeType, list[int]]],
        input_nodes: Union[str, tuple[str, Optional[mx.array]]],
        batch_size: int = 1,
        shuffle: bool = False,
        replace: bool = False,
    ):
        self.data = data
        self.num_neighbors = num_neighbors
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.replace = replace

        self.num_nodes = data.num_nodes
        self.input_type, self.input_nodes = self._resolve_input_nodes(input_nodes)

        # Pre-compute CSC per edge type once; evaluate so the C++ sampler can
        # read the buffers directly.
        self._colptr_dict, self._row_dict, self._perm_dict = data.to_csc_dict()
        mx.eval(
            list(self._colptr_dict.values()),
            list(self._row_dict.values()),
            list(self._perm_dict.values()),
        )

        # Iteration state
        self._indices = np.arange(len(self.input_nodes))
        self._current_index = 0

    def _resolve_input_nodes(
        self, input_nodes: Union[str, tuple[str, Optional[mx.array]]]
    ) -> tuple[str, mx.array]:
        if isinstance(input_nodes, str):
            node_type, nodes = input_nodes, None
        else:
            node_type, nodes = input_nodes

        if node_type not in self.num_nodes:
            raise ValueError(
                f"Unknown input node type '{node_type}'. "
                f"Node types: {list(self.num_nodes)}"
            )

        if nodes is None:
            nodes = mx.arange(self.num_nodes[node_type], dtype=mx.int64)
        elif nodes.dtype == mx.bool_:
            if nodes.shape[0] != self.num_nodes[node_type]:
                raise ValueError(
                    f"Boolean mask for '{node_type}' has length {nodes.shape[0]}, "
                    f"expected {self.num_nodes[node_type]}"
                )
            nodes = mx.array(np.nonzero(np.asarray(nodes))[0].astype(np.int64))
        else:
            nodes = nodes.astype(mx.int64)
        return node_type, nodes

    def __len__(self) -> int:
        """Number of mini-batches per epoch."""
        return (len(self.input_nodes) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> "_HeteroNeighborLoader":
        self._current_index = 0
        if self.shuffle:
            np.random.shuffle(self._indices)
        return self

    def __next__(self) -> HeteroGraphData:
        if self._current_index >= len(self.input_nodes):
            raise StopIteration

        # 1. Select seed nodes
        end = min(self._current_index + self.batch_size, len(self.input_nodes))
        batch_indices = self._indices[self._current_index : end]
        seed_nodes = self.input_nodes[mx.array(batch_indices)]
        num_seeds = len(batch_indices)
        self._current_index = end

        # 2. Sample the K-hop neighborhood across all edge types
        out = hetero_neighbor_sample(
            colptr_dict=self._colptr_dict,
            row_dict=self._row_dict,
            seed_dict={self.input_type: seed_nodes},
            num_neighbors=self.num_neighbors,
            num_nodes_dict=self.num_nodes,
            replace=self.replace,
        )
        n_id_dict = out.node_id

        # 3. Local edge indices and original edge ids per edge type
        edge_index_dict = {}
        e_id_dict = {}
        for edge_type in self.data.edge_index_dict:
            rows, cols = out.row[edge_type], out.col[edge_type]
            if rows.size > 0:
                edge_index_dict[edge_type] = mx.stack([rows, cols], axis=0)
            else:
                edge_index_dict[edge_type] = mx.zeros((2, 0), dtype=mx.int64)
            e_id_dict[edge_type] = self._perm_dict[edge_type][out.edge_id[edge_type]]

        # 4. Slice standard attributes
        node_features_dict = self._slice_node_dict(
            self.data.node_features_dict, n_id_dict
        )
        node_labels_dict = self._slice_node_dict(self.data.node_labels_dict, n_id_dict)
        edge_features_dict = self._slice_edge_dict(
            self.data.edge_features_dict, e_id_dict
        )

        # 5. Custom per-node attributes (e.g. author_train_mask)
        kwargs: dict[str, Any] = {
            "batch_size": num_seeds,
            "input_type": self.input_type,
            "n_id_dict": n_id_dict,
            "e_id_dict": e_id_dict,
            "num_sampled_nodes_dict": out.num_sampled_nodes,
            "num_sampled_edges_dict": out.num_sampled_edges,
            "num_nodes_dict": {t: int(ids.shape[0]) for t, ids in n_id_dict.items()},
        }
        kwargs.update(self._slice_custom_attributes(n_id_dict))

        return HeteroGraphData(
            edge_index_dict=edge_index_dict,
            node_features_dict=node_features_dict,
            edge_features_dict=edge_features_dict,
            graph_features=self.data.graph_features,
            node_labels_dict=node_labels_dict,
            graph_labels=self.data.graph_labels,
            **kwargs,
        )

    @staticmethod
    def _slice_node_dict(
        values: Optional[dict[str, mx.array]], n_id_dict: dict[str, mx.array]
    ) -> Optional[dict[str, mx.array]]:
        if not values:
            return None
        return {t: v[n_id_dict[t]] for t, v in values.items() if t in n_id_dict}

    @staticmethod
    def _slice_edge_dict(
        values: Optional[dict[EdgeType, mx.array]], e_id_dict: dict[EdgeType, mx.array]
    ) -> Optional[dict[EdgeType, mx.array]]:
        if not values:
            return None
        return {et: v[e_id_dict[et]] for et, v in values.items() if et in e_id_dict}

    # Attributes handled explicitly above, or that describe the full graph.
    _SKIP = {
        "edge_index_dict",
        "node_features_dict",
        "edge_features_dict",
        "graph_features",
        "node_labels_dict",
        "edge_labels_dict",
        "edge_labels_index_dict",
        "graph_labels",
        "num_nodes_dict",
    }

    def _slice_custom_attributes(
        self, n_id_dict: dict[str, mx.array]
    ) -> dict[str, Any]:
        sliced: dict[str, Any] = {}
        # Longest node type first, so "paper_author" wins over "paper" as a prefix.
        node_types = sorted(self.num_nodes, key=len, reverse=True)
        for name, value in vars(self.data).items():
            if name.startswith("_") or name in self._SKIP:
                continue
            if isinstance(value, dict) and value and set(value) <= set(n_id_dict):
                # Dict keyed by node type, e.g. {"author": mask, "paper": mask}
                if all(
                    isinstance(v, mx.array)
                    and v.ndim >= 1
                    and v.shape[0] == self.num_nodes[t]
                    for t, v in value.items()
                ):
                    sliced[name] = {t: v[n_id_dict[t]] for t, v in value.items()}
            elif isinstance(value, mx.array) and value.ndim >= 1:
                # Flat array named "<node_type>_...", e.g. author_train_mask
                for node_type in node_types:
                    if (
                        name.startswith(node_type + "_")
                        and value.shape[0] == self.num_nodes[node_type]
                    ):
                        sliced[name] = value[n_id_dict[node_type]]
                        break
        return sliced
