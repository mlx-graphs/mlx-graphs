import math
from typing import Any, Iterable, Literal, Optional, Union

import mlx.core as mx
import numpy as np

from mlx_graphs.data import GraphData, HeteroGraphData
from mlx_graphs.sampling import hetero_neighbor_sample, neighbor_sample

from ._utils import (
    check_index_bounds,
    index_array,
    slice_graph,
    to_numpy,
)
from .hetero_neighbor_loader import EdgeType, _HeteroNeighborLoader
from .neighbor_loader import _HomogeneousNeighborLoader


class NegativeSampling:
    """Configuration of the negative sampling done by :class:`LinkNeighborLoader`.

    Negative nodes are drawn uniformly at random (or proportionally to
    ``src_weight`` / ``dst_weight``), without checking that the resulting
    pairs are absent from the graph: like in PyG, negatives are not
    guaranteed to be true negatives.

    Args:
        mode: ``"binary"`` samples random negative edges, added to
            ``edge_label_index`` with label ``0``. ``"triplet"`` samples
            ``amount`` random negative destination nodes for every positive
            edge, returned as ``src_index`` / ``dst_pos_index`` /
            ``dst_neg_index``.
        amount: Number of negatives per positive edge. Can be a float for
            ``"binary"`` (``ceil(amount * num_positives)`` negatives), must be
            an integer for ``"triplet"``. Defaults to 1
        src_weight: Optional sampling weight of every source node.
        dst_weight: Optional sampling weight of every destination node.

    Example:

    .. code-block:: python

        NegativeSampling("binary", amount=2)
        NegativeSampling("triplet", amount=3)
    """

    def __init__(
        self,
        mode: Literal["binary", "triplet"],
        amount: Union[int, float] = 1,
        src_weight: Optional[Union[mx.array, np.ndarray]] = None,
        dst_weight: Optional[Union[mx.array, np.ndarray]] = None,
    ):
        if mode not in ("binary", "triplet"):
            raise ValueError(f"`mode` must be 'binary' or 'triplet' (got '{mode}')")
        if amount <= 0:
            raise ValueError(f"`amount` must be positive (got {amount})")
        if mode == "triplet" and amount != math.ceil(amount):
            raise ValueError(
                f"`amount` must be an integer for 'triplet' negative sampling "
                f"(got {amount})"
            )
        self.mode = mode
        self.amount = int(amount) if mode == "triplet" else amount
        self.src_weight = None if src_weight is None else to_numpy(src_weight)
        self.dst_weight = None if dst_weight is None else to_numpy(dst_weight)

    def is_binary(self) -> bool:
        return self.mode == "binary"

    def is_triplet(self) -> bool:
        return self.mode == "triplet"

    @staticmethod
    def _sample(num: int, num_nodes: int, weight: Optional[np.ndarray]) -> np.ndarray:
        if weight is None:
            return np.random.randint(0, num_nodes, size=num).astype(np.int64)
        if weight.size != num_nodes:
            raise ValueError(
                f"Negative sampling weights have {weight.size} entries, "
                f"expected {num_nodes}"
            )
        p = weight.astype(np.float64) / weight.sum()
        return np.random.choice(num_nodes, size=num, p=p).astype(np.int64)

    def sample_src(self, num: int, num_nodes: int) -> np.ndarray:
        return self._sample(num, num_nodes, self.src_weight)

    def sample_dst(self, num: int, num_nodes: int) -> np.ndarray:
        return self._sample(num, num_nodes, self.dst_weight)

    @classmethod
    def cast(
        cls, value: Union[None, str, dict, "NegativeSampling"]
    ) -> Optional["NegativeSampling"]:
        if value is None or isinstance(value, NegativeSampling):
            return value
        if isinstance(value, str):
            return cls(value)  # type: ignore
        if isinstance(value, dict):
            return cls(**value)
        raise TypeError(f"Cannot convert {type(value).__name__} to NegativeSampling")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(mode={self.mode}, amount={self.amount})"


def _relabel(
    src: np.ndarray, dst: np.ndarray, same_space: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Unique seed nodes of ``src`` / ``dst`` and the local id of each entry."""
    if same_space:
        seeds, inv = np.unique(np.concatenate([src, dst]), return_inverse=True)
        inv = inv.reshape(-1)
        return seeds, seeds, inv[: src.size], inv[src.size :]
    src_seeds, src_loc = np.unique(src, return_inverse=True)
    dst_seeds, dst_loc = np.unique(dst, return_inverse=True)
    return src_seeds, dst_seeds, src_loc.reshape(-1), dst_loc.reshape(-1)


def _link_seeds(
    src: np.ndarray,
    dst: np.ndarray,
    label: Optional[mx.array],
    neg: Optional[NegativeSampling],
    num_src_nodes: int,
    num_dst_nodes: int,
    same_space: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Adds negatives to a batch of supervision edges and relabels them.

    Returns the seed nodes of the source and destination node spaces (equal
    when ``same_space``) and the batch attributes, as local ids into the
    seeds (which come first in the sampled node lists).
    """
    num_pos = src.size
    out: dict[str, Any] = {}

    if neg is not None and neg.is_triplet():
        dst_neg = neg.sample_dst(num_pos * int(neg.amount), num_dst_nodes)
        src_seeds, dst_seeds, src_loc, dst_loc = _relabel(
            src, np.concatenate([dst, dst_neg]), same_space
        )
        dst_neg_index = dst_loc[num_pos:].reshape(num_pos, -1)
        if neg.amount == 1:
            dst_neg_index = dst_neg_index.reshape(-1)
        out["src_index"] = index_array(src_loc, src_seeds.size)
        out["dst_pos_index"] = index_array(dst_loc[:num_pos], dst_seeds.size)
        out["dst_neg_index"] = index_array(dst_neg_index, dst_seeds.size)
        return src_seeds, dst_seeds, out

    if neg is not None and neg.is_binary():
        num_neg = math.ceil(num_pos * neg.amount)
        src = np.concatenate([src, neg.sample_src(num_neg, num_src_nodes)])
        dst = np.concatenate([dst, neg.sample_dst(num_neg, num_dst_nodes)])
        if label is None:
            label = mx.ones(num_pos)
        neg_label = mx.zeros((num_neg, *label.shape[1:]), dtype=label.dtype)
        label = mx.concatenate([label, neg_label])

    src_seeds, dst_seeds, src_loc, dst_loc = _relabel(src, dst, same_space)
    out["edge_label_index"] = index_array(
        np.stack([src_loc, dst_loc]), max(src_seeds.size, dst_seeds.size)
    )
    if label is not None:
        out["edge_label"] = label
    return src_seeds, dst_seeds, out


class _HomogeneousLinkNeighborLoader:
    """Homogeneous implementation behind :class:`LinkNeighborLoader`."""

    def __init__(
        self,
        data: GraphData,
        num_neighbors: list[int],
        edge_label_index: Optional[mx.array],
        edge_label: Optional[mx.array],
        neg_sampling: Optional[NegativeSampling],
        replace: bool,
        directed: bool,
    ):
        self.data = data
        self.num_neighbors = num_neighbors
        self.neg_sampling = neg_sampling
        self.replace = replace
        self.directed = directed

        self.num_nodes = data.num_nodes
        self.num_edges = data.num_edges
        if edge_label_index is None:
            edge_label_index = data.edge_index
        self.edge_label_index = to_numpy(edge_label_index).astype(np.int64)
        check_index_bounds(self.edge_label_index, self.num_nodes, "edge_label_index")
        self.edge_label = edge_label

        self._colptr, self._row, self._edge_perm = _HomogeneousNeighborLoader._to_csc(
            data.edge_index, self.num_nodes
        )
        mx.eval(self._colptr, self._row)

    def sample(self, input_id: np.ndarray) -> GraphData:
        src, dst = self.edge_label_index[:, input_id]
        label = None
        if self.edge_label is not None:
            label = self.edge_label[mx.array(input_id)]
        seeds, _, meta = _link_seeds(
            src,
            dst,
            label,
            self.neg_sampling,
            self.num_nodes,
            self.num_nodes,
            same_space=True,
        )

        # The C++ sampler reads the buffers directly: evaluate them first.
        seed_nodes = mx.array(seeds.astype(np.int64))
        mx.eval(seed_nodes)
        samples, rows, cols, edges = neighbor_sample(
            self._colptr,
            self._row,
            seed_nodes,
            self.num_neighbors,
            self.replace,
            self.directed,
        )
        n_id = to_numpy(samples).astype(np.int64)
        e_id = self._edge_perm[to_numpy(edges).astype(np.int64)]
        edge_index = np.stack(
            [to_numpy(rows).reshape(-1), to_numpy(cols).reshape(-1)]
        ).astype(np.int64)

        return slice_graph(
            self.data,
            n_id,
            e_id,
            edge_index,
            num_nodes=self.num_nodes,
            num_edges=self.num_edges,
            n_id=index_array(n_id, self.num_nodes),
            e_id=index_array(e_id, self.num_edges),
            input_id=index_array(input_id, self.edge_label_index.shape[1]),
            **meta,
        )


class _HeteroLinkNeighborLoader(_HeteroNeighborLoader):
    """Heterogeneous implementation behind :class:`LinkNeighborLoader`."""

    def __init__(
        self,
        data: HeteroGraphData,
        num_neighbors: Union[list[int], dict[EdgeType, list[int]]],
        edge_label_index: Any,
        edge_label: Optional[mx.array],
        neg_sampling: Optional[NegativeSampling],
        replace: bool,
    ):
        self.data = data
        self.num_neighbors = num_neighbors
        self.neg_sampling = neg_sampling
        self.replace = replace
        self.num_nodes = data.num_nodes

        if isinstance(edge_label_index, tuple) and len(edge_label_index) == 2:
            edge_type, index = edge_label_index
        else:
            edge_type, index = edge_label_index, None
        edge_type = tuple(edge_type)
        if edge_type not in data.edge_index_dict:
            raise ValueError(
                f"Unknown edge type {edge_type}. "
                f"Edge types: {list(data.edge_index_dict)}"
            )
        if index is None:
            index = data.edge_index_dict[edge_type]
        self.input_type: EdgeType = edge_type  # type: ignore
        self.edge_label_index = to_numpy(index).astype(np.int64)
        src_type, _, dst_type = edge_type
        for row, node_type in ((0, src_type), (1, dst_type)):
            check_index_bounds(
                self.edge_label_index[row],
                self.num_nodes[node_type],
                f"edge_label_index ('{node_type}')",
            )
        self.edge_label = edge_label

        self._colptr_dict, self._row_dict, self._perm_dict = data.to_csc_dict()
        mx.eval(
            list(self._colptr_dict.values()),
            list(self._row_dict.values()),
            list(self._perm_dict.values()),
        )

    def sample(self, input_id: np.ndarray) -> HeteroGraphData:
        src_type, _, dst_type = self.input_type
        src, dst = self.edge_label_index[:, input_id]
        label = None
        if self.edge_label is not None:
            label = self.edge_label[mx.array(input_id)]
        same = src_type == dst_type
        src_seeds, dst_seeds, meta = _link_seeds(
            src,
            dst,
            label,
            self.neg_sampling,
            self.num_nodes[src_type],
            self.num_nodes[dst_type],
            same_space=same,
        )
        seed_dict = {src_type: mx.array(src_seeds.astype(np.int64))}
        if not same:
            seed_dict[dst_type] = mx.array(dst_seeds.astype(np.int64))
        mx.eval(list(seed_dict.values()))

        out = hetero_neighbor_sample(
            colptr_dict=self._colptr_dict,
            row_dict=self._row_dict,
            seed_dict=seed_dict,
            num_neighbors=self.num_neighbors,
            num_nodes_dict=self.num_nodes,
            replace=self.replace,
        )
        n_id_dict, edge_index_dict, e_id_dict = self._format_sample(out)

        kwargs: dict[str, Any] = {
            "input_type": self.input_type,
            "input_id": index_array(input_id, self.edge_label_index.shape[1]),
            "n_id_dict": n_id_dict,
            "e_id_dict": e_id_dict,
            "num_sampled_nodes_dict": out.num_sampled_nodes,
            "num_sampled_edges_dict": out.num_sampled_edges,
            "num_nodes_dict": {t: int(ids.shape[0]) for t, ids in n_id_dict.items()},
        }
        kwargs.update(self._slice_custom_attributes(n_id_dict))
        kwargs.update(meta)

        return HeteroGraphData(
            edge_index_dict=edge_index_dict,
            node_features_dict=self._slice_node_dict(
                self.data.node_features_dict, n_id_dict
            ),
            edge_features_dict=self._slice_edge_dict(
                self.data.edge_features_dict, e_id_dict
            ),
            graph_features=self.data.graph_features,
            node_labels_dict=self._slice_node_dict(
                self.data.node_labels_dict, n_id_dict
            ),
            graph_labels=self.data.graph_labels,
            **kwargs,
        )


class LinkNeighborLoader:
    """Mini-batch loader for link-level tasks (e.g. link prediction) on a
    single large graph, following PyG's ``LinkNeighborLoader``.

    Iterates over the supervision edges ``edge_label_index``. For each batch
    of edges (plus optional negative samples), the source and destination
    nodes are used as seeds of a neighbor sampling (as in
    :class:`NeighborLoader`), and the sampled subgraph is returned with the
    supervision edges relabeled to local node ids.

    Each batch holds, besides the sliced subgraph:

    - ``n_id`` / ``e_id``: global ids of the sampled nodes and original ids of
      the sampled edges (``n_id_dict`` / ``e_id_dict`` for heterogeneous
      graphs). The seed nodes come first in ``n_id``.
    - ``input_id``: positions in ``edge_label_index`` of the positive edges
      of the batch.
    - Without negative sampling, or with ``"binary"`` negative sampling:
      ``edge_label_index`` (``[2, num_pos + num_neg]``, positives first, as
      local node ids) and ``edge_label`` (``1`` for positives and ``0`` for
      negatives when ``edge_label`` is not given; when ``edge_label`` is given
      and contains ``0``, it is incremented by one so that ``0`` means
      "negative").
    - With ``"triplet"`` negative sampling: ``src_index``,
      ``dst_pos_index`` (``[batch_size]``) and ``dst_neg_index``
      (``[batch_size]`` or ``[batch_size, amount]``), as local node ids.
    - For heterogeneous graphs: ``input_type``, the supervised edge type.
      ``edge_label_index`` row 0 indexes nodes of its source type, row 1
      nodes of its destination type.

    Supervision edges are not removed from the message-passing graph.

    Args:
        data: The full graph, a :class:`GraphData` or a
            :class:`HeteroGraphData`.
        num_neighbors: Number of neighbors to sample per hop (see
            :class:`NeighborLoader`).
        edge_label_index: The supervision edges. For a :class:`GraphData`, a
            ``[2, num_edges]`` array (defaults to ``data.edge_index``). For a
            :class:`HeteroGraphData` (required), an edge type (all its edges)
            or a tuple ``(edge_type, [2, num_edges] array)``.
        edge_label: Optional label of every supervision edge.
        neg_sampling: Negative sampling configuration: a
            :class:`NegativeSampling`, a mode string (``"binary"`` or
            ``"triplet"``) or a dict of :class:`NegativeSampling` arguments.
        neg_sampling_ratio: Shortcut for
            ``neg_sampling=NegativeSampling("binary", neg_sampling_ratio)``.
        batch_size: Number of supervision edges per mini-batch.
        shuffle: Whether to shuffle the supervision edges at each epoch.
        replace: Whether to sample neighbors with replacement.
        directed: Whether to only include edges in the sampling direction.
            Only ``True`` is supported for heterogeneous graphs.
        sampler: Iterable yielding positions in ``edge_label_index`` for each
            epoch. Overrides ``shuffle``.

    Example:

    .. code-block:: python

        from mlx_graphs.loaders import LinkNeighborLoader

        loader = LinkNeighborLoader(
            data,
            num_neighbors=[10, 5],
            edge_label_index=train_edge_index,
            neg_sampling_ratio=1.0,
            batch_size=256,
            shuffle=True,
        )
        for batch in loader:
            h = model(batch.node_features, batch.edge_index)
            src, dst = batch.edge_label_index
            logits = (h[src] * h[dst]).sum(-1)
            loss = bce(logits, batch.edge_label)
    """

    def __init__(
        self,
        data: Union[GraphData, HeteroGraphData],
        num_neighbors: Union[list[int], dict[EdgeType, list[int]]],
        edge_label_index: Any = None,
        edge_label: Optional[mx.array] = None,
        neg_sampling: Union[None, str, dict, NegativeSampling] = None,
        neg_sampling_ratio: Optional[float] = None,
        batch_size: int = 1,
        shuffle: bool = False,
        replace: bool = False,
        directed: bool = True,
        sampler: Optional[Iterable[int]] = None,
    ):
        if neg_sampling_ratio is not None and neg_sampling_ratio != 0.0:
            if neg_sampling is not None:
                raise ValueError(
                    "Pass either `neg_sampling` or `neg_sampling_ratio`, not both."
                )
            neg_sampling = NegativeSampling("binary", neg_sampling_ratio)
        neg = NegativeSampling.cast(neg_sampling)

        if edge_label is not None and neg is not None:
            if neg.is_triplet():
                raise ValueError(
                    "'edge_label' needs to be undefined for 'triplet'-based "
                    "negative sampling. Use `src_index`, `dst_pos_index` and "
                    "`dst_neg_index` of the returned mini-batch instead."
                )
            if edge_label.size > 0 and mx.min(edge_label).item() == 0:
                # Increment labels such that `zero` now denotes "negative".
                edge_label = edge_label + 1

        if isinstance(data, HeteroGraphData):
            if edge_label_index is None:
                raise ValueError(
                    "For a HeteroGraphData, `edge_label_index` must name the "
                    "supervised edge type: edge_type or (edge_type, index)."
                )
            if not directed:
                raise ValueError(
                    "`directed=False` is not supported for heterogeneous graphs."
                )
            self._impl: Any = _HeteroLinkNeighborLoader(
                data,
                num_neighbors,
                edge_label_index,
                edge_label,
                neg,
                replace,
            )
        elif isinstance(data, GraphData):
            if isinstance(num_neighbors, dict):
                raise ValueError(
                    "A dict of `num_neighbors` per edge type is only supported "
                    "for heterogeneous graphs; pass a list for a GraphData."
                )
            self._impl = _HomogeneousLinkNeighborLoader(
                data,
                num_neighbors,
                edge_label_index,
                edge_label,
                neg,
                replace,
                directed,
            )
        else:
            raise TypeError(
                "`data` must be a GraphData or a HeteroGraphData, "
                f"got {type(data).__name__}."
            )

        num_label_edges = self._impl.edge_label_index.shape[1]
        if edge_label is not None and edge_label.shape[0] != num_label_edges:
            raise ValueError(
                f"`edge_label` has {edge_label.shape[0]} entries but there are "
                f"{num_label_edges} supervision edges."
            )

        self.data = data
        self.neg_sampling = neg
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sampler = sampler
        self._indices = np.arange(num_label_edges)
        self._current_index = 0

    @property
    def edge_label_index(self) -> np.ndarray:
        """The supervision edges (global node ids)."""
        return self._impl.edge_label_index

    @property
    def edge_label(self) -> Optional[mx.array]:
        """The labels of the supervision edges (incremented, see above)."""
        return self._impl.edge_label

    def __len__(self) -> int:
        """Number of mini-batches per epoch."""
        num = len(self.sampler) if self.sampler is not None else len(self._indices)  # type: ignore
        return (num + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> "LinkNeighborLoader":
        self._current_index = 0
        if self.sampler is not None:
            self._indices = np.asarray(list(self.sampler), dtype=np.int64)
        else:
            self._indices = np.arange(self.edge_label_index.shape[1])
            if self.shuffle:
                np.random.shuffle(self._indices)
        return self

    def __next__(self) -> Union[GraphData, HeteroGraphData]:
        if self._current_index >= len(self._indices):
            raise StopIteration
        end = min(self._current_index + self.batch_size, len(self._indices))
        input_id = self._indices[self._current_index : end].astype(np.int64)
        self._current_index = end
        return self._impl.sample(input_id)
