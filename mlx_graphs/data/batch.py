from typing import Optional, Union, overload

import mlx.core as mx
import numpy as np

from mlx_graphs.data.collate import collate
from mlx_graphs.data.data import GraphData


class _BatchAlias:
    """Read-only property that an instance attribute of the same name takes
    precedence over (a non-data descriptor), so that graphs with their own
    ``batch`` or ``ptr`` attribute can still be batched."""

    def __init__(self, fget):
        self.fget = fget
        self.__doc__ = fget.__doc__

    def __get__(self, obj, objtype=None):
        return self if obj is None else self.fget(obj)


class GraphDataBatch(GraphData):
    """Concatenates multiple `GraphData` into a single unified `GraphDataBatch`
    for efficient computation and parallelization over multiple graphs.

    All graphs remain disconnected in the batch, meaning that any pairs of graphs
    have no nodes or edges in common.
    `GraphDataBatch` can be especially used to speed up graph classification tasks,
    where multiple graphs can easily fit into memory and be processed in parallel.

    Args:
        graphs: List of `GraphData` objects to batch together

    Example:

    .. code-block:: python

        from mlx_graphs.data.batch import GraphDataBatch

        graphs = [
            GraphData(
                edge_index=mx.array([[0, 0, 0], [1, 1, 1]]),
                node_features=mx.zeros((3, 1)),
            ),
            GraphData(
                edge_index=mx.array([[1, 1, 1], [2, 2, 2]]),
                node_features=mx.ones((3, 1)),
            ),
            GraphData(
                edge_index=mx.array([[3, 3, 3], [4, 4, 4]]),
                node_features=mx.ones((3, 1)) * 2,
            )
        ]
        batch = GraphDataBatch(graphs)
        >>> GraphDataBatch(
            edge_index(shape=[2, 9], int32)
            node_features(shape=[9, 1], float32))

        batch.num_graphs
        >>> 3

        batch[1]
        >>> GraphData(
            edge_index(shape=[2, 3], int32)
            node_features(shape=[3, 1], float32))

        batch[1:]
        >>> [
                GraphData(
                    edge_index(shape=[2, 3], int32)
                    node_features(shape=[3, 1], float32)),
                GraphData(
                    edge_index(shape=[2, 3], int32)
                    node_features(shape=[3, 1], float32))
            ]
    """

    def __init__(self, graphs: list[GraphData], **kwargs):
        batch_kwargs = collate(graphs)
        super().__init__(_num_graphs=len(graphs), **batch_kwargs, **kwargs)

    @property
    def num_graphs(self) -> int:
        """Number of graphs in the batch."""
        return self._num_graphs  # type: ignore - provided via collate

    @_BatchAlias
    def batch_indices(self):
        """Mask indicating for each node its corresponding batch index.

        Graphs that carry their own ``batch_indices`` (e.g. the outputs of
        :class:`~mlx_graphs.loaders.ShaDowKHopSampler`, one subgraph per seed)
        keep it, offset per graph so that ids stay distinct, as PyG does with
        ``batch``."""
        return self._batch_indices  # type: ignore - provided via collate

    @_BatchAlias
    def batch(self) -> mx.array:
        """PyG-style alias of :attr:`batch_indices`: the index of the graph of
        each node."""
        return self.batch_indices

    @_BatchAlias
    def ptr(self) -> mx.array:
        """Offsets of the nodes of each graph in the batch, as in PyG: the nodes
        of graph ``i`` are ``ptr[i]`` to ``ptr[i + 1] - 1``. Has
        ``num_graphs + 1`` entries (int32)."""
        cumsum = getattr(self, "_cumsum_edge_index", None)
        if cumsum is not None:
            return cumsum.astype(mx.int32)
        counts = np.bincount(np.asarray(self.batch_indices), minlength=self.num_graphs)
        ptr = np.zeros(self.num_graphs + 1, dtype=np.int32)
        np.cumsum(counts, out=ptr[1:])
        return mx.array(ptr)

    @overload
    def __getitem__(self, idx: int) -> GraphData:
        ...

    @overload
    def __getitem__(self, idx: Union[slice, mx.array, list[int]]) -> list[GraphData]:
        ...

    def __getitem__(
        self, idx: Union[int, slice, mx.array, np.ndarray, list[int]]
    ) -> Union[GraphData, list[GraphData]]:
        """Graphs of the batch, indexed like a Python list of graphs.

        An integer (negative ones count from the end) returns a
        :class:`GraphData`. A slice, a list, tuple or 1D array of indices, or
        a boolean mask of shape ``[num_graphs]`` returns a list of graphs.
        A string returns the attribute of that name.
        """
        if isinstance(idx, str):
            # Attribute access by name, e.g. `batch["node_features"]`.
            return super().__getitem__(idx)

        num_graphs = self.num_graphs
        if isinstance(idx, (int, np.integer)) or (
            isinstance(idx, mx.array) and idx.ndim == 0
        ):
            index = int(idx.item() if isinstance(idx, mx.array) else idx)
            return self._get_graph(self._check_index(np.array([index]))[0])

        if isinstance(idx, slice):
            plan = self._unbatch_plan()
            return [self._get_graph(i, plan) for i in range(*idx.indices(num_graphs))]

        if isinstance(idx, (mx.array, np.ndarray, list, tuple)):
            index = np.array(idx) if isinstance(idx, mx.array) else np.asarray(idx)
            if index.ndim != 1:
                raise ValueError("Batch indexing only supports 1D index arrays.")
            if index.dtype == np.bool_:
                if index.size != num_graphs:
                    raise IndexError(
                        f"Boolean mask of size {index.size} for a batch of "
                        f"{num_graphs} graphs."
                    )
                index = np.nonzero(index)[0]
            elif index.size == 0:
                return []
            elif not np.issubdtype(index.dtype, np.integer):
                raise IndexError(
                    f"Batch indices must be integers or booleans (got {index.dtype})."
                )
            plan = self._unbatch_plan()
            return [self._get_graph(i, plan) for i in self._check_index(index)]

        raise TypeError(
            "GraphDataBatch indices should be int, slice, a list or 1D array "
            f"of indices or a boolean mask (got '{type(idx).__name__}')."
        )

    def _check_index(self, index: np.ndarray) -> list[int]:
        """Positive graph indices from ``index`` (negative indices count from
        the end). Raises an ``IndexError`` if any is out of range."""
        num_graphs = self.num_graphs
        index = index.astype(np.int64)
        if index.size > 0 and (index.min() < -num_graphs or index.max() >= num_graphs):
            raise IndexError(
                f"Batch index out of range for a batch of {num_graphs} graphs."
            )
        return np.where(index < 0, index + num_graphs, index).tolist()

    def __len__(self):
        """The length of a batch is its number of graphs."""
        return self.num_graphs

    def _unbatch_plan(self) -> dict:
        """How to get each attribute of a graph back from the batch, computed
        once per indexing call: the slice bounds of every graph are read on
        the host here, so extracting a graph needs no device synchronization.
        """
        batch_dict = self.to_dict()
        # vectors added by the loader's `follow_batch` only exist in batches
        follow_batch = batch_dict.get("_follow_batch", ())
        plan: dict = {}
        for attr, value in batch_dict.items():
            if attr.startswith("_") or attr in follow_batch:
                continue
            if f"_scalar_{attr}" in batch_dict:
                # Scalars were stored one per graph
                plan[attr] = ("scalar", np.array(value))
            elif f"_list_{attr}" in batch_dict:
                plan[attr] = ("list", value)
            else:
                sizes = np.array(batch_dict[f"_size_{attr}"], dtype=np.int64)
                bounds = np.zeros(sizes.size + 1, dtype=np.int64)
                np.cumsum(sizes, out=bounds[1:])
                plan[attr] = (
                    "array",
                    value,
                    bounds.tolist(),
                    batch_dict[f"_cat_dim_{attr}"],
                    batch_dict.get(f"_cumsum_{attr}")
                    if f"_inc_{attr}" in batch_dict
                    else None,
                    f"_stack_{attr}" in batch_dict,
                )
        num_nodes = batch_dict.get("_num_nodes_per_graph")
        if num_nodes is not None:
            plan["_num_nodes"] = np.array(num_nodes).tolist()
        return plan

    def _get_graph(self, idx: int, plan: Optional[dict] = None) -> GraphData:
        """
        Returns a `GraphData` from the batch with its original attributes and edge
        index.
        """
        if plan is None:
            plan = self._unbatch_plan()
        single_graph_dict = {}
        for attr, entry in plan.items():
            if attr == "_num_nodes":
                single_graph_dict["num_nodes"] = entry[idx]
                continue
            kind = entry[0]
            if kind == "scalar":
                single_graph_dict[attr] = entry[1][idx].item()
                continue
            if kind == "list":
                single_graph_dict[attr] = entry[1][idx]
                continue

            _, value, bounds, cat_dim, cumsum, stacked = entry
            # Rows of the graph along the concatenation dimension
            index = [slice(None)] * value.ndim
            index[cat_dim] = slice(bounds[idx], bounds[idx + 1])
            original_value = value[tuple(index)]

            # Incremented attributes get their original values back by
            # subtracting the offset of the graph, in the dtype of the
            # attribute (e.g. int32 `edge_index` stays int32)
            if cumsum is not None:
                offset = cumsum[idx].astype(original_value.dtype)
                original_value = original_value - offset

            if stacked:  # 0-dim in every graph
                original_value = original_value.reshape(())
            single_graph_dict[attr] = original_value

        return GraphData(**single_graph_dict)


def batch(graphs: list[GraphData]) -> GraphDataBatch:
    """Constructs a `GraphDataBatch` object from a list of `GraphData`.

    Args:
        batch: List of `GraphData` to merge into a single batch

    Returns:
        `GraphDataBatch` storing a large batched graph
    """
    return GraphDataBatch(graphs)


def unbatch(batch: GraphDataBatch) -> list[GraphData]:
    """Reconstructs a list of `GraphData` objects from a `GraphDataBatch`.

    Args:
        batch: `GraphDataBatch` to unbatch

    Returns:
        List of unbatched `GraphData`
    """
    return batch[:]  # type: ignore
