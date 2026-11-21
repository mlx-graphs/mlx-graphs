from typing import Any, Callable, Iterable, Optional, Sequence, Union

import mlx.core as mx
import numpy as np

from mlx_graphs.data import GraphData, GraphDataBatch, batch
from mlx_graphs.datasets.dataset import Dataset


class Dataloader:
    """
    Default data loader to batch and iterate over multiple graphs.

    Args:
        dataset: `Dataset` or list of `GraphData` to batch and iterate over
        batch_size: Number of graphs to load per batch. Defaults to 1
        shuffle: Whether to reshuffle the order of graphs at the start of each
            epoch. Defaults to False
        drop_last: Whether to drop the last batch if it holds fewer than
            ``batch_size`` graphs. Defaults to False
        follow_batch: Names of attributes for which to create a batch vector
            ``<name>_batch`` assigning each of their elements (along the
            concatenation dimension) to its graph, like PyG's ``x_batch``.
            E.g. ``follow_batch=["node_features"]`` adds
            ``batch.node_features_batch``. Defaults to None
        sampler: Iterable yielding the dataset indices to draw for an epoch,
            e.g. an :class:`ImbalancedSampler`. Mutually exclusive with
            ``shuffle`` and ``batch_sampler``. Defaults to None
        batch_sampler: Iterable yielding lists of dataset indices, one list per
            batch, e.g. a :class:`DynamicBatchSampler`. Mutually exclusive with
            ``batch_size``, ``shuffle``, ``sampler`` and ``drop_last``.
            Defaults to None
        transform: Function applied to every graph before collating.
            Defaults to None
        collate_fn: Function turning a list of graphs into a batch. Defaults
            to :func:`~mlx_graphs.data.batch`, which builds a
            :class:`GraphDataBatch`. ``follow_batch`` requires the default.

    Example:

    .. code-block:: python

        from mlx_graphs.datasets import TUDataset
        from mlx_graphs.loaders import Dataloader

        dataset = TUDataset("MUTAG")
        loader = Dataloader(
            dataset,
            batch_size=32,
            shuffle=True,
            drop_last=True,
            follow_batch=["node_features"],
        )
        for b in loader:
            b.node_features_batch  # graph index of every node
    """

    def __init__(
        self,
        dataset: Union[Dataset, Sequence[GraphData]],
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        follow_batch: Optional[Sequence[str]] = None,
        sampler: Optional[Iterable[int]] = None,
        batch_sampler: Optional[Iterable[Sequence[int]]] = None,
        transform: Optional[Callable[[GraphData], GraphData]] = None,
        collate_fn: Optional[Callable[[list[GraphData]], Any]] = None,
    ):
        if batch_size < 1:
            raise ValueError(f"`batch_size` must be positive (got {batch_size}).")
        if sampler is not None and shuffle:
            raise ValueError("`sampler` and `shuffle` are mutually exclusive.")
        if batch_sampler is not None and (
            batch_size != 1 or shuffle or sampler is not None or drop_last
        ):
            raise ValueError(
                "`batch_sampler` is mutually exclusive with `batch_size`, "
                "`shuffle`, `sampler` and `drop_last`."
            )
        if follow_batch and collate_fn is not None:
            raise ValueError("`follow_batch` requires the default `collate_fn`.")

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.follow_batch = list(follow_batch or [])
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.transform = transform
        self.collate_fn = collate_fn

        self._indices = list(range(len(dataset)))
        self._batches: Optional[list[list[int]]] = None
        self._current_index = 0

    def _shuffle_indices(self):
        np.random.shuffle(self._indices)

    def _make_batches(self) -> list[list[int]]:
        """Splits the indices of one epoch into batches."""
        if self.batch_sampler is not None:
            return [list(b) for b in self.batch_sampler]

        if self.sampler is not None:
            indices = [int(i) for i in self.sampler]
        else:
            if self.shuffle:
                self._shuffle_indices()
            indices = self._indices

        batches = [
            list(indices[i : i + self.batch_size])
            for i in range(0, len(indices), self.batch_size)
        ]
        if self.drop_last and batches and len(batches[-1]) < self.batch_size:
            batches.pop()
        return batches

    def __len__(self) -> int:
        """Number of batches per epoch."""
        if self.batch_sampler is not None:
            return len(self.batch_sampler)  # type: ignore - may raise
        num = len(self.sampler) if self.sampler is not None else len(self.dataset)  # type: ignore
        if self.drop_last:
            return num // self.batch_size
        return (num + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        # Every `for batch in loader` is a new epoch: start from the first
        # batch, and draw a new order when shuffling (including the first).
        self._batches = self._make_batches()
        self._current_index = 0
        return self

    def __next__(self) -> Any:
        """
        Get the next batch of data.

        Returns:
            A batch of graphs (a :class:`GraphDataBatch` with the default
            ``collate_fn``).
        """
        if self._batches is None:
            iter(self)
        if self._current_index >= len(self._batches):  # type: ignore
            raise StopIteration

        batch_indices = self._batches[self._current_index]  # type: ignore
        self._current_index += 1

        graphs = [
            self.dataset[i]  # type: ignore - this is a list[GraphData]
            for i in batch_indices
        ]
        if self.transform is not None:
            graphs = [self.transform(g) for g in graphs]

        if self.collate_fn is not None:
            return self.collate_fn(graphs)

        batched_data = batch(graphs)
        for attr in self.follow_batch:
            _add_follow_batch(batched_data, attr)
        return batched_data


def _add_follow_batch(batched: GraphDataBatch, attr: str) -> None:
    """Adds ``<attr>_batch``, the graph index of each element of ``attr``.

    PyG names (``x``, ``edge_attr``, ``y``) are aliases of stored attributes,
    so their sizes are read from the attribute they stand for.
    """
    attributes = batched.to_dict()
    source = attr
    if f"_size_{attr}" not in attributes:
        if attr == "y":
            has_node_labels = "_size_node_labels" in attributes
            source = "node_labels" if has_node_labels else "graph_labels"
        else:
            source = {"x": "node_features", "edge_attr": "edge_features"}.get(
                attr, attr
            )
    if f"_size_{source}" not in attributes:
        raise ValueError(f"Cannot follow the batch of missing attribute '{attr}'.")
    sizes = np.asarray(attributes[f"_size_{source}"], dtype=np.int64)
    num_graphs = sizes.size
    name = f"{attr}_batch"
    batch_vector = np.repeat(np.arange(num_graphs, dtype=np.int32), sizes)
    setattr(batched, name, mx.array(batch_vector))
    # Not an attribute of the graphs: `batch[i]` / `unbatch` leave it out, as
    # PyG does, so that the graphs come back as they were batched.
    batched._follow_batch = (*getattr(batched, "_follow_batch", ()), name)


class DataListLoader(Dataloader):
    """Data loader yielding plain Python lists of graphs instead of a batch.

    Useful to handle the collation yourself, e.g. to split a batch across
    devices. It takes the same arguments as :class:`Dataloader`, except
    ``collate_fn`` and ``follow_batch``.

    Example:

    .. code-block:: python

        loader = DataListLoader(graphs, batch_size=4, shuffle=True)
        for graph_list in loader:
            assert isinstance(graph_list, list)
    """

    def __init__(
        self,
        dataset: Union[Dataset, Sequence[GraphData]],
        batch_size: int = 1,
        shuffle: bool = False,
        **kwargs,
    ):
        kwargs.pop("collate_fn", None)
        kwargs.pop("follow_batch", None)
        super().__init__(
            dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=list, **kwargs
        )


def _dense_collate(graphs: list[GraphData]) -> GraphData:
    keys = [k for k, v in vars(graphs[0]).items() if not k.startswith("_")]
    out = {}
    for key in keys:
        values = [getattr(g, key, None) for g in graphs]
        if any(v is None for v in values):
            out[key] = None
            continue
        if not all(isinstance(v, mx.array) for v in values):
            out[key] = values
            continue
        shapes = {tuple(v.shape) for v in values}
        if len(shapes) != 1:
            raise ValueError(
                f"DenseDataLoader requires attribute '{key}' to have the same "
                f"shape in every graph (got {sorted(shapes)}). Pad the graphs "
                "first, e.g. with `to_dense_adj` / `to_dense_batch`."
            )
        out[key] = mx.stack(values)
    return GraphData(**out)


class DenseDataLoader(Dataloader):
    """Data loader stacking dense graphs along a new batch dimension.

    Every array attribute must have the same shape in all graphs (e.g. graphs
    padded to the same number of nodes with a dense ``adj`` matrix). Each
    batch is a :class:`GraphData` whose arrays have shape
    ``[batch_size, ...]``. It takes the same arguments as :class:`Dataloader`,
    except ``collate_fn`` and ``follow_batch``.

    Example:

    .. code-block:: python

        graphs = [
            GraphData(
                edge_index=mx.zeros((2, 0), dtype=mx.int32),
                node_features=mx.ones((5, 3)),
                adj=mx.zeros((5, 5)),
            )
            for _ in range(8)
        ]
        loader = DenseDataLoader(graphs, batch_size=4)
        next(iter(loader)).adj.shape  # [4, 5, 5]
    """

    def __init__(
        self,
        dataset: Union[Dataset, Sequence[GraphData]],
        batch_size: int = 1,
        shuffle: bool = False,
        **kwargs,
    ):
        kwargs.pop("collate_fn", None)
        kwargs.pop("follow_batch", None)
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=_dense_collate,
            **kwargs,
        )
