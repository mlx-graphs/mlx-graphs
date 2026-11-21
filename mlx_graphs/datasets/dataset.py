import copy
import os
import warnings
from typing import Callable, Literal, Optional, Sequence, Union

import mlx.core as mx
import numpy as np

from mlx_graphs.data import GraphData
from mlx_graphs.datasets.base_dataset import DEFAULT_BASE_DIR, BaseDataset


class Dataset(BaseDataset):
    """
    A dataset class for graph data. ``download`` and ``process`` methods must be
    implemented by children classes. The ``save`` and ``load`` methods save and load
    only the processed ``self.graphs`` attribute by default. You may want to
    override them to store/load additional processed attributes.

    Graph data within the dataset should be stored in ``self.graphs`` as
    a `list[GraphData]`. The creation and preprocessing of this list of graphs
    is typically done within the overridden ``process`` method.

    Args:
        name: name of the dataset
        base_dir: Directory where to store dataset files. Default is
            in the local directory ``.mlx_graphs_data/``.
        pre_transform: A function/transform that takes in a ``GraphData`` object and
            returns a transformed version. The transformation is applied before
            the first access.
        transform: A function/transform that takes in a ``GraphData`` object and returns
            a transformed version. The transformation is applied before every access,
            i.e., during the ``__getitem__`` call.
            By default, no transformation is applied.
    """

    def __init__(
        self,
        name: str,
        base_dir: Optional[str] = None,
        pre_transform: Optional[Callable] = None,
        transform: Optional[Callable] = None,
    ):
        self._name = name
        self._base_dir = base_dir if base_dir else DEFAULT_BASE_DIR
        self.transform = transform
        self.pre_transform = pre_transform
        self.graphs: list[GraphData] = []
        self._load()

    @property
    def processed_path(self) -> str:
        """
        The path where raw files are stored. Defaults at `<base_dir>/<name>/processed`
        """
        return os.path.expanduser(os.path.join(self._base_dir, self.name, "processed"))

    @property
    def num_graphs(self) -> int:
        """Returns the number of graphs in the dataset."""
        return len(self)

    @property
    def num_node_classes(self) -> Union[int, dict[str, int]]:
        """Returns the number of node classes to predict."""
        return self._num_classes("node")

    @property
    def num_edge_classes(self) -> Union[int, dict[str, int]]:
        """Returns the number of edge classes to predict."""
        return self._num_classes("edge")

    @property
    def num_graph_classes(self) -> Union[int, dict[str, int]]:
        """Returns the number of graph classes to predict."""
        return self._num_classes("graph")

    @property
    def num_node_features(self) -> int:
        """Returns the number of node features, after ``transform`` (as in PyG)."""
        return self[0].num_node_features

    @property
    def num_edge_features(self) -> int:
        """Returns the number of edge features, after ``transform`` (as in PyG)."""
        return self[0].num_edge_features

    @property
    def num_graph_features(self) -> int:
        """Returns the number of graph features, after ``transform`` (as in PyG)."""
        return self[0].num_graph_features

    @property
    def num_classes(self) -> int:
        """PyG-style number of classes of the labels ``y`` (the node labels if
        the graphs have some, the graph labels otherwise, see
        :attr:`GraphData.y`), inferred as in PyG from the labels of all the
        graphs (after ``transform``):

        - integer labels with one entry per node (graph): ``max + 1``;
        - floating-point labels with one entry per node (graph): the number of
          distinct values (with a warning if there are more than two);
        - labels with several entries per node (graph), e.g. multi-label
          targets of shape ``[*, num_tasks]``: their last dimension.

        Returns ``0`` if no graph has labels.
        """
        graphs = (
            self.graphs
            if self.transform is None
            else [self[i] for i in range(len(self))]
        )
        labels = []
        for graph in graphs:
            y = getattr(graph, "y", None)
            if y is not None:
                y = np.array(y) if isinstance(y, mx.array) else np.asarray(y)
                labels.append(y.reshape(1) if y.ndim == 0 else y)
        if len(labels) == 0:
            return 0
        y = np.concatenate(labels, axis=0)
        if y.size == 0:
            return 0
        if y.size == y.shape[0] and not np.issubdtype(y.dtype, np.floating):
            return int(y.max()) + 1
        if y.size == y.shape[0]:
            num_classes = int(np.unique(y).size)
            if num_classes > 2:
                warnings.warn(
                    "Found floating-point labels while calling "
                    "`dataset.num_classes`. Returning the number of unique "
                    "elements. Please make sure that this is expected.",
                    stacklevel=2,
                )
            return num_classes
        return int(y.shape[-1])

    @property
    def num_features(self) -> int:
        """PyG-style alias of :attr:`num_node_features`."""
        return self.num_node_features

    def index_select(
        self, idx: Union[slice, mx.array, np.ndarray, Sequence]
    ) -> "Dataset":
        """Returns a new dataset with the graphs at the indices ``idx``.

        The graphs are shared with this dataset (not copied), and the
        ``transform`` of the dataset is still applied lazily when a graph is
        accessed.

        Args:
            idx: The graphs to select, as a slice, a list or array of indices
                (negative indices are allowed), or a boolean mask of shape
                ``[len(dataset)]``.

        Returns:
            The dataset restricted to the selected graphs.

        Example:

        .. code-block:: python

            subset = dataset.index_select([0, 2, 4])
            len(subset)
            # 3
        """
        indices = range(len(self))
        if isinstance(idx, slice):
            selected = list(indices[idx])
        else:
            idx_np = np.asarray(np.array(idx) if isinstance(idx, mx.array) else idx)
            idx_np = idx_np.reshape(-1)
            if idx_np.dtype == np.bool_:
                if idx_np.size != len(self):
                    raise IndexError(
                        f"The boolean mask has {idx_np.size} entries but the "
                        f"dataset has {len(self)} graphs"
                    )
                selected = np.nonzero(idx_np)[0].tolist()
            elif idx_np.size == 0 or np.issubdtype(idx_np.dtype, np.integer):
                selected = [indices[i] for i in idx_np.astype(np.int64).tolist()]
            else:
                raise IndexError(
                    "Dataset indexing failed. Accepted indices are: slice, "
                    "integer or boolean mx.array / np.ndarray, list, tuple "
                    f"(got dtype '{idx_np.dtype}')"
                )

        dataset = copy.copy(self)
        dataset.graphs = [self.graphs[i] for i in selected]
        return dataset

    def shuffle(
        self, return_perm: bool = False
    ) -> Union["Dataset", tuple["Dataset", mx.array]]:
        """Returns a copy of the dataset with the graphs in random order.

        The permutation comes from ``mx.random.permutation``, so it is
        reproducible with ``mx.random.seed``. The graphs themselves are shared,
        not copied.

        Args:
            return_perm: Whether to also return the permutation. Default:
                ``False``

        Returns:
            The shuffled dataset, and the permutation if ``return_perm`` is
            ``True`` (graph ``i`` of the shuffled dataset is graph ``perm[i]``
            of this one).

        Example:

        .. code-block:: python

            mx.random.seed(42)
            shuffled, perm = dataset.shuffle(return_perm=True)
        """
        perm = mx.random.permutation(len(self)).astype(mx.int32)
        dataset = self.index_select(perm)
        if return_perm:
            return dataset, perm
        return dataset

    def _num_classes(
        self, task: Literal["node", "edge", "graph"]
    ) -> Union[int, dict[str, int]]:
        flattened_labels = []
        num_classes_dict = {}
        for g in self.graphs:
            labels = getattr(g, f"{task}_labels")
            if labels is not None:
                flattened_labels.append(labels)

        if len(flattened_labels) == 0 and len(num_classes_dict) == 0:
            return 0
        else:
            if len(flattened_labels) > 0:
                flattened_labels = np.concatenate(flattened_labels)
                return np.unique(flattened_labels).size
            else:
                if task == "node" or task == "edge":
                    return {
                        key: np.unique(np.concatenate(labels)).size
                        for key, labels in num_classes_dict.items()
                    }
                else:  # task == "graph"
                    graph_labels = num_classes_dict.get(None)
                    if graph_labels is not None:
                        return np.unique(np.concatenate(graph_labels)).size
                    else:
                        return 0

    def __len__(self):
        """Number of examples in the dataset"""
        return len(self.graphs)

    def __getitem__(
        self,
        idx: Union[int, np.integer, slice, mx.array, np.ndarray, Sequence],
    ) -> Union["Dataset", GraphData]:
        """
        Returns graphs from the ``Dataset`` at given indices.

        If ``idx`` contains multiple indices (e.g. list or slice), then
        another ``Dataset`` object containing the corresponding indexed graphs
        is returned.
        If ``idx`` is a single index (e.g. int), then a single ``GraphData``
        is returned.

        Args:
            idx: Indices or index of the graphs to gather from the dataset.

        Returns:
            A ``Dataset`` if ``idx`` contains multiple elements, or a
                ``GraphData`` otherwise.
        """
        indices = range(len(self))

        if isinstance(idx, (int, np.integer)) or (
            isinstance(idx, mx.array) and idx.ndim == 0  # type: ignore
        ):
            index = indices[idx]  # type:ignore - idx here is a singleton
            data = self.graphs[index]

            if self.transform is not None:
                data = self.transform(data)

            return data

        if isinstance(idx, (slice, mx.array, np.ndarray)) or (
            isinstance(idx, Sequence) and not isinstance(idx, str)
        ):
            # The transform stays lazy: it is applied when a graph of the
            # returned dataset is accessed, not twice.
            return self.index_select(idx)

        raise IndexError(
            f"Dataset indexing failed. Accepted indices are: int, mx.array, "
            f"list, tuple, np.ndarray (got '{type(idx).__name__}')"
        )

    def __repr__(self):
        return (
            self.name if len(self) is None else f"{self.name}(num_graphs={len(self)})"
        )
