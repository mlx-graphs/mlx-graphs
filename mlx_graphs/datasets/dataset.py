import copy
import os
from abc import ABC, abstractmethod
from typing import Literal, Optional, Sequence, Union

import mlx.core as mx
import numpy as np

from mlx_graphs.data import GraphData

# Default path for downloaded datasets is the current working directory
DEFAULT_BASE_DIR = os.path.join(os.getcwd(), ".mlx_graphs_data/")


class Dataset(ABC):
    """
    Base dataset class. ``download`` and ``process`` methods must be
    implemented by children classes.

    Graph data within the dataset should be stored in ``self.graphs`` as
    a List[GraphData]. The creation and preprocessing of this list of graphs
    is typically done within the overridden ``process`` method.

    Args:
        name: name of the dataset
        base_dir: Directory where to store dataset files. Default is
            in the local directory ``.mlx_graphs_data/``.
    """

    def __init__(
        self,
        name: str,
        base_dir: Optional[str] = None,
    ):
        self._name = name
        self._base_dir = base_dir if base_dir else DEFAULT_BASE_DIR

        self.graphs: list[GraphData] = []
        self._load()

    @property
    def name(self) -> str:
        """
        Name of the dataset
        """
        return self._name

    @property
    def raw_path(self) -> Optional[str]:
        """
        The path where raw files are stored. Defaults at `<base_dir>/<name>/raw`

        """
        if self._base_dir is not None:
            return os.path.expanduser(os.path.join(self._base_dir, self.name, "raw"))
        return None

    @property
    def processed_path(self) -> Optional[str]:
        """
        The path where raw files are stored. Defaults at `<base_dir>/<name>/processed`
        """
        if self._base_dir is not None:
            return os.path.expanduser(
                os.path.join(self._base_dir, self.name, "processed")
            )
        return None

    @property
    def num_graphs(self) -> int:
        """Returns the number of graphs in the dataset."""
        return len(self)

    @property
    def num_node_classes(self) -> int:
        """Returns the number of node classes to predict."""
        return self._num_classes("node")

    @property
    def num_edge_classes(self) -> int:
        """Returns the number of edge classes to predict."""
        return self._num_classes("edge")

    @property
    def num_graph_classes(self) -> int:
        """Returns the number of graph classes to predict."""
        return self._num_classes("graph")

    @property
    def num_node_features(self) -> int:
        """Returns the number of node features."""
        return self.graphs[0].num_node_features

    @property
    def num_edge_features(self) -> int:
        """Returns the number of edge features."""
        return self.graphs[0].num_edge_features

    @property
    def num_graph_features(self) -> int:
        """Returns the number of graph features."""
        return self.graphs[0].num_graph_features

    @abstractmethod
    def download(self):
        """Download the dataset at `self.raw_path`."""
        pass

    @abstractmethod
    def process(self):
        """Process the dataset and possibly save it at `self.processed_path`"""
        pass

    def _download(self):
        if self._base_dir is not None and self.raw_path is not None:
            if os.path.exists(self.raw_path):
                return
            os.makedirs(self.raw_path, exist_ok=True)
            self.download()

    def _load(self):
        self._download()
        self.process()

    def _num_classes(self, task: Literal["node", "edge", "graph"]) -> int:
        flattened_labels = [
            getattr(g, f"{task}_labels")
            for g in self.graphs
            if getattr(g, f"{task}_labels") is not None
        ]

        if len(flattened_labels) == 0:
            return 0

        flattened_labels = np.concatenate(flattened_labels)
        return np.unique(flattened_labels).size

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
            return self.graphs[index]

        if isinstance(idx, slice):
            indices = indices[idx]

        elif isinstance(idx, mx.array) and idx.dtype in [  # type: ignore
            mx.int64,
            mx.int32,
            mx.int16,
        ]:
            return self[idx.flatten().tolist()]  # type: ignore - idx is a mx.array

        elif isinstance(idx, np.ndarray) and idx.dtype == np.int64:
            return self[idx.flatten().tolist()]

        elif isinstance(idx, Sequence) and not isinstance(idx, str):
            indices = [indices[i] for i in idx]

        else:
            raise IndexError(
                f"Dataset indexing failed. Accepted indices are: int, mx.array, "
                f"list, tuple, np.ndarray (got '{type(idx).__name__}')"
            )

        dataset = copy.copy(self)
        dataset.graphs = [self.graphs[i] for i in indices]
        return dataset

    def __repr__(self):
        return (
            self.name if len(self) is None else f"{self.name}(num_graphs={len(self)})"
        )
