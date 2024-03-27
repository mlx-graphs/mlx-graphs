import glob
import os
from abc import abstractmethod
from typing import Sequence, Union

import mlx.core as mx
import numpy as np

from mlx_graphs.data import GraphData
from mlx_graphs.datasets.dataset import Dataset


class LazyDataset(Dataset):
    """
    This dataset is designed to handle very large datasets stored on disk.
    Unlike ``Dataset`` that loads the entire data from the dataset into memory at once,
    this dataset only loads the requested parts of the dataset into memory.
    This paradigm enables to work on arbitrary large datasets, given that enough storage
    is available on the Mac, and that the dataset can be easily split into multiple
    parts.

    A useful scenario where this dataset can be powerful is for large temporal
    datasets, where the overall graph can be divided into discrete/continuous
    graph snapshots. In this case, there is no need to store all data into
    memory as we can load and process each graph snapshot independently.

    As for traditional datasets, it is usually useful to implement the ``download``
    method to download and store the raw files on disk.
    The actual processing takes usually place after indexing (i.e. after a call
    to __get__item). To implement the actual processing of a graph at a given index,
    you have to implement the method ``load_lazily``, which usually returns a
    ``GraphData`` object from a given ``graph_path`` path to a raw file to process
    lazily. The ``LazyDataset`` can be seen as an abstraction to load into memory
    a graph `i` based on a file `i` located on disk.
    One can then use advanced indexing to merge multiple graphs/files together by
    implementing ``__concat__``, which specifies how to concatenate together
    multiple graphs.

    Args:
        name: Name of the dataset, used by __repr__ and for disk storage
        raw_file_extension: File extension of the downloaded raw files, required
            to locate them individually on disk.
        num_nodes: Number of nodes in the dataset. Only required if
            ``add_one_hot_node_features`` is used.
    Example:

    .. code-block:: python

        from mlx_graphs.datasets import LANLDataset, Planetoid


        # A traditional Dataset:
        cora = Planetoid("cora")  # all data are into memory
        >>> cora(num_graphs=1)

        # A LazyDataset:
        lanl = LANLDataset()  # all data are on disk
        >>> LANL(num_graphs=83519)

        lanl[0]  # loads only the first graph into memory
        >>> GraphData(
            edge_index(shape=(2, 224), int64)
            edge_features(shape=(224, 2), float32)
            edge_labels(shape=(224,), int64)
            edge_timestamps(shape=(224,), int64))

        lanl[:1440]  # merges 1440 graphs into a single one and loads it into memory
        >>> GraphData(
            edge_index(shape=(2, 762125), int64)
            edge_features(shape=(762125, 2), float32)
            edge_labels(shape=(762125,), int64)
            edge_timestamps(shape=(762125,), int64))
    """

    def __init__(
        self, name: str, raw_file_extension: str, num_nodes: int = None, *args, **kwargs
    ):
        self.graph_file_name = "graph"
        self.raw_file_extension = raw_file_extension
        self.num_nodes = num_nodes

        self._all_files = None
        self._eye = None

        super().__init__(name=name, *args, **kwargs)

        self.range_indices = range(len(self))

    @abstractmethod
    def load_lazily(self, graph_path: str) -> GraphData:
        """
        This method should be overridden to describe the logic to build a graph
        from a raw file stored on disk at the provided path ``graph_path``.

        Args:
            graph_path: Path to the raw file we want to load and transform
                into a graph.

        Returns:
            A graph loaded into memory.
        """
        pass

    @abstractmethod
    def __concat__(self, items: list[GraphData]) -> GraphData:
        """
        This method should be overriden to handle indexing with sequences.

        Ex: dataset[[0, 1, 2]] -> __getitem__() -> yields __concat__([g1, g2, g3])

        Args:
            items: Sequence of graphs to concatenate into a single graph
                for sequence indexing.

        Returns:
            A single graph that merges all graphs given in ``items``.
        """
        pass

    def __len__(self) -> int:
        """
        A ``LazyDataset`` can know its size by counting the number
        of files available on disk.
        """
        if self._all_files is not None:
            return len(self._all_files)
        return len(self.all_files())

    def __getitem__(
        self,
        idx: Union[int, np.integer, slice, mx.array, np.ndarray, Sequence],
    ) -> GraphData:
        """
        Loads and returns the graph at ``idx`` from the disk.

        Args:
            idx: Index or indices of the graph(s) to gather lazily from the disk.
                If ``idx`` is a number, returns the specific dataset.
                If ``idx`` is a sequence, returns the concatenation of all graphs
                into a large one.

        Returns:
            A ``GraphData`` object representing the graph at index ``idx``.
        """
        indices = range(len(self))

        if isinstance(idx, (int, np.integer)) or (
            isinstance(idx, mx.array) and idx.ndim == 0  # type: ignore
        ):
            index = self.range_indices[idx]  # type:ignore - idx here is a singleton
            graph_path = self.graph_path_at_index(index)

            graph = self.load_lazily(graph_path)

            if self.transform is not None:
                graph = self.transform(graph)

            return graph

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

        graphs = [self[i] for i in indices]
        concat_graph = self.__concat__(graphs)
        return concat_graph

    def graph_path_at_index(self, idx: int) -> str:
        """
        Returns the absolute to path to the file that stores the objects for loading
        lazily the graph at index ``idx``.

        Args:
            idx: Index of the graph stored on disk to which get the path.

        Returns:
            The path to the graph stored on disk.
        """
        path = os.path.join(self.raw_path, self.graph_file_name)
        return f"{path}_{idx}.{self.raw_file_extension}"

    def all_files(self) -> list[str]:
        """
        Returns the list of paths to all files available in the the ``raw_path``
        of the dataset.
        """
        # All existing snapshots are collected, and sorted by timestamp.
        extension = self.raw_file_extension

        # Select all snapshot files present in the folder.
        raw_path = os.path.join(self.raw_path, f"*{extension}")
        all_files = glob.glob(raw_path)

        if len(all_files) == 0:
            raise FileNotFoundError(f"No files found in {raw_path}")

        self._all_files = all_files
        return all_files

    def _load(self):
        already_processed = (
            os.path.exists(self.processed_path)
            and len(os.listdir(self.processed_path)) > 0
        )
        already_downloaded = (
            os.path.exists(self.raw_path) and len(os.listdir(self.raw_path)) > 0
        )

        if already_processed:
            return

        if not already_downloaded:
            self._download()
            print(f"Processing {self.name} raw data ...", end=" ")

        os.makedirs(self.processed_path, exist_ok=True)
        self.process()
        print("Done")

    def add_one_hot_node_features(self, graph: GraphData) -> GraphData:
        """
        If one-hot vectors are used as vectors, it isn't necessary to
        store these features on disk, so we can load them lazily
        when yielding the graph.
        """
        assert self.num_nodes, "This dataset required `num_nodes` to compute one-hot."

        if self._eye is None:
            self._eye = mx.eye(self.num_nodes)

        graph.node_features = self._eye
        return graph

    def __hash__(self):
        """
        Computes a unique hash that will yields a different value if
        one of these attributes is changed.
        Useful to generate a unique id of this object for saving on disk.
        """
        return hash(
            (
                self.name,
                self.num_nodes,
                self.transform,
            )
        )
