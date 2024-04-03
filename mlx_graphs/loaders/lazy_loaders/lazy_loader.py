import math
import os
import pickle
from abc import ABC, abstractmethod
from copy import deepcopy

from tqdm import tqdm

from mlx_graphs.data import GraphData
from mlx_graphs.datasets import LazyDataset
from mlx_graphs.datasets.utils.io import get_index_from_filename


class LazyDataLoader(ABC):
    """
    Base class for building a lazy loader, used to lazily load parts
    of a ``LazyDataset``.

    A working ``LazyDataset`` contains files on disk where each file represent
    a graph that may be merged with other graphs/files.
    A ``LazyLoader`` iterates over these graphs on a given ``ranges``,
    indicating the start and end of the files to iterate on. For instance,
    if the dataset contains 1000 files on disk, a (0, 300) range will result
    in a loader that will yield graphs from this range.
    The files generated within a ``LazyDataset`` are sorted by name in such a way
    that this indexing can effectively occur.

    If ``batch_size`` is used, multiple files will be merged into a single large
    graph comprising all ``batch_size`` graphs. For example, using ``batch_size=100``
    for a range (0, 300) will yield 3 graphs containing 100 graphs.
    To successfully use ``batch_size`` to merge multiple graphs together, the
    downstream ``LazyDataset`` requires an implementation of the ``__concat__`` method.

    Args:
        dataset: An instance of a ``LazyDataset`` dataset.
        ranges: A tuple of integers indicating the start and end of the files
            to iterate on. The end of the range is included, e.g. a range (0, 10)
            iterates over 11 graphs.
        batch_size: Number of graphs to merge into a single graph.
        force_processing: Whether to force re-processing all files from the dataset.
            By default, the processed graphs are stored on disk and reused in future
            iterations. This behavior can be removed with ``force_processing=True``.
        tqdm_bar: Whether to show a tqdm progression bar when iterating over the
            loader. Default to ``True``.

    Example:

    .. code-block:: python

        import os
        import pickle

        import mlx.core as mx
        from mlx_graphs.data.data import GraphData
        from mlx_graphs.datasets.lazy_dataset import LazyDataset
        from mlx_graphs.loaders import LazyDataLoader


        graphs = [GraphData(
            edge_index=mx.random.randint(0, 10, (2, 100)),
            edge_labels=mx.ones((100, )) * i,
        ) for i in range(100)]

        class Dataset1(LazyDataset):
            def __init__(self):
                super().__init__("test_dataset", raw_file_extension="pkl")

            #  Downloads external files
            def download(self):
                pass

            #  In practice, here occurs the preprocessing of the downloaded files
            #  and the storage on disk
            def process(self):
                for i, data in enumerate(graphs):
                    path = self.graph_path_at_index(i)
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path, "wb") as f:
                        pickle.dump(data, f)

            #  Given a file, we define here how to load it into memory
            def load_lazily(self, graph_path):
                with open(graph_path, "rb") as f:
                    obj = pickle.load(f)
                    return obj

            #  We define here how to merge multiple graph together
            def __concat__(self, items: list[GraphData]) -> GraphData:
                return GraphData(
                    edge_index=mx.concatenate([g.edge_index for g in items], axis=1),
                    edge_labels=mx.concatenate([g.edge_labels for g in items], axis=0),
                )

        class Loader1(LazyDataLoader):

            #  Manages the indexing. ``self.current_batch`` is incremented
            #  automatically.
            def process_graph(self) -> GraphData:
                start = self.current_batch * self._batch_size
                end = min(
                    (self.current_batch + 1) * self._batch_size,
                    len(self._all_sorted_snapshots)
                )
                return self.dataset[start: end]


        dataset = Dataset1()
        >>> test_dataset(num_graphs=100)

        #  1 iteration = 1 graph
        loader = Loader1(dataset, ranges=(0, 10), batch_size=1)
        next(loader)
        >>> GraphData(
            edge_index(shape=(2, 100), int32)
            edge_labels(shape=(100,), float32))

        #  1 iteration = 10 graphs
        loader = Loader1(dataset, ranges=(0, 10), batch_size=10)
        next(loader)
        >>> GraphData(
            edge_index(shape=(2, 1000), int32)
            edge_labels(shape=(1000,), float32))

        #  1 iteration = 30 graphs, starting from the 50th graph
        loader = Loader1(dataset, ranges=(50, 99), batch_size=30)
        g = next(loader)
        >>> GraphData(
            edge_index(shape=(2, 3000), int32)
            edge_labels(shape=(3000,), float32))

        g.edge_labels
        >>> array([50, 50, 50, ..., 79, 79, 79], dtype=float32)
    """

    def __init__(
        self,
        dataset: LazyDataset,
        ranges: tuple[int, int],
        batch_size: int,
        force_processing: bool = False,
        tqdm_bar: bool = True,
        **kwargs,
    ):
        self.dataset = deepcopy(dataset)
        self._batch_size = batch_size
        self._start_range = ranges[0]
        self._end_range = ranges[1]
        self.force_processing = force_processing
        self._tqdm_bar = tqdm_bar

        self.current_batch = 0
        self.progress_bar = None

        self._all_sorted_snapshots, range_indices = self._get_all_snapshots()
        self._nb_batches = self._get_nb_batches()

        self.dataset.range_indices = range_indices
        os.makedirs(self.processed_path, exist_ok=True)

    @abstractmethod
    def process_graph(self) -> GraphData:
        """
        Computes the next graph from the loader iterator.
        The state of the loader is stored using ``self.current_batch``.

        Returns:
            The graph that corresponds to the current state of the loader.
        """
        pass

    @property
    def processed_path(self) -> str:
        """
        Every `LazyDataLoader` has its own folder on disk to store its
        processed graphs.

        Returns:
            The path to the folder where the loader's files are located.
        """
        unique_id = str(hash(self))
        return os.path.join(self.dataset.processed_path, unique_id)

    def save_processed_graph(self, graph: GraphData):
        """
        Given a processed graph, stores it on disk for later reuse.

        Args:
            graph: Graph to store on disk.
        """
        path = self.current_graph_path
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(graph, f)

    def load_processed_graph(self) -> GraphData:
        """
        Loads a graph stored on disk.

        Returns:
            The graph stored into memory.
        """
        path = self.current_graph_path
        with open(path, "rb") as f:
            graph = pickle.load(f)

        return graph

    def __iter__(self):
        self.current_batch = 0
        if self._tqdm_bar:
            self.progress_bar = tqdm(total=self._nb_batches)
        return self

    def __next__(self) -> GraphData:
        """
        Get the next graph from the iterator.

        Returns:
            The next graph.
        """
        if self.current_batch >= self._nb_batches:
            self.reset_to_start()
            raise StopIteration

        graph = None
        if os.path.exists(self.current_graph_path) and not self.force_processing:
            graph = self.load_processed_graph()
        else:
            graph = self.process_graph()

        def tic():
            self.current_batch += 1
            if self.progress_bar is not None:
                self.progress_bar.update(1)

        if len(graph.edge_index) == 0:
            tic()
            return self.__next__()

        tic()
        return graph

    def reset_to_start(self):
        self.current_batch = 0
        if self.progress_bar is not None:
            self.progress_bar.close()
            self.progress_bar = None

    def _get_all_snapshots(self) -> list[str]:
        """
        Returns the list of all the paths to snapshost for the current time range.
        """
        all_snapshots = self.dataset.all_files()

        all_sorted_snapshots = sorted(
            all_snapshots, key=lambda x: get_index_from_filename(x)
        )

        start_index = [
            i
            for i, sc in enumerate(all_sorted_snapshots)
            if get_index_from_filename(sc) == self._start_range
        ]

        end_index = [
            i
            for i, sc in enumerate(all_sorted_snapshots)
            if get_index_from_filename(sc) == self._end_range
        ]

        assert (
            len(start_index) == 1
        ), f"The starting snapshot {self._start_range} does not exist in files."
        assert (
            len(end_index) == 1
        ), f"The end snapshot {self._end_range} does not exist in files."
        start_index = start_index[0]
        end_index = end_index[0]

        snapshots = all_sorted_snapshots[start_index : end_index + 1]
        indices = [get_index_from_filename(sc) for sc in snapshots]

        return snapshots, indices

    def _get_nb_batches(self) -> int:
        """
        Returns the number of batch to process the current range of data.
        Some snapshots don't have any data (no existing file), so we need
        to count the number of existing snapshots to compute the number of batches.
        """
        nb_batches = math.ceil(len(self._all_sorted_snapshots) / self._batch_size)
        return nb_batches

    @property
    def nb_batches(self):
        return self._nb_batches

    def graph_path_at_index(self, idx: int) -> str:
        """Returns the path to a processed graph stored on disk"""
        return os.path.join(self.processed_path, f"{idx}.pkl")

    @property
    def current_graph_path(self) -> str:
        """Returns the path to the current graph on disk"""
        return self.graph_path_at_index(self.current_batch + self._start_range)

    def __len__(self):
        """The length of a loader is its number of batches"""
        return self.nb_batches

    def __hash__(self):
        """
        Computes a unique hash that will yields a different value if
        one of these attributes is changed.
        Useful to generate a unique id of this object for saving on disk.
        """
        return hash(
            (
                self.dataset,
                self._batch_size,
            )
        )
