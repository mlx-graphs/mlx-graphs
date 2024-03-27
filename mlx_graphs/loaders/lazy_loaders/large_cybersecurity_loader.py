from abc import abstractmethod
from typing import Literal

import numpy as np
from joblib import Parallel, delayed

from mlx_graphs.data import GraphData
from mlx_graphs.datasets import LazyDataset
from mlx_graphs.utils import remove_self_loops
from mlx_graphs.utils.validators import validate_package

from .lazy_loader import LazyDataLoader


class LargeCybersecurityDataLoader(LazyDataLoader):
    """
    This loader contains boilerplate code used for loading large cybersecutiy
    datasets (LANL, OpTC) lazily.

    The logic behind graph compression (i.e. replacing duplicate edges by a single
    edge with additional edge features) is implemented in this class.
    It supports multiprocessing to read multiple files in parallel for building
    a single graph.

    By default, every time a graph has been processed, it is stored on disk
    in order to be reused in the next iterations.

    Args:
        dataset: An instance of a ``LANLDataset`` dataset
        split: The portion of the dataset to iterate on ("train" | "valid" | "test").
        time_range: A dictionnary indicating the range in minutes for each set.
            The ``time_range`` contains the ranges that will be used by ``split``.
            The end of the range is included, e.g. a range (0, 10)
            iterates over 11 graphs.
        batch_size: The duration of a snapshot graph.
        nb_processes: The number of processes to spawn to process each snapshot.
            This affects only the processing of the graphs, not the loading
            from disk. Default to ``1``, without using any multiprocessing.
        compress_edges: Whether to merge the edges of the resulting graph.
            If ``True``, all duplicate edges will be removed to keep only one edge,
            with additional edge features. If ``False``,
            the graph simply concatenates all snapshots into a single graph.
            Default to ``False``.
        remove_self_loops: Whether to remove the self-loops in the graph.
            Default to ``True``.
        force_processing: Whether to force the loader to process the raw csv files of
            the dataset. If set to ``True``, all csv files will be processed and
            the generated graphs will be saved on disk. If set to ``False``, the
            csv files will be processed only once and stored on disk, then the version
            stored on disk will be directly loaded at the next iteration instead
            of re-computing the same graphs. Default to ``False``.
        save_on_disk: Whether to save the graphs on disk when processed a first time.
            If set to `True`, the first iteration on the loader processes the graphs
            and saves them on disk, and future iterations simply load the saved graphs.
            Default to ``True``.
    """

    def __init__(
        self,
        dataset: LazyDataset,
        split: Literal["all", "train", "valid", "test"],
        time_range: dict[str, tuple[int]],
        batch_size: int,
        nb_processes: int = 1,
        compress_edges: bool = False,
        remove_self_loops: bool = True,
        force_processing: bool = False,
        save_on_disk: bool = True,
        **kwargs,
    ):
        self._nb_processes = nb_processes
        self._remove_self_loops = remove_self_loops
        self._compress_edges = compress_edges
        self._save_on_disk = save_on_disk

        ranges = time_range[split.upper()]
        super().__init__(
            dataset=dataset,
            ranges=ranges,
            batch_size=batch_size,
            force_processing=force_processing,
        )

    @abstractmethod
    def compress_graph(self, df: "DataFrame", edge_feats: np.array) -> GraphData:  # noqa: F821
        """
        Removes all duplicate edges and replaces them by a single edge with
        additinal edge features. This is used to reduce drastically the size
        of the graph while preserving some statistics about the number of edges.

        Args:
            df: Dataframe containing edges, labels and additional features
            edge_features: Array with edge features

        Returns:
            A compressed graph containing unique edges, with edge features, edge labels
            and possibly edge timestamps.
        """
        pass

    @validate_package("joblib")
    def process_graph(self) -> GraphData:
        start = self.current_batch * self._batch_size
        end = min(
            (self.current_batch + 1) * self._batch_size, len(self._all_sorted_snapshots)
        )
        workers = self._nb_processes

        # Only gets snapshots from the current snapshot.
        nb_snapshots = len(self._all_sorted_snapshots[start:end])

        # We don't want more workers than the number of snapshots to process.
        workers = min(workers, nb_snapshots)

        # Assign a number of snapshot to each worker.
        jobs_per_worker = [nb_snapshots // workers] * workers

        # Add remaining jobs to latter workers.
        remainder = nb_snapshots % workers
        for w in range(remainder):
            jobs_per_worker[workers - 1 - w] += 1

        # Arguments for each worker.
        worker_snapshots = []
        for w in range(workers):
            upto = min(start + jobs_per_worker[w], end)

            snaps = self._all_sorted_snapshots[start:upto]
            worker_snapshots.append(snaps)
            start += jobs_per_worker[w]

        # Run the workers with multiprocessing.
        results = Parallel(n_jobs=workers, prefer="processes")(
            delayed(self._load_chunk_of_snapshots)(
                worker_snapshots[i],
            )
            for i in range(workers)
        )

        all_df_adj = [r[0] for r in results]
        all_edge_feats = [r[1] for r in results]

        # Merge the results from all workers.
        df_adj, edge_feats = self._merge_workers_output_compressed(
            all_df_adj, all_edge_feats
        )

        if self._compress_edges:
            graph = self.compress_graph(df_adj, edge_feats)
        else:
            graph = self.dataset.to_graphdata(df_adj, edge_feats)

        if self._remove_self_loops:
            graph = self._rm_self_loops(graph)

        # Save on disk
        if self._save_on_disk:
            self.save_processed_graph(graph)

        # Add the node features after saving on disk to save storage
        graph = self.dataset.add_one_hot_node_features(graph)

        return graph

    def load_processed_graph(self) -> GraphData:
        graph = super().load_processed_graph()
        graph = self.dataset.add_one_hot_node_features(graph)

        return graph

    @validate_package("pandas")
    def _load_chunk_of_snapshots(
        self, files: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Loop over the snapshots from a single worker and returns
        the concatenation of these snapshots.

        Args:
            files: the files on which to loop on.

        Returns:
            dataframe
            edge features
        """
        import pandas as pd

        all_df_adjs, all_edge_feats = [], []

        for file in files:
            df_adj, edge_feats = self.dataset.load_lazily(file, as_pandas_df=True)
            all_df_adjs.append(df_adj)
            all_edge_feats.append(edge_feats)

        # Concatenate the dataframes from chunk.
        df_adj = pd.concat(all_df_adjs, ignore_index=True)

        # Concatenate edge features in the same way.
        edge_feats = np.concatenate((all_edge_feats), axis=0)

        return (
            df_adj,
            edge_feats,
        )

    @validate_package("pandas")
    def _merge_workers_output_compressed(self, all_df_adj, all_edge_feats):
        """
        Concatenates the result of all workers for compressed graphs.

        Args:
            all_df_adjs: array of dataframes
            all_edge_feats: array of edge feat arrays
        """
        import pandas as pd

        all_df_adj = pd.concat(all_df_adj, axis=0).reset_index()
        all_edge_feats = np.concatenate((all_edge_feats), axis=0)

        return (
            all_df_adj,
            all_edge_feats,
        )

    def _rm_self_loops(self, graph: GraphData):
        edge_attributes = [
            graph.edge_features,
            graph.edge_labels,
        ]
        if hasattr(graph, "edge_timestamps") and graph.edge_timestamps is not None:
            edge_attributes.append(graph.edge_timestamps)

        results = remove_self_loops(
            edge_index=graph.edge_index,
            edge_attributes=edge_attributes,
        )
        edge_index, edge_features, edge_labels, *_ = results

        graph = GraphData(
            node_features=graph.node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            edge_labels=edge_labels,
        )

        # Add edge_timestamps if present
        if len(results) == 4:
            graph.edge_timestamps = results[-1]
        return graph

    def __hash__(self):
        """We want to re-process graphs if one of those attributes is changed"""

        return hash(
            (
                super().__hash__(),
                self._compress_edges,
                self._remove_self_loops,
            )
        )
