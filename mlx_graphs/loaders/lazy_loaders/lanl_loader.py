from collections import defaultdict
from typing import Literal

import mlx.core as mx
import numpy as np

from mlx_graphs.data import GraphData
from mlx_graphs.datasets import LazyDataset
from mlx_graphs.datasets.lanl_dataset import (
    LANL_DST,
    LANL_FLOW_BYTE_COUNT,
    LANL_FLOW_DST,
    LANL_FLOW_DUR,
    LANL_FLOW_PKT_COUNT,
    LANL_FLOW_SRC,
    LANL_LABEL,
    LANL_SRC,
)
from mlx_graphs.loaders.lazy_loaders.large_cybersecurity_loader import (
    LargeCybersecurityDataLoader,
)

try:
    import pandas as pd
except ImportError:
    raise ImportError("Install pandas to use the LANLDataLoader")

"""
For custom splits to leverage the overall dataset, just change the values provided in
``time_range``. The range should not exceed 83518, the total number of files (minutes).
"""

LANL_BATCH_SIZE = 60  # Each graph yielded by the loader contains 60min of data
LANL_FIRST_ATTACK = 150885  # time in seconds
LANL_FIRST_ATTACK_FILE = 2513

# Dataset ranges: the +7 is to round snapshots to get exactly
# snapshots of 60 files, and no less.
LANL_TRAIN_END = LANL_FIRST_ATTACK_FILE - 4 * LANL_BATCH_SIZE + 7
LANL_VALID_END = LANL_FIRST_ATTACK_FILE - LANL_BATCH_SIZE + 7
LANL_TEST_END = 83518  # 58 days

LANL_TIME_RANGES = {
    "TRAIN": (0, LANL_TRAIN_END),  # 38 hours
    "VALID": (LANL_TRAIN_END + 1, LANL_VALID_END),  # 3 hours
    "TEST": (LANL_VALID_END + 1, LANL_TEST_END),  # ~56 days
    "ALL": (0, LANL_TEST_END),  # 58 days
}


def standardize_euler_argus(edge_features: np.ndarray) -> np.ndarray:
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    res = np.zeros(edge_features.shape)
    for i in range(edge_features.shape[1]):
        x = edge_features[:, i]
        x = x.astype(np.float32)
        x = (x.astype(np.int64) / (x.std() + 1e-6)).astype(np.int64)
        x = sigmoid(x)
        res[:, i] = x
    return res


class LANLDataLoader(LargeCybersecurityDataLoader):
    """
    This loader can be used to iterate over the ``LANLDataset``.
    By default, it yields a snapshot graph of 60 minutes of data (i.e. 60 files).
    This duration can be changed by setting ``batch_size`` accordingly.

    If ``compress_edges=True``, the yielded graph is compressed
    in such a way that all duplicate edges are compressed into a single edge with
    additional edge features like the count of edges, the count of success auth, etc.
    This compression is used to drastically reduce the size of the graph. This approach
    has been successfully used in papers [1, 2].

    If compression is used, additional features extracted from flows are also added
    as edge features.

    On the first iteration on the loader:

        - reads ``batch_size`` csv files from the provided ``dataset``
        - builds a large graph from the concatenated csv files
        - if ``compress_edges=True``, compresses the graph into a smaller graph without
          any duplicate edges.
          This graph is a ``GraphData`` with the following attributes:

          - ``edge_index``: an mx.array with shape (2, num_edges), the graph structure
          - ``edge_features``: an mx.array with shape (num_edges, 13) and the features
            [#edges, #successes, #failures, #src_type_user,
            #src_type_computer, #src_type_anonymous, #num_flows, mean_flow_duration,
            std_flow_duration, mean_pkt_count, std_pkt_count, mean_byte_count,
            std_byte_count])
          - ``edge_labels``: an mx.array with shape (num_edges,) with the label of each
            edge (1 for attack, 0 for benign). A malicious label will be assigned
            to an edge if 1 or more malicious edge is included in the compressed
            edge.
          - ``node_features``: an mx.array with shape (num_nodes, num_nodes) with
            one-hot encoded vectors for each node.
            Note: the ``edge_timestamps`` from LANLDataset are not included anymore
            as they become incompatible with the compression of the edges.
            By default, the features are standardized using min-max standardization.

        - saves the ``GraphData`` on disk as a `.pkl` for later reuse
          in the future iterations on the loader.

    On the second iteration:

        - the processed graphs already exist on disk, so they are directly loaded.
          If the underlying LANL dataset is modified or the ``batch_size`` and
          ``transform`` args of the loader are changed, the graphs are automatically
          re-processed instead of loading them from disk.
        - if one wants to overwrite existing graphs, set ``force_processing=True``

    References:
        [1] `Euler: Detecting Network Lateral Movement via Scalable Temporal Link \
            Prediction \
            <https://www.ndss-symposium.org/wp-content/uploads/2022-107A-paper.pdf>`_

        [2] `Understanding and Bridging the Gap Between Unsupervised Network \
            Representation Learning and Security Analytics \
            <https://c0ldstudy.github.io/commons/papers/SP2024_paper118.pdf>`_

    Args:
        dataset: An instance of a ``LANLDataset`` dataset
        split: The portion of the dataset to iterate on
            ("all" | "train" | "valid" | "test"). Default to "all".
        time_range: A dictionnary indicating the range in minutes for each set.
            The ``time_range`` contains the ranges that will be used by ``split``.
            By default: "train" contains the range between 0 and 38 hours, comprising
            all the benign activity before any attack. "valid" contains the 3 hours
            after the "train" activity, also comprising benign activity only.
            "test" contains all the activity after "valid", comprising both benign
            and malicious events and lasting for roughly 46 days.
        batch_size: The duration of a snapshot graph. Default to 60 min per graph.
        nb_processes: The number of processes to spawn to load each snapshot. Default
            to ``1``, without using any multiprocessing.
        compress_edges: Whether to merge the edges of the resulting graph.
            If ``True``, all duplicate edges will be removed to keep only one edge,
            with additional edge features. If ``False``,
            the graph simply concatenates all snapshots into a single graph.
            Default to ``False``.
        remove_self_loops: Whether to remove the self-loops in the graph.
            Setting this arg to ``True`` may be extremly slow. Default to ``False``.
        force_processing: Whether to force the loader to process the raw csv files of
            the dataset. If set to ``True``, all csv files will be processed and
            the generated graphs will be saved on disk if ``save_on_disk`` is set
            to `True`. If set to ``False``, the csv files will be processed only
            once and stored on disk, then the version stored on disk will be directly
            loaded at the next iteration instead of re-computing the same graphs.
            Default to ``False``.
        save_on_disk: Whether to save the graphs on disk when processed a first time.
            If set to `True`, the first iteration on the loader processes the graphs
            and saves them on disk, and future iterations simply load the saved graphs.

    Example:

    .. code-block:: python

        from mlx_graphs.datasets import LANLDataset
        from mlx_graphs.loaders import LANLDataLoader

        dataset = LANLDataset()

        # Each iteration yields a 60min compressed graph
        loader = LANLDataLoader(dataset, split="train", batch_size=60)

        next(graph)
        >>> GraphData(
            edge_index(shape=(2, 2758), int32)
            node_features(shape=(17685, 17685), float32)
            edge_features(shape=(2758, 10), float32)
            edge_labels(shape=(2758,), int32))

        for graph in loader:
            print(graph)

        >>> GraphData(
            edge_index(shape=(2, 2739), int32)
            node_features(shape=(17685, 17685), float32)
            edge_features(shape=(2739, 10), float32)
            edge_labels(shape=(2739,), int32))
    """

    def __init__(
        self,
        dataset: LazyDataset,
        split: Literal["all", "train", "valid", "test"] = "all",
        time_range: dict[str, tuple[int]] = LANL_TIME_RANGES,
        batch_size: int = LANL_BATCH_SIZE,
        nb_processes: int = 1,
        compress_edges: bool = False,
        remove_self_loops: bool = False,
        force_processing: bool = False,
        save_on_disk: bool = True,
        **kwargs,
    ):
        super().__init__(
            dataset=dataset,
            split=split,
            time_range=time_range,
            batch_size=batch_size,
            nb_processes=nb_processes,
            compress_edges=compress_edges,
            remove_self_loops=remove_self_loops,
            force_processing=force_processing,
            save_on_disk=save_on_disk,
            **kwargs,
        )

    def compress_graph(
        self, df: pd.DataFrame, edge_feats: np.ndarray, df_flows: pd.DataFrame
    ) -> GraphData:
        df_adj = df[[LANL_SRC, LANL_DST, LANL_LABEL]]

        NUM_AUTH_FEATURES = 6
        NUM_FLOW_FEATURES = 7
        nb_e_feats = NUM_AUTH_FEATURES + NUM_FLOW_FEATURES

        edge_to_feats = defaultdict(lambda: np.zeros((nb_e_feats,)))
        edge_to_labels = defaultdict(int)
        edge_to_auth_count = defaultdict(int)
        edge_to_flow_count = defaultdict(int)

        # Flow features (but only present in 2% of edges)
        edge_to_pkt_count = defaultdict(list)
        edge_to_byte_count = defaultdict(list)
        edge_to_duration = defaultdict(list)

        df_adj = df_adj.to_dict()

        # Auth
        for i, (src, dst, y) in enumerate(
            zip(
                df_adj[LANL_SRC].values(),
                df_adj[LANL_DST].values(),
                df_adj[LANL_LABEL].values(),
            )
        ):
            edge = (src, dst)
            is_success = int(edge_feats[i][0])

            # Sucess/failure
            if is_success:
                edge_to_feats[edge][1] += 1
            else:
                edge_to_feats[edge][2] += 1

            # Source user type
            if edge_feats[i][1] == 1:
                edge_to_feats[edge][3] += 1
            elif edge_feats[i][1] == 2:
                edge_to_feats[edge][4] += 1
            elif edge_feats[i][1] == 3:
                edge_to_feats[edge][5] += 1

            edge_to_labels[edge] = max(edge_to_labels[edge], y)
            edge_to_auth_count[edge] += 1

        # Flows
        df_flows = df_flows.to_dict()
        if len(df_flows) > 0:
            for i, (src, dst, dur, pkt_count, byte_count) in enumerate(
                zip(
                    df_flows[LANL_FLOW_SRC].values(),
                    df_flows[LANL_FLOW_DST].values(),
                    df_flows[LANL_FLOW_DUR].values(),
                    df_flows[LANL_FLOW_PKT_COUNT].values(),
                    df_flows[LANL_FLOW_BYTE_COUNT].values(),
                )
            ):
                edge = (src, dst)
                edge_to_flow_count[edge] += 1
                edge_to_pkt_count[edge].append(pkt_count)
                edge_to_byte_count[edge].append(byte_count)
                edge_to_duration[edge].append(dur)

        def mean_or_zero(x):
            return 0.0 if len(x) == 0 else np.mean(x)

        def std_or_zero(x):
            return 0.0 if len(x) == 0 else np.std(x)

        # convert to np arrays
        edge_index, labels, e_feats = [], [], []
        for edge, feats in edge_to_feats.items():
            feats[0] = edge_to_auth_count[
                edge
            ]  # add the total number of edges between two nodes.
            feats[6] = edge_to_flow_count[edge]  # add the total number of flows.
            feats[7] = mean_or_zero(
                edge_to_duration[edge]
            )  # add the mean of duration, as in argus.
            feats[8] = std_or_zero(
                edge_to_duration[edge]
            )  # add the std of duration, as in argus.
            feats[9] = mean_or_zero(
                edge_to_pkt_count[edge]
            )  # add the mean of packet count, as in argus.
            feats[10] = std_or_zero(
                edge_to_pkt_count[edge]
            )  # add the std of packet count, as in argus.
            feats[11] = mean_or_zero(
                edge_to_byte_count[edge]
            )  # add the mean of byte count, as in argus.
            feats[12] = std_or_zero(
                edge_to_byte_count[edge]
            )  # add the std of byte count, as in argus.

            edge_index.append(edge)
            e_feats.append(feats)
            labels.append(edge_to_labels[edge])

        edge_index, e_feats, labels = (
            np.array(edge_index),
            np.array(e_feats),
            np.array(labels),
        )
        edge_index = np.array([edge_index[:, 0], edge_index[:, 1]])

        # Standardize all features along the 1-axis.
        e_feats = standardize_euler_argus(e_feats)

        graph = GraphData(
            edge_index=mx.array(edge_index, dtype=mx.int64),
            edge_labels=mx.array(labels, dtype=mx.int64),
            edge_features=mx.array(e_feats, dtype=mx.float32),
        )
        return graph

    def _process_results(self, results):
        all_df_adj = [r[0] for r in results]
        all_edge_feats = [r[1] for r in results]
        all_flows = [r[2] for r in results]

        # Merge the results from all workers.
        df_adj, edge_feats, flows = self._merge_workers_output_compressed(
            all_df_adj,
            all_edge_feats,
            all_flows,
        )

        if self._compress_edges:
            graph = self.compress_graph(df_adj, edge_feats, flows)
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

    def _load_chunk_of_snapshots(
        self, files: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        all_df_adjs, all_edge_feats, all_flows = [], [], []

        for file in files:
            df_adj, edge_feats, flows = self.dataset.load_lazily(file, from_loader=True)
            all_df_adjs.append(df_adj)
            all_edge_feats.append(edge_feats)
            all_flows.append(flows)

        # Concatenate the dataframes from chunk.
        df_adj = pd.concat(all_df_adjs, ignore_index=True)

        # Concatenate edge features in the same way.
        edge_feats = np.concatenate((all_edge_feats), axis=0)

        # Concat flows
        flows = pd.concat(all_flows, ignore_index=True)

        return (
            df_adj,
            edge_feats,
            flows,
        )

    def _merge_workers_output_compressed(self, all_df_adj, all_edge_feats, all_flows):
        all_df_adj = pd.concat(all_df_adj, axis=0).reset_index()
        all_edge_feats = np.concatenate((all_edge_feats), axis=0)
        all_flows = pd.concat(all_flows, axis=0, ignore_index=True)

        return (
            all_df_adj,
            all_edge_feats,
            all_flows,
        )

    def _standardize_edge_feats(self, edge_feats: np.ndarray):
        """For categorical features with
        0s, we want to compute the statistics only on the non-0 values.
        We also want to update only the non-0 features.
        """
        nonzero = edge_feats.nonzero()[0]
        if len(nonzero) == 0:
            return edge_feats

        e = 1e-6
        std = edge_feats[nonzero].std()

        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x))

        standardized = (edge_feats[nonzero] - edge_feats[nonzero].mean()) / (std + e)
        standardized = sigmoid(standardized)

        edge_feats[nonzero] = standardized
        return edge_feats
