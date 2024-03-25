from collections import defaultdict
from typing import Literal

import mlx.core as mx
import numpy as np

from mlx_graphs.data import GraphData
from mlx_graphs.datasets import LazyDataset
from mlx_graphs.datasets.lanl_dataset import (
    LANL_DST,
    LANL_LABEL,
    LANL_SRC,
)

from .large_cybersecurity_loader import LargeCybersecurityDataLoader

"""
By default, we use a train/eval/test splits that span on a 14 days period instead
of the original 58 days, done in ``Understanding and Bridging the Gap Between
Unsupervised Network Representation Learning and Security Analytics``.

For custom splits to leverage the overall dataset, just change the values provided in
``LANL_TIMES_RANGES``. The range should not exceed 83518, the total number of files
(minutes).
"""

LANL_BATCH_SIZE = 60  # Each graph yielded by the loader contains 60min of data
LANL_FIRST_ATTACK = 150885  # time in seconds
LANL_FIRST_ATTACK_FILE = 2513

# Dataset ranges: the +7 is to round snapshots to get exactly
# snapshots of 60 files, and no less.
LANL_TRAIN_END = LANL_FIRST_ATTACK_FILE - 4 * LANL_BATCH_SIZE + 7
LANL_VALID_END = LANL_FIRST_ATTACK_FILE - LANL_BATCH_SIZE + 7
LANL_TEST_END = 20160  # 14 days

# NOTE: In case one wants to use the full dataset comprising 58 days instead of 14 days,
# uncomment this:
# LANL_NUM_NODES = 17685
# LANL_TEST_END = 83518
# TODO(tristan): make default to all dataset.

# Split ranges for train/eval/test sets. The range is defined in number of files
# (i.e. number of minutes)
LANL_TIME_RANGES = {
    "TRAIN": (0, LANL_TRAIN_END),  # 38 hours
    "VALID": (LANL_TRAIN_END + 1, LANL_VALID_END),  # 3 hours
    "TEST": (LANL_VALID_END + 1, LANL_TEST_END),  # 295 hours
    "ALL": (0, LANL_TEST_END),  # 336 hours
}


class LANLDataLoader(LargeCybersecurityDataLoader):
    """
    This loader can be used to iterate over the ``LANLDataset``.
    By default, it yields a snapshot graph of 60 minutes of data (i.e. 60 files).
    This duration can be changed by setting ``batch_size``.

    For each snapshot, all 60 files are read and a large graph is built and compressed
    in such a way that all duplicate edges are compressed into a single edge with
    additional edge features like the count of edges, the count of success auth, etc.
    This compression is used to drastically reduce the size of the graph. This approach
    has been successfully used in papers [1, 2].

    On the first iteration on the loader:

        - reads ``batch_size`` csv files from the provided ``dataset``
        - builds a large graph from the concatenated csv files
        - compresses the graph into a smaller graph without any duplicate edges
          This graph is a ``GraphData``, with the following attributes:

          - ``edge_index``: an mx.array with shape (2, num_edges), the graph structure
          - ``edge_features``: an mx.array with shape (num_edges, 6) and the features
            [#edges, #successes, #failures, #src_type_user,
            #src_type_computer, #src_type_anonymous])
          - ``edge_labels``: an mx.array with shape (num_edges,) with the label of each
            edge (1 for attack, 0 for benign). A malicious label will be assigned
            to an edge if 1 or more malicious edge is included in the compressed
            edge.
          - ``node_features``: an mx.array with shape (num_nodes, num_nodes) with
            one-hot encoded vectors for each node.
            Note: the ``edge_timestamps`` from LANLDataset are not included anymore
            as they become incompatible with the compression of the edges.
            By default, the features are standardized using min-max standardization
            (see ``_standardize_edge_feats``).

        - saves the compressed ``GraphData`` on disk as `.pkl` for later reuse

    On the second iteration:

        - the compressed graphs already exist on disk, so they are directly loaded
        - if one wants to overwrite existing graphs, set ``force_process=True``

    References:
        [1] `Euler: Detecting Network Lateral Movement via Scalable Temporal Link \
            Prediction \
            <https://www.ndss-symposium.org/wp-content/uploads/2022-107A-paper.pdf>`_

        [2] `Understanding and Bridging the Gap Between Unsupervised Network \
            Representation Learning and Security Analytics \
            <https://c0ldstudy.github.io/commons/papers/SP2024_paper118.pdf>`_

    Args:
        dataset: An instance of a ``LANLDataset`` dataset
        split: The portion of the dataset to iterate on ("train" | "valid" | "test").
        time_range: A dictionnary indicating the range in minutes for each set.
            The ``time_range`` contains the ranges that will be used by ``split``.
            By default: "train" contains the range between 0 and 38 hours, comprising
            all the benign activity before any attack. "valid" contains the 3 hours
            after the "train" activity, also comprising benign activity only.
            "test" contains all the activity after "valid", comprising both benign
            and malicious events. By default, the test set id bounded to not exceed
            14 days.
        batch_size: The duration of a snapshot graph. Default to 60 min per graph.
        nb_processes: The number of processes to spawn to load each snapshot. Default
            to ``1``, without using any multiprocessing.
        use_compress_graph: Whether to compress the resulting graph.
            If ``True``, all duplicate edges will be removed to keep only one edge,
            with additional edge features. If ``False``,
            the graph simply concatenates all snapshots into a single graph.
            Default to ``True``.
        remove_self_loops: Whether to remove the self-loops in the graph.
            Default to ``True``.
        force_process: Whether to force the loader to process the raw csv files of
            the dataset. If set to ``True``, all csv files will be processed and
            the generated graphs will be saved on disk. If set to ``False``, the
            csv files will be processed only once and stored on disk, then the version
            stored on disk will be directly loaded at the next iteration instead
            of re-computing the same graphs. Default to ``False``.

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
            node_features(shape=(13184, 13184), float32)
            edge_features(shape=(2758, 10), float32)
            edge_labels(shape=(2758,), int32))

        for graph in loader:
            print(graph)

        >>> GraphData(
            edge_index(shape=(2, 2739), int32)
            node_features(shape=(13184, 13184), float32)
            edge_features(shape=(2739, 10), float32)
            edge_labels(shape=(2739,), int32))
    """

    def __init__(
        self,
        dataset: LazyDataset,
        split: Literal["train", "valid", "test"],
        time_range: dict[str, tuple[int]] = LANL_TIME_RANGES,
        batch_size: int = LANL_BATCH_SIZE,
        nb_processes: int = 1,
        use_compress_graph: bool = True,
        remove_self_loops: bool = True,
        force_process: bool = False,
        **kwargs,
    ):
        super().__init__(
            dataset=dataset,
            split=split,
            time_range=time_range,
            batch_size=batch_size,
            nb_processes=nb_processes,
            use_compress_graph=use_compress_graph,
            remove_self_loops=remove_self_loops,
            force_process=force_process,
            **kwargs,
        )

    def compress_graph(self, df, edge_feats) -> GraphData:
        df_adj = df[[LANL_SRC, LANL_DST, LANL_LABEL]]

        nb_e_feats = 6
        edge_to_feats = defaultdict(lambda: np.zeros((nb_e_feats,)))
        edge_to_labels = defaultdict(int)
        edge_to_count = defaultdict(int)

        df_adj = df_adj.to_dict()

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
            edge_to_count[edge] += 1

        # convert to np arrays
        edge_index, labels, e_feats = [], [], []
        for edge, feats in edge_to_feats.items():
            feats[0] = edge_to_count[
                edge
            ]  # add the total number of edges between two nodes.
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
        for i in range(nb_e_feats):
            e_feats[:, i] = self._standardize_edge_feats(e_feats[:, i])

        graph = GraphData(
            edge_index=mx.array(edge_index, dtype=mx.int64),
            edge_labels=mx.array(labels, dtype=mx.int64),
            edge_features=mx.array(e_feats, dtype=mx.float32),
        )
        return graph

    def _standardize_edge_feats(self, edge_feats):
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
