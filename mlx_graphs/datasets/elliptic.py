from typing import Optional

import mlx.core as mx
import numpy as np

from mlx_graphs.data import GraphData
from mlx_graphs.datasets.dataset import Dataset
from mlx_graphs.datasets.utils import download, extract_archive


class EllipticBitcoinDataset(Dataset):
    """The Elliptic Bitcoin dataset of Bitcoin transactions from the
    `"Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional
    Networks for Financial Forensics" <https://arxiv.org/abs/1908.02591>`_
    paper.

    :class:`EllipticBitcoinDataset` maps Bitcoin transactions to real entities
    belonging to licit categories (exchanges, wallet providers, miners,
    licit services, etc.) versus illicit ones (scams, malware, terrorist
    organizations, ransomware, Ponzi schemes, etc.)

    There exists 203,769 node transactions and 234,355 directed edge payments
    flows, with two percent of nodes (4,545) labelled as illicit, and
    twenty-one percent of nodes (42,019) labelled as licit.
    The remaining transactions are unknown

    Args:
        base_dir: Directory where to store dataset files. Default is
            in the local directory ``.mlx_graphs_data/``.
    """

    def __init__(self, base_dir: Optional[str] = None):
        super().__init__(name="ellipticBitcoin", base_dir=base_dir)

    @property
    def raw_file_names(self):
        return [
            "elliptic_txs_features.csv",
            "elliptic_txs_edgelist.csv",
            "elliptic_txs_classes.csv",
        ]

    def download(self):
        # This url is unable to download the data for elliptic bitcoin dataset
        # lets try with pytorch geometric to verify the data
        url = "https://data.pyg.org/datasets/elliptic/"
        for files in self.raw_file_names:
            download(f"{url}{files}.zip", self.raw_path)
            extract_archive(f"{self.raw_path}/{files}.zip", f"{self.raw_path}")

    def process(self, train=True):
        tx_features_from_np = np.loadtxt(
            f"{self.raw_path}/{self.raw_file_names[0]}",
            dtype=float,
            delimiter=",",
            usecols=np.arange(2, 167),
        )
        node_ids = np.loadtxt(
            f"{self.raw_path}/{self.raw_file_names[0]}",
            dtype=str,
            delimiter=",",
            usecols=np.arange(0, 2),
        )
        edge_file = f"{self.raw_path}/{self.raw_file_names[1]}"
        label_file = f"{self.raw_path}/{self.raw_file_names[2]}"

        tx_edges_from_np = np.loadtxt(edge_file, dtype=str, delimiter=",", skiprows=1)
        tx_labels_from_np = np.loadtxt(label_file, dtype=str, delimiter=",", skiprows=1)

        node_features_np = mx.array(tx_features_from_np)

        mapping = {"unknown": 2, "1": 1, "2": 0}

        tx_labels_from_np_classes = tx_labels_from_np[:, 1]
        tx_labels_from_np_classes[tx_labels_from_np_classes == "2"] = 0
        tx_labels_from_np_classes[tx_labels_from_np_classes == "1"] = 1
        tx_labels_from_np_classes[tx_labels_from_np_classes == "unknown"] = 2

        mapping = {idx: i for i, idx in enumerate(node_ids[:, 0])}

        tx_labels_from_np_classes = tx_labels_from_np_classes.astype(int)
        y_numpy = mx.array(tx_labels_from_np_classes.astype(int))

        tx_edges_from_np[:, 0] = np.vectorize(mapping.get)(tx_edges_from_np[:, 0])
        tx_edges_from_np[:, 1] = np.vectorize(mapping.get)(tx_edges_from_np[:, 1])

        tx_edges_from_np = tx_edges_from_np.astype(int)
        edge_index_numpy_array = mx.array(tx_edges_from_np.T)

        time_step = mx.array(node_ids[:, 1].astype(int))

        train_mask = (time_step < 35) & (y_numpy != 2)
        test_mask = (time_step >= 35) & (y_numpy != 2)

        graph = GraphData(
            edge_index=edge_index_numpy_array,
            node_features=node_features_np,
            node_labels=y_numpy,
        )
        graph.train_mask = train_mask
        graph.test_mask = test_mask
        self.graphs = [graph]
