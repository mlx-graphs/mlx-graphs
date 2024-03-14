from typing import Optional

import mlx.core as mx
import pandas as pd

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
        Dataset (_type_): _description_
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
        tx_features = pd.read_csv(
            f"{self.raw_path}/{self.raw_file_names[0]}", header=None
        )
        edge_file = f"{self.raw_path}/{self.raw_file_names[1]}"
        label_file = f"{self.raw_path}/{self.raw_file_names[2]}"
        tx_edges = pd.read_csv(edge_file)
        tx_labels = pd.read_csv(label_file)

        # Get the node features ready for preprocessing
        columns = {0: "txId", 1: "time_step"}
        tx_features = tx_features.rename(columns=columns)
        node_features = mx.array(tx_features.loc[:, 2:].values)

        mapping = {"unknown": 2, "1": 1, "2": 0}
        tx_labels["class"] = tx_labels["class"].map(mapping)

        y = mx.array(tx_labels["class"].values)

        mapping = {idx: i for i, idx in enumerate(tx_features["txId"].values)}
        tx_edges.loc[:, "txId1"] = tx_edges["txId1"].map(mapping)
        tx_edges.loc[:, "txId2"] = tx_edges["txId2"].map(mapping)
        edge_index = mx.array(tx_edges.values.T)
        # Timestamp based split:
        # train_mask: 1 - 34 time_step, test_mask: 35-49 time_step
        time_step = mx.array(tx_features["time_step"].values)
        train_mask = (time_step < 35) & (y != 2)
        test_mask = (time_step >= 35) & (y != 2)

        print("Train mask values is ", train_mask)
        graph = GraphData(
            edge_index=edge_index,
            node_features=node_features,
            node_labels=mx.array(tx_labels["class"].values),
        )
        graph.train_mask = train_mask
        graph.test_mask = test_mask
        self.graphs = [graph]
