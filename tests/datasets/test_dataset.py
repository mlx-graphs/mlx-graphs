import os
import pickle

import mlx.core as mx
import numpy as np
import pytest

from mlx_graphs.data.data import GraphData, HeteroGraphData
from mlx_graphs.datasets import Dataset, HeteroGraphDataset


def test_fake_dataset():
    # childred dataset with no implemented download and process methods
    # can't be instantiated
    class FakeDataset(Dataset):
        pass

    with pytest.raises(TypeError):
        _ = FakeDataset("fake_dataset")  # type: ignore


def test_dataset_properties(tmp_path):
    data = GraphData(
        edge_index=mx.ones((2, 10)),
        node_features=mx.ones((10, 5)),
        edge_labels=mx.ones((10, 1)),
    )

    class Dataset1(Dataset):
        def __init__(self):
            super().__init__("test_dataset", base_dir=tmp_path)

        def download(self):
            pass

        def process(self):
            self.graphs = [data]

    dataset = Dataset1()

    assert dataset.num_node_features == 5
    assert dataset.num_edge_features == 0
    assert dataset.num_graph_features == 0

    assert dataset.num_edge_classes == 1
    assert dataset.num_node_classes == 0
    assert dataset.num_graph_classes == 0

    assert dataset.num_graphs == 1
    assert len(dataset) == 1

    assert mx.array_equal(dataset[0].node_features, data.node_features)
    assert mx.array_equal(dataset[0].edge_labels, data.edge_labels)

    assert mx.array_equal(dataset[-1].node_features, data.node_features)
    assert mx.array_equal(dataset[-1].edge_labels, data.edge_labels)

    for seq in [
        dataset[:],
        dataset[mx.array([0])],
        dataset[np.array([0, 0])],
        dataset[[0]],
    ]:
        assert isinstance(seq, Dataset1)

    dataset.graphs = []

    assert dataset.num_edge_classes == 0
    assert dataset.num_node_classes == 0
    assert dataset.num_graph_classes == 0

    # test saved files
    processed_file_name = os.path.join(tmp_path, "test_dataset/processed/graphs.pkl")
    assert os.path.exists(processed_file_name)

    with open(processed_file_name, "rb") as f:
        saved_graphs = pickle.load(f)
        assert mx.array_equal(saved_graphs[0].node_features, data.node_features)


def test_dataset_properties_hetero_graph_data(tmp_path):
    hetero_data = HeteroGraphData(
        edge_index_dict={
            ("author", "writes", "paper"): mx.array([[0, 1], [1, 2]]),
            ("paper", "cites", "paper"): mx.array([[0, 1], [1, 2]]),
        },
        node_features_dict={
            "author": mx.array([[1, 2], [3, 4], [5, 6]]),
            "paper": mx.array([[7, 8], [9, 10], [11, 12]]),
        },
        edge_features_dict={
            ("author", "writes", "paper"): mx.array([[13, 14], [15, 16]]),
            ("paper", "cites", "paper"): mx.array([[17, 18], [19, 20]]),
        },
        node_labels_dict={"author": mx.array([0, 1, 2]), "paper": mx.array([0, 1, 2])},
        edge_labels_dict={
            ("author", "writes", "paper"): mx.array([0, 1]),
            ("paper", "cites", "paper"): mx.array([1, 0]),
        },
        graph_features=mx.array([21, 22]),
        graph_labels=mx.array([1]),
    )

    class HeteroGraphDataSet(HeteroGraphDataset):
        def __init__(self):
            super().__init__("HeteroGraphDataSet", base_dir=tmp_path)

        def download(self):
            pass

        def process(self):
            self.graphs = [hetero_data]

    heteroGraphDataSet = HeteroGraphDataSet()
    assert heteroGraphDataSet.num_node_features["author"] == 2
    assert heteroGraphDataSet.num_graph_features == 1
    assert heteroGraphDataSet.num_edge_classes[("author", "writes", "paper")] == 2
    assert heteroGraphDataSet.num_node_classes["author"] == 3

    processed_file_name = os.path.join(
        tmp_path, "HeteroGraphDataSet/processed/graphs.pkl"
    )
    assert os.path.exists(processed_file_name)

    with open(processed_file_name, "rb") as f:
        saved_graphs = pickle.load(f)
        print(saved_graphs[0])
        assert mx.array_equal(
            saved_graphs[0].node_features_dict["author"],
            heteroGraphDataSet.graphs[0].node_features_dict["author"],
        )


def test_dataset_transform(tmp_path):
    data_list = [
        GraphData(
            edge_index=mx.ones((2, 10)),
            node_features=mx.ones((10, 5)),
            edge_labels=mx.ones((10, 1)),
        )
        for _ in range(3)
    ]

    def custom_transform(graph):
        graph.feat = "test_transform"
        return graph

    class Dataset1(Dataset):
        def __init__(self, transform):
            super().__init__("test_dataset", tmp_path, transform=transform)

        def download(self):
            pass

        def process(self):
            self.graphs = data_list

    dataset = Dataset1(custom_transform)

    for i in range(len(dataset)):
        assert dataset[i].feat == "test_transform"
