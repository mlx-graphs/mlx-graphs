import mlx.core as mx
import numpy as np
import pytest

from mlx_graphs.data.data import GraphData
from mlx_graphs.datasets.dataset import Dataset


def test_fake_dataset():
    # childred dataset with no implemented download and process methods
    # can't be instantiated
    class FakeDataset(Dataset):
        pass

    with pytest.raises(TypeError):
        _ = FakeDataset("fake_dataset")


def test_dataset_properties():
    data = GraphData(
        node_features=mx.ones((10, 5)),
        edge_labels=mx.ones((10, 1)),
    )

    class Dataset1(Dataset):
        def download(self):
            pass

        def process(self):
            self.graphs = [data]

    dataset = Dataset1("name")

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


test_dataset_properties()
