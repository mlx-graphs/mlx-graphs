import os
import pickle

import mlx.core as mx
import numpy as np

from mlx_graphs.data.data import GraphData
from mlx_graphs.datasets.lazy_dataset import LazyDataset


def test_lazy_dataset(tmp_path):
    data0 = GraphData(
        node_features=mx.ones((10, 5)),
        edge_labels=mx.ones((10, 1)),
    )
    data1 = GraphData(
        node_features=mx.ones((100, 10)),
        edge_labels=mx.ones((100, 1)),
    )

    class Dataset1(LazyDataset):
        def __init__(self):
            super().__init__(
                "test_dataset", raw_file_extension="pkl", base_dir=tmp_path
            )

        def download(self):
            pass

        def process(self):
            for i, data in enumerate([data0, data1]):
                path = self.graph_path_at_index(i)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as f:
                    pickle.dump(data, f)

        def load_lazily(self, graph_path):
            with open(graph_path, "rb") as f:
                obj = pickle.load(f)
                return obj

        def __concat__(self, items: list[GraphData]) -> GraphData:
            return GraphData(
                edge_labels=mx.concatenate([g.edge_labels for g in items], axis=0),
                node_features=items[0].node_features,
            )

    dataset = Dataset1()

    # Attributes
    assert len(dataset) == 2
    assert dataset.num_graphs == 2

    for i, data in enumerate([data0, data1]):
        assert mx.array_equal(dataset[i].node_features, data.node_features)
        assert mx.array_equal(dataset[i].edge_labels, data.edge_labels)

    # Indexing
    for graph in [
        dataset[0:2],
        dataset[range(2)],
        dataset[[0, 1]],
        dataset[np.array([0, 1])],
        dataset[mx.array([0, 1])],
    ]:
        assert graph.node_features.shape == (10, 5)
        assert graph.edge_labels.shape == (110, 1)
