import os
import pickle

import mlx.core as mx
import pytest

from mlx_graphs.data.data import GraphData
from mlx_graphs.datasets.lazy_dataset import LazyDataset
from mlx_graphs.loaders import LazyDataLoader


def test_lazy_loader():
    graphs = [
        GraphData(
            edge_index=mx.random.randint(0, 10, (2, 100)),
            edge_labels=mx.ones((100,)) * i,
        )
        for i in range(100)
    ]

    class Dataset1(LazyDataset):
        def __init__(self):
            super().__init__("test_dataset", raw_file_extension="pkl", num_nodes=0)

        def download(self):
            pass

        def process(self):
            for i, data in enumerate(graphs):
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
                edge_index=mx.concatenate([g.edge_index for g in items], axis=1),
                edge_labels=mx.concatenate([g.edge_labels for g in items], axis=0),
            )

    class Loader1(LazyDataLoader):
        def process_graph(self) -> GraphData:
            start = self.current_batch * self._batch_size
            end = min(
                (self.current_batch + 1) * self._batch_size,
                len(self._all_sorted_snapshots),
            )
            return self.dataset[start:end]

    dataset = Dataset1()

    # Batch size 1
    loader = Loader1(dataset, ranges=(0, 10), batch_size=1)
    graph = next(loader)
    assert graph.edge_index.shape == (2, 100)
    assert mx.array_equal(graph.edge_labels, mx.ones((100,)) * 0)

    # Batch size > 1
    loader = Loader1(dataset, ranges=(0, 10), batch_size=10)
    graph = next(loader)
    assert graph.edge_index.shape == (2, 1000)
    assert graph.edge_labels.shape == (1000,)
    assert mx.array_equal(graph.edge_labels[500:600], mx.ones((100,)) * 5)
    assert mx.array_equal(graph.edge_labels[900:], mx.ones((100,)) * 9)
    assert mx.array_equal(graph.edge_labels[:100], mx.ones((100,)) * 0)

    # Batch size > 1 range > 0
    loader = Loader1(dataset, ranges=(50, 99), batch_size=30)
    graph = next(loader)
    assert graph.edge_index.shape == (2, 30 * 100)
    assert graph.edge_labels.shape == (30 * 100,)
    assert mx.array_equal(graph.edge_labels[:100], mx.ones((100,)) * 50)
    assert mx.array_equal(graph.edge_labels[-100:], mx.ones((100,)) * 79)
    assert mx.array_equal(graph.edge_labels[500:600], mx.ones((100,)) * 55)
    assert mx.array_equal(graph.edge_labels[900:1000], mx.ones((100,)) * 59)

    # Iterations
    loader = Loader1(dataset, ranges=(0, 10), batch_size=6)
    graph = next(loader)
    assert graph.edge_labels.shape == (600,)
    assert mx.array_equal(graph.edge_labels[500:], mx.ones((100,)) * 5)

    graph = next(loader)
    assert graph.edge_labels.shape == (500,)
    assert mx.array_equal(graph.edge_labels[400:], mx.ones((100,)) * 10)

    with pytest.raises(StopIteration):
        graph = next(loader)

    # Overflow
    loader = Loader1(dataset, ranges=(50, 99), batch_size=60)
    graph = next(loader)
    assert graph.edge_index.shape == (2, 50 * 100)

    # Out of range
    with pytest.raises(AssertionError):
        loader = Loader1(dataset, ranges=(50, 100), batch_size=60)
        graph = next(loader)
