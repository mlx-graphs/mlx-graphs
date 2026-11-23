import mlx.core as mx
import numpy as np
import pytest

from mlx_graphs.data import GraphData
from mlx_graphs.datasets import Dataset


def make_dataset(tmp_path, transform=None, node_labels=True):
    class ToyDataset(Dataset):
        def __init__(self):
            super().__init__("toy_dataset", tmp_path, transform=transform)

        def download(self):
            pass

        def process(self):
            self.graphs = [
                GraphData(
                    edge_index=mx.array([[0, 1], [1, 0]]),
                    node_features=mx.full((2, 3), i, dtype=mx.float32),
                    node_labels=mx.array([i % 3, 0]) if node_labels else None,
                    graph_labels=mx.array([i % 2]),
                    graph_id=i,
                )
                for i in range(6)
            ]

    return ToyDataset()


def ids(dataset) -> list[int]:
    return [graph.graph_id for graph in dataset.graphs]


def test_num_classes_and_num_features(tmp_path):
    dataset = make_dataset(tmp_path)
    assert dataset.num_features == dataset.num_node_features == 3
    assert dataset.num_classes == dataset.num_node_classes == 3

    graph_level = make_dataset(tmp_path / "graph", node_labels=False)
    assert graph_level.num_classes == graph_level.num_graph_classes == 2


@pytest.mark.parametrize(
    "labels, expected",
    [
        ([[0], [3]], 4),  # integer labels: max + 1, even with missing classes
        ([[0.0], [1.0]], 2),  # float labels: number of distinct values
        ([[[0.0, 1.0, 1.0]], [[1.0, 0.0, 0.0]]], 3),  # multi-label: last dim
    ],
)
def test_num_classes_infers_like_pyg(tmp_path, labels, expected):
    class LabelDataset(Dataset):
        def __init__(self):
            super().__init__("labels", tmp_path)

        def download(self):
            pass

        def process(self):
            self.graphs = [
                GraphData(
                    edge_index=mx.array([[0, 1], [1, 0]]),
                    node_features=mx.zeros((2, 1)),
                    graph_labels=mx.array(y),
                )
                for y in labels
            ]

    assert LabelDataset().num_classes == expected


def test_num_features_counts_transformed_graphs(tmp_path):
    # As in PyG, the feature sizes are those of the graphs the dataset returns.
    def add_two_features(graph):
        graph = graph.clone()
        num_nodes = graph.node_features.shape[0]
        graph.node_features = mx.concatenate(
            [graph.node_features, mx.zeros((num_nodes, 2))], axis=1
        )
        return graph

    dataset = make_dataset(tmp_path, transform=add_two_features)
    assert dataset.num_node_features == dataset.num_features == 5
    assert dataset.graphs[0].num_node_features == 3  # stored graphs untouched


@pytest.mark.parametrize(
    "idx, expected",
    [
        (slice(1, 4), [1, 2, 3]),
        ([5, 0, -1], [5, 0, 5]),
        ((2, 3), (2, 3)),
        (mx.array([4, 1]), [4, 1]),
        (np.array([4, 1], dtype=np.int32), [4, 1]),
        (mx.array([True, False, True, False, False, True]), [0, 2, 5]),
        (np.array([False] * 6), []),
        ([], []),
    ],
)
def test_index_select(tmp_path, idx, expected):
    dataset = make_dataset(tmp_path)
    subset = dataset.index_select(idx)
    assert ids(subset) == list(expected)
    assert len(dataset) == 6
    # Same as indexing with `[]`.
    assert ids(dataset[idx]) == list(expected)


def test_index_select_errors(tmp_path):
    dataset = make_dataset(tmp_path)
    with pytest.raises(IndexError):
        dataset.index_select([6])
    with pytest.raises(IndexError):
        dataset.index_select(np.array([True, False]))
    with pytest.raises(IndexError):
        dataset.index_select(mx.array([0.5]))


def test_shuffle(tmp_path):
    dataset = make_dataset(tmp_path)
    mx.random.seed(0)
    shuffled, perm = dataset.shuffle(return_perm=True)
    assert ids(shuffled) == np.asarray(perm).tolist()
    assert sorted(ids(shuffled)) == list(range(6))
    assert ids(dataset) == list(range(6))

    mx.random.seed(0)
    again = dataset.shuffle()
    assert isinstance(again, Dataset)
    assert ids(again) == ids(shuffled)


def test_transform_is_applied_once_on_subsets(tmp_path):
    def add_one(graph):
        graph = graph.clone()
        graph.node_features = graph.node_features + 1
        return graph

    dataset = make_dataset(tmp_path, transform=add_one)
    for subset in (dataset[1:3], dataset.index_select([1, 2]), dataset.shuffle()):
        for graph in (subset[i] for i in range(len(subset))):
            assert graph.node_features[0, 0].item() == graph.graph_id + 1


def test_shuffle_permutation_is_int32(tmp_path):
    _, perm = make_dataset(tmp_path).shuffle(return_perm=True)
    assert perm.dtype == mx.int32
