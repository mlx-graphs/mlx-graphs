import mlx.core as mx
import numpy as np
import pytest

from mlx_graphs.data import GraphData
from mlx_graphs.transforms import RemoveTrainingClasses
from mlx_graphs.utils import to_pyg


def test_repr():
    assert str(RemoveTrainingClasses([1, 2])) == "RemoveTrainingClasses([1, 2])"


def test_remove_training_classes_example():
    graph = GraphData(
        edge_index=mx.array([[0, 1], [1, 2]]),
        node_labels=mx.array([0, 1, 2]),
        train_mask=mx.array([True, True, True]),
    )
    out = RemoveTrainingClasses([1])(graph)
    assert out.train_mask.tolist() == [True, False, True]
    # The input graph is not modified.
    assert graph.train_mask.tolist() == [True, True, True]


@pytest.mark.parametrize("classes", [[], [0], [1, 3], [0, 1, 2, 3]])
def test_remove_training_classes_matches_pyg(classes):
    pyg_transforms = pytest.importorskip("torch_geometric.transforms")
    rng = np.random.default_rng(0)
    graph = GraphData(
        edge_index=mx.array([[0, 1], [1, 2]]),
        node_labels=mx.array(rng.integers(0, 4, size=20)),
        train_mask=mx.array(rng.random(20) < 0.6),
        num_nodes=20,
    )
    expected = pyg_transforms.RemoveTrainingClasses(classes)(to_pyg(graph))
    out = RemoveTrainingClasses(classes)(graph)
    assert np.array_equal(np.asarray(out.train_mask), expected.train_mask.numpy())


def test_remove_training_classes_requires_mask():
    graph = GraphData(edge_index=mx.array([[0], [1]]), node_labels=mx.array([0, 1]))
    with pytest.raises(ValueError):
        RemoveTrainingClasses([0])(graph)
