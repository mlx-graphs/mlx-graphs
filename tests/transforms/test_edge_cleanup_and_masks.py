import mlx.core as mx
import numpy as np
import pytest

from mlx_graphs.data import GraphData
from mlx_graphs.transforms import (
    AddRemainingSelfLoops,
    IndexToMask,
    MaskToIndex,
    RemoveDuplicatedEdges,
)


def random_graph(seed: int, num_nodes: int = 12, num_edges: int = 30) -> GraphData:
    """Random graph with duplicated edges and self-loops; the last 2 nodes are
    isolated."""
    rng = np.random.default_rng(seed)
    edges = rng.integers(0, num_nodes - 2, size=(2, num_edges)).astype(np.int32)
    return GraphData(
        edge_index=mx.array(edges),
        node_features=mx.array(rng.normal(size=(num_nodes, 3)).astype(np.float32)),
        edge_features=mx.array(rng.normal(size=(num_edges, 2)).astype(np.float32)),
        edge_weights=mx.array(rng.random(num_edges).astype(np.float32)),
        edge_labels=mx.arange(num_edges),
    )


def test_repr():
    assert str(AddRemainingSelfLoops()) == (
        "AddRemainingSelfLoops(attr='edge_weights', fill_value=1.0)"
    )
    assert (
        str(RemoveDuplicatedEdges()) == "RemoveDuplicatedEdges(keys=None, reduce='add')"
    )
    assert str(IndexToMask("train_index", 5)) == (
        "IndexToMask(attrs=['train_index'], sizes=5, replace=False)"
    )
    assert str(MaskToIndex()) == "MaskToIndex(attrs=None, replace=False)"


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("fill_value", [1.0, 0.5, "mean", "add"])
def test_add_remaining_self_loops_matches_pyg(seed, fill_value):
    torch = pytest.importorskip("torch")
    pyg_transforms = pytest.importorskip("torch_geometric.transforms")
    from torch_geometric.data import Data

    graph = random_graph(seed)
    out = AddRemainingSelfLoops(fill_value=fill_value)(graph)

    data = Data(
        x=torch.tensor(np.asarray(graph.node_features)),
        edge_index=torch.tensor(np.asarray(graph.edge_index)).long(),
        edge_weight=torch.tensor(np.asarray(graph.edge_weights)),
    )
    expected = pyg_transforms.AddRemainingSelfLoops(
        "edge_weight", fill_value=fill_value
    )(data)
    assert np.array_equal(np.asarray(out.edge_index), expected.edge_index.numpy())
    assert np.allclose(np.asarray(out.edge_weights), expected.edge_weight.numpy())
    # The other per-edge attributes follow the edges.
    assert out.edge_features.shape[0] == out.num_edges
    assert out.edge_labels.shape[0] == out.num_edges
    # The input graph is unchanged.
    assert graph.edge_index.shape[1] == 30


def test_add_remaining_self_loops_without_attr():
    graph = GraphData(edge_index=mx.array([[0, 1, 1], [1, 0, 1]]), num_nodes=3)
    out = AddRemainingSelfLoops()(graph)
    assert out.edge_index.tolist() == [[0, 1, 0, 1, 2], [1, 0, 0, 1, 2]]
    assert getattr(out, "edge_weights", None) is None
    assert out.num_nodes == 3


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("reduce", ["add", "mean", "max"])
def test_remove_duplicated_edges_matches_pyg(seed, reduce):
    torch = pytest.importorskip("torch")
    pyg_transforms = pytest.importorskip("torch_geometric.transforms")
    from torch_geometric.data import Data

    graph = random_graph(seed)
    out = RemoveDuplicatedEdges(reduce=reduce)(graph)

    data = Data(
        x=torch.tensor(np.asarray(graph.node_features)),
        edge_index=torch.tensor(np.asarray(graph.edge_index)).long(),
        edge_attr=torch.tensor(np.asarray(graph.edge_features)),
        edge_weight=torch.tensor(np.asarray(graph.edge_weights)),
    )
    expected = pyg_transforms.RemoveDuplicatedEdges(reduce=reduce)(data)
    assert np.array_equal(np.asarray(out.edge_index), expected.edge_index.numpy())
    assert np.allclose(np.asarray(out.edge_features), expected.edge_attr.numpy())
    assert np.allclose(np.asarray(out.edge_weights), expected.edge_weight.numpy())
    assert out.edge_labels.shape[0] == out.num_edges


def test_remove_duplicated_edges_with_keys():
    graph = GraphData(
        edge_index=mx.array([[0, 1, 0], [1, 2, 1]]),
        edge_weights=mx.array([1.0, 2.0, 3.0]),
        edge_features=mx.array([[1.0], [2.0], [3.0]]),
    )
    out = RemoveDuplicatedEdges("edge_weights", reduce="max")(graph)
    assert out.edge_index.tolist() == [[0, 1], [1, 2]]
    assert out.edge_weights.tolist() == [3.0, 2.0]
    # Attributes that are not listed are left untouched, as in PyG.
    assert out.edge_features.shape == (3, 1)


def test_index_to_mask_and_back():
    graph = GraphData(
        edge_index=mx.array([[0, 1, 2], [1, 2, 3]]),
        node_features=mx.ones((5, 1)),
        train_index=mx.array([0, 2]),
        test_index=mx.array([4]),
        edge_label_index=mx.array([1]),
    )
    out = IndexToMask()(graph)
    assert out.train_mask.tolist() == [True, False, True, False, False]
    assert out.test_mask.tolist() == [False, False, False, False, True]
    assert out.edge_label_mask.tolist() == [False, True, False]  # edge-level
    assert hasattr(out, "train_index")
    assert not hasattr(graph, "train_mask")

    back = MaskToIndex(replace=True)(IndexToMask(replace=True)(graph))
    assert back.train_index.tolist() == [0, 2]
    assert back.test_index.tolist() == [4]
    assert not hasattr(back, "train_mask")
    assert hasattr(graph, "train_index")

    out = IndexToMask(["train_index", "test_index"], sizes=[6, 7])(graph)
    assert out.train_mask.shape == (6,) and out.test_mask.shape == (7,)
    out = IndexToMask("train_index", sizes=3)(graph)
    assert out.train_mask.tolist() == [True, False, True]
    assert not hasattr(out, "test_mask")
    with pytest.raises(ValueError):
        IndexToMask(sizes=[1])(graph)


def test_mask_to_index():
    graph = GraphData(
        edge_index=mx.array([[0, 1, 2], [1, 2, 3]]),
        train_mask=mx.array([True, False, True, False]),
        val_mask=mx.array([False, True, False, False]),
    )
    out = MaskToIndex("train_mask")(graph)
    assert out.train_index.tolist() == [0, 2]
    assert not hasattr(out, "val_index")
    out = MaskToIndex()(graph)
    assert out.val_index.tolist() == [1]
    assert out.train_mask is graph.train_mask
