import mlx.core as mx
import numpy as np
import pytest

from mlx_graphs.data import GraphData
from mlx_graphs.utils.pyg import from_pyg, to_pyg

torch = pytest.importorskip("torch")
pyg_data = pytest.importorskip("torch_geometric.data")


def make_pyg_graph():
    """A 5-node PyG graph in which node 4 is isolated."""
    return pyg_data.Data(
        x=torch.arange(15, dtype=torch.float32).reshape(5, 3),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
        edge_attr=torch.arange(8, dtype=torch.float32).reshape(4, 2),
        edge_weight=torch.tensor([0.5, 1.0, 1.5, 2.0]),
        y=torch.tensor([0, 1, 0, 1, 2]),
        train_mask=torch.tensor([True, True, False, False, False]),
        pos=torch.ones(5, 2),
        name="toy",
    )


def test_from_pyg_renames_attributes():
    data = make_pyg_graph()
    graph = from_pyg(data)

    assert isinstance(graph, GraphData)
    assert np.array_equal(np.asarray(graph.edge_index), data.edge_index.numpy())
    assert np.array_equal(np.asarray(graph.node_features), data.x.numpy())
    assert np.array_equal(np.asarray(graph.edge_features), data.edge_attr.numpy())
    assert np.array_equal(np.asarray(graph.edge_weights), data.edge_weight.numpy())
    assert np.array_equal(np.asarray(graph.node_labels), data.y.numpy())
    assert graph.graph_labels is None
    # Other attributes keep their name.
    assert graph.train_mask.dtype == mx.bool_
    assert graph.train_mask.tolist() == [True, True, False, False, False]
    assert graph.pos.shape == (5, 2)
    assert graph.name == "toy"
    # No PyG name is stored (`x`, `edge_attr` and `y` are read-only aliases).
    for key in ("x", "edge_attr", "edge_weight", "y"):
        assert key not in vars(graph)
    assert not hasattr(graph, "edge_weight")


def test_from_pyg_graph_level_labels():
    data = pyg_data.Data(
        x=torch.ones(5, 3),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        y=torch.tensor([1]),
    )
    graph = from_pyg(data)
    assert graph.node_labels is None
    assert graph.graph_labels.tolist() == [1]

    # A 0-dimensional label is a graph label as well.
    data.y = torch.tensor(2.5)
    graph = from_pyg(data)
    assert graph.node_labels is None
    assert graph.graph_labels.item() == 2.5

    # Multi-task graph labels of shape [1, num_tasks].
    data.y = torch.zeros(1, 12)
    assert from_pyg(data).graph_labels.shape == (1, 12)


def test_from_pyg_sets_num_nodes_explicitly():
    # Without node features, the isolated nodes 3 and 4 are only known through
    # the explicit number of nodes.
    data = pyg_data.Data(edge_index=torch.tensor([[0, 1], [1, 2]]), num_nodes=5)
    graph = from_pyg(data)
    assert graph.num_nodes == 5
    assert graph.node_features is None

    graph = from_pyg(make_pyg_graph())
    assert graph.num_nodes == 5
    assert graph.has_isolated_nodes()


def test_from_pyg_dtypes():
    data = pyg_data.Data(
        x=torch.ones(3, 2, dtype=torch.float64),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        y=torch.tensor([0, 1, 0], dtype=torch.int32),
    )
    graph = from_pyg(data)
    assert graph.node_features.dtype == mx.float32  # float64 is not kept
    assert graph.edge_index.dtype == mx.int64
    assert graph.node_labels.dtype == mx.int32


def test_from_pyg_empty_edges():
    data = pyg_data.Data(
        x=torch.ones(3, 2), edge_index=torch.zeros(2, 0, dtype=torch.long)
    )
    graph = from_pyg(data)
    assert graph.edge_index.shape == (2, 0)
    assert graph.num_nodes == 3


def test_from_pyg_invalid_inputs():
    with pytest.raises(TypeError):
        from_pyg({"edge_index": torch.zeros(2, 0)})
    with pytest.raises(TypeError):
        from_pyg(pyg_data.HeteroData())
    with pytest.raises(ValueError):
        from_pyg(pyg_data.Data(x=torch.ones(3, 2)))  # no edge_index


def test_to_pyg_renames_attributes():
    graph = GraphData(
        edge_index=mx.array([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=mx.uint32),
        node_features=mx.ones((5, 3)),
        edge_features=mx.ones((4, 2)),
        node_labels=mx.array([0, 1, 0, 1, 2], dtype=mx.int32),
        edge_weights=mx.array([0.5, 1.0, 1.5, 2.0]),
        train_mask=mx.array([True, True, False, False, False]),
        name="toy",
    )
    data = to_pyg(graph)

    assert isinstance(data, pyg_data.Data)
    assert data.edge_index.dtype == torch.long
    assert data.edge_index.tolist() == [[0, 1, 2, 3], [1, 2, 3, 0]]
    assert data.x.shape == (5, 3) and data.x.dtype == torch.float32
    assert data.edge_attr.shape == (4, 2)
    assert data.edge_weight.tolist() == [0.5, 1.0, 1.5, 2.0]
    assert data.y.dtype == torch.long
    assert data.y.tolist() == [0, 1, 0, 1, 2]
    assert data.train_mask.dtype == torch.bool
    assert data.name == "toy"
    assert data.num_nodes == 5
    assert "node_features" not in data and "node_labels" not in data
    data.validate(raise_on_error=True)


def test_to_pyg_labels():
    edge_index = mx.array([[0, 1], [1, 2]])

    graph = GraphData(edge_index, mx.ones((3, 2)), graph_labels=mx.array([1]))
    data = to_pyg(graph)
    assert data.y.tolist() == [1]
    assert "graph_labels" not in data

    # With both kinds of labels, `y` holds the node labels.
    graph = GraphData(
        edge_index,
        mx.ones((3, 2)),
        node_labels=mx.array([0, 1, 0]),
        graph_labels=mx.array([1]),
    )
    data = to_pyg(graph)
    assert data.y.tolist() == [0, 1, 0]
    assert data.graph_labels.tolist() == [1]

    graph = GraphData(edge_index, mx.ones((3, 2)))
    assert to_pyg(graph).y is None


def test_to_pyg_num_nodes():
    # Explicit count with isolated nodes.
    graph = GraphData(edge_index=mx.array([[0, 1], [1, 2]]), num_nodes=6)
    data = to_pyg(graph)
    assert data.num_nodes == 6
    assert "_num_nodes" not in data

    # Without an explicit count, skipped node ids are still covered.
    graph = GraphData(edge_index=mx.array([[0, 5], [5, 0]]))
    assert to_pyg(graph).num_nodes == 6

    graph = GraphData(edge_index=mx.zeros((2, 0), dtype=mx.int32), num_nodes=2)
    data = to_pyg(graph)
    assert data.edge_index.shape == (2, 0)
    assert data.num_nodes == 2


def test_round_trip_from_pyg():
    data = make_pyg_graph()
    back = to_pyg(from_pyg(data))

    assert sorted(back.keys()) == sorted(data.keys() + ["num_nodes"])
    for key in data.keys():
        if isinstance(data[key], torch.Tensor):
            assert back[key].dtype == data[key].dtype, key
            assert torch.equal(back[key], data[key]), key
        else:
            assert back[key] == data[key], key
    assert back.num_nodes == data.num_nodes


def test_round_trip_from_mlx():
    graph = GraphData(
        edge_index=mx.array([[0, 1, 2], [1, 2, 0]], dtype=mx.int64),
        node_features=mx.random.normal((4, 3)),
        edge_features=mx.random.normal((3, 2)),
        graph_labels=mx.array([[1.0, 0.0]]),
        val_mask=mx.array([False, True, False, True]),
    )
    back = from_pyg(to_pyg(graph))

    assert back.num_nodes == 4
    assert back.node_labels is None
    original, converted = graph.to_dict(), back.to_dict()
    converted.pop("_num_nodes")
    assert sorted(original) == sorted(converted)
    for key, value in original.items():
        assert converted[key].dtype == value.dtype, key
        assert mx.array_equal(converted[key], value), key


def test_from_pyg_dataset_graph():
    datasets = pytest.importorskip("torch_geometric.datasets")
    data = datasets.KarateClub()[0]
    graph = from_pyg(data)
    assert graph.num_nodes == 34
    assert graph.edge_index.shape == (2, 156)
    assert graph.node_features.shape == (34, 34)
    assert graph.node_labels.shape == (34,)
    assert graph.train_mask.shape == (34,)
    assert graph.is_undirected()
