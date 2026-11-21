import pickle

import mlx.core as mx
import numpy as np
import pytest

from mlx_graphs.data import GraphData, GraphDataBatch, HeteroGraphData
from mlx_graphs.utils import from_pyg, to_pyg


def make_graph(**kwargs) -> GraphData:
    return GraphData(
        edge_index=mx.array([[0, 1, 1, 2, 3], [1, 0, 2, 3, 0]]),
        node_features=mx.arange(8).reshape(4, 2).astype(mx.float32),
        edge_features=mx.arange(5).reshape(5, 1).astype(mx.float32),
        node_labels=mx.array([0, 1, 0, 1]),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Aliases
# ---------------------------------------------------------------------------


def test_x_and_edge_attr_aliases():
    graph = make_graph()
    assert graph.x is graph.node_features
    assert graph.edge_attr is graph.edge_features

    graph.x = mx.ones((4, 3))
    graph.edge_attr = mx.zeros((5, 2))
    assert graph.node_features.shape == (4, 3)
    assert graph.edge_features.shape == (5, 2)
    assert graph.num_features == graph.num_node_features == 3


def test_aliases_in_constructor():
    graph = GraphData(
        edge_index=mx.array([[0, 1], [1, 2]]),
        x=mx.ones((3, 2)),
        edge_attr=mx.ones((2, 4)),
        y=mx.array([0, 1, 0]),
    )
    assert graph.node_features.shape == (3, 2)
    assert graph.edge_features.shape == (2, 4)
    assert graph.node_labels.tolist() == [0, 1, 0]
    assert graph.graph_labels is None
    # Aliases are not stored separately.
    assert set(vars(graph)) == {
        "edge_index",
        "node_features",
        "edge_features",
        "graph_features",
        "node_labels",
        "edge_labels",
        "graph_labels",
    }


def test_y_alias():
    graph = GraphData(edge_index=mx.array([[0, 1], [1, 2]]), num_nodes=3)
    assert graph.y is None

    # One entry per node: node labels.
    graph.y = mx.array([1, 2, 3])
    assert graph.node_labels.tolist() == [1, 2, 3]
    assert graph.y is graph.node_labels

    # Existing node labels are replaced, whatever the shape.
    graph.y = mx.array([4, 5, 6])
    assert graph.node_labels.tolist() == [4, 5, 6]
    graph.y = None
    assert graph.node_labels is None and graph.y is None

    # Otherwise: graph labels.
    graph.y = mx.array([7])
    assert graph.graph_labels.tolist() == [7]
    assert graph.node_labels is None
    assert graph.y is graph.graph_labels

    # Node labels take precedence when reading.
    graph.node_labels = mx.array([0, 0, 0])
    assert graph.y is graph.node_labels


def test_aliases_do_not_break_repr_to_dict_and_pickle():
    graph = make_graph()
    graph.x = mx.ones((4, 2))
    assert "\tx(" not in repr(graph) and "node_features" in repr(graph)
    assert "x" not in graph.to_dict() and "y" not in graph.to_dict()

    restored = pickle.loads(pickle.dumps(graph))
    assert mx.array_equal(restored.x, graph.x)
    assert mx.array_equal(restored.y, graph.y)


def test_aliases_with_batching():
    graphs = [make_graph(), make_graph()]
    batch = GraphDataBatch(graphs)
    assert batch.x.shape == (8, 2)
    assert batch.y.tolist() == [0, 1, 0, 1, 0, 1, 0, 1]
    assert batch.edge_attr.shape == (10, 1)
    assert batch[1].x.shape == (4, 2)


def test_aliases_with_pyg_conversion():
    pytest.importorskip("torch_geometric")
    graph = make_graph()
    data = to_pyg(graph)
    assert np.array_equal(data.x.numpy(), np.asarray(graph.x))
    assert np.array_equal(data.y.numpy(), np.asarray(graph.y))
    back = from_pyg(data)
    assert mx.array_equal(back.x, graph.x)
    assert mx.array_equal(back.edge_attr, graph.edge_attr)
    assert mx.array_equal(back.y, graph.y)


def test_y_in_constructor_is_resolved_after_other_arguments():
    # Node 2 and 3 are isolated: the number of nodes comes from `x` or
    # `num_nodes`, whatever the order of the keyword arguments.
    edge_index = mx.array([[0, 1], [1, 0]])
    y = mx.array([0, 1, 0, 1])
    for graph in (
        GraphData(edge_index, y=y, x=mx.ones((4, 2))),
        GraphData(edge_index, y=y, num_nodes=4),
    ):
        assert graph.node_labels.tolist() == [0, 1, 0, 1]
        assert graph.graph_labels is None


def test_y_of_single_node_graphs_is_a_graph_label():
    # Graph classification with PyG names: a single-node graph must get
    # graph labels like the others, or the graphs cannot be batched.
    def make(num_nodes, label):
        src = list(range(num_nodes - 1))
        return GraphData(
            edge_index=mx.array([src, [i + 1 for i in src]], dtype=mx.int32),
            x=mx.ones((num_nodes, 2)),
            y=mx.array([label]),
        )

    graphs = [make(3, 0), make(1, 1), make(2, 0)]
    assert all(g.node_labels is None for g in graphs)
    assert GraphDataBatch(graphs).y.tolist() == [0, 1, 0]


def test_from_pyg_y_of_single_node_graph_is_a_graph_label():
    torch = pytest.importorskip("torch")
    pyg_data = pytest.importorskip("torch_geometric.data")
    data = pyg_data.Data(
        x=torch.ones(1, 2),
        edge_index=torch.zeros(2, 0, dtype=torch.long),
        y=torch.tensor([1]),
    )
    graph = from_pyg(data)
    assert graph.graph_labels.tolist() == [1] and graph.node_labels is None


def test_unpickling_graphs_with_pyg_named_attributes():
    # Before `x`, `edge_attr` and `y` were aliases, they were stored as custom
    # attributes: such pickles must not be hidden by the aliases.
    graph = GraphData(edge_index=mx.array([[0, 1, 2], [1, 2, 0]]))
    graph.__dict__.update(
        x=mx.ones((4, 3)), y=mx.array([0, 1, 0, 1]), edge_attr=mx.ones((3, 2))
    )
    restored = pickle.loads(pickle.dumps(graph))
    assert "x" not in vars(restored) and "y" not in vars(restored)
    assert restored.node_features.shape == (4, 3)
    assert restored.edge_features.shape == (3, 2)
    assert restored.node_labels.tolist() == [0, 1, 0, 1]
    batch = GraphDataBatch([restored, pickle.loads(pickle.dumps(graph))])
    assert batch.x.shape == (8, 3) and batch.edge_index[0, 3].item() == 4

    # Both names set: the old custom attribute is kept under another name.
    graph = make_graph()
    graph.__dict__["x"] = mx.zeros((4, 7))
    with pytest.warns(UserWarning):
        restored = pickle.loads(pickle.dumps(graph))
    assert restored.x.shape == (4, 2) and restored.x_.shape == (4, 7)


def test_graphs_with_batch_or_ptr_attributes_can_be_batched():
    graphs = [make_graph(batch=mx.zeros(4), ptr=mx.array([0, 4])) for _ in range(2)]
    batch = GraphDataBatch(graphs)
    # The graphs' own attributes take precedence over the PyG aliases.
    assert batch.batch.shape == (8,) and batch.ptr.tolist() == [0, 4, 0, 4]
    assert batch[1].ptr.tolist() == [0, 4]
    assert batch.batch_indices.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]


# ---------------------------------------------------------------------------
# Dictionary-like access
# ---------------------------------------------------------------------------


def test_keys_contains_getitem_setitem_delitem():
    graph = make_graph(train_mask=mx.array([True, False, True, False]))
    assert graph.keys() == [
        "edge_index",
        "node_features",
        "edge_features",
        "node_labels",
        "train_mask",
    ]
    assert "node_features" in graph and "x" in graph and "y" in graph
    assert "graph_labels" not in graph and "missing" not in graph
    assert graph["x"] is graph.node_features
    assert graph["train_mask"].tolist() == [True, False, True, False]
    with pytest.raises(KeyError):
        graph["graph_labels"]
    with pytest.raises(TypeError):
        graph[0]

    graph["val_mask"] = mx.array([False] * 4)
    assert "val_mask" in graph.keys()
    del graph["val_mask"]
    assert "val_mask" not in graph
    del graph["x"]
    assert graph.node_features is None
    with pytest.raises(KeyError):
        del graph["missing"]

    graph.num_nodes = 4
    assert "num_nodes" in graph.keys() and graph["num_nodes"] == 4


def test_batch_string_keys():
    batch = GraphDataBatch([make_graph(), make_graph()])
    assert batch["node_features"].shape == (8, 2)
    assert isinstance(batch[0], GraphData)


def test_clone():
    graph = make_graph(train_mask=mx.array([True, False, True, False]))
    clone = graph.clone()
    clone.node_features[0] = 100.0
    clone.train_mask[0] = False
    assert graph.node_features[0, 0].item() == 0.0
    assert graph.train_mask[0].item()
    assert mx.array_equal(clone.edge_index, graph.edge_index)


# ---------------------------------------------------------------------------
# Node and edge attributes, subgraphs
# ---------------------------------------------------------------------------


def test_is_node_and_edge_attr():
    graph = make_graph(
        train_mask=mx.ones((4,), dtype=mx.bool_), edge_weights=mx.ones(5)
    )
    assert graph.is_node_attr("node_features") and graph.is_node_attr("x")
    assert graph.is_node_attr("train_mask")
    assert graph.is_edge_attr("edge_attr") and graph.is_edge_attr("edge_weights")
    assert not graph.is_edge_attr("edge_index")
    assert not graph.is_node_attr("edge_features")


@pytest.mark.parametrize(
    "subset",
    [
        [1, 2, 3],
        [3, 1, 0],
        [True, False, True, True],
        mx.array([2, 0]),
        np.array([0, 1, 2, 3]),
        [],
    ],
)
def test_subgraph_matches_pyg(subset):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    graph = make_graph(train_mask=mx.array([True, False, True, False]))
    sub = graph.subgraph(subset)

    torch_subset = torch.as_tensor(np.asarray(subset))
    if torch_subset.dtype != torch.bool:
        torch_subset = torch_subset.long()
    expected = to_pyg(graph).subgraph(torch_subset)

    assert sub.num_nodes == expected.num_nodes
    assert np.array_equal(np.asarray(sub.edge_index), expected.edge_index.numpy())
    assert np.array_equal(np.asarray(sub.x), expected.x.numpy())
    assert np.array_equal(np.asarray(sub.edge_attr), expected.edge_attr.numpy())
    assert np.array_equal(np.asarray(sub.y), expected.y.numpy())
    assert np.array_equal(np.asarray(sub.train_mask), expected.train_mask.numpy())
    # The original graph is unchanged.
    assert graph.num_nodes == 4 and graph.num_edges == 5


def test_subgraph_without_node_features_keeps_isolated_nodes():
    graph = GraphData(edge_index=mx.array([[0, 1], [1, 2]]), num_nodes=5)
    sub = graph.subgraph([0, 3, 4])
    assert sub.num_nodes == 3
    assert sub.edge_index.shape == (2, 0)


@pytest.mark.parametrize(
    "subset", [[0, 3], [4, 0, 2], [True, False, False, True, True], []]
)
def test_edge_subgraph_matches_pyg(subset):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    graph = make_graph(edge_weights=mx.arange(5).astype(mx.float32) / 10)
    sub = graph.edge_subgraph(subset)

    torch_subset = torch.as_tensor(np.asarray(subset))
    if torch_subset.dtype != torch.bool:
        torch_subset = torch_subset.long()
    expected = to_pyg(graph).edge_subgraph(torch_subset)

    assert sub.num_nodes == 4
    assert np.array_equal(np.asarray(sub.edge_index), expected.edge_index.numpy())
    assert np.array_equal(np.asarray(sub.edge_attr), expected.edge_attr.numpy())
    assert np.allclose(np.asarray(sub.edge_weights), expected.edge_weight.numpy())
    assert np.array_equal(np.asarray(sub.x), expected.x.numpy())


def test_edge_subgraph_out_of_range():
    with pytest.raises(IndexError):
        make_graph().edge_subgraph([5])


# ---------------------------------------------------------------------------
# Validation and coalescing
# ---------------------------------------------------------------------------


def test_validate():
    assert make_graph().validate()
    assert GraphData(edge_index=mx.array([[0, 7], [7, 0]])).validate()

    bad_graphs = [
        GraphData(edge_index=mx.array([0, 1, 2])),
        GraphData(edge_index=mx.array([[0.0, 1.0], [1.0, 0.0]])),
        GraphData(edge_index=mx.array([[0, -1], [1, 0]])),
        GraphData(edge_index=mx.array([[0, 1], [1, 4]]), node_features=mx.ones((4, 1))),
        GraphData(edge_index=mx.array([[0, 1], [1, 4]]), num_nodes=3),
        GraphData(edge_index=mx.array([[0, 1], [1, 0]]), edge_features=mx.ones((3, 1))),
        GraphData(
            edge_index=mx.array([[0, 1], [1, 0]]),
            node_labels=mx.ones((3,)),
            num_nodes=2,
        ),
    ]
    for graph in bad_graphs:
        with pytest.raises(ValueError):
            graph.validate()
        with pytest.warns(UserWarning):
            assert not graph.validate(raise_on_error=False)


def test_coalesce_matches_pyg():
    pytest.importorskip("torch_geometric")
    rng = np.random.default_rng(0)
    edges = rng.integers(0, 6, size=(2, 20))
    graph = GraphData(
        edge_index=mx.array(edges),
        edge_features=mx.array(rng.normal(size=(20, 2)).astype(np.float32)),
        num_nodes=6,
    )
    assert not graph.is_coalesced()
    coalesced = graph.coalesce()
    expected = to_pyg(graph).coalesce()

    assert coalesced.is_coalesced()
    assert to_pyg(coalesced).is_coalesced()
    assert np.array_equal(np.asarray(coalesced.edge_index), expected.edge_index.numpy())
    assert np.allclose(
        np.asarray(coalesced.edge_attr), expected.edge_attr.numpy(), atol=1e-6
    )
    assert coalesced.num_nodes == 6
    # The original graph is unchanged, and a coalesced graph is returned as is.
    assert graph.num_edges == 20
    assert coalesced.coalesce() is coalesced


def test_is_coalesced():
    assert GraphData(edge_index=mx.array([[0, 0, 1], [1, 2, 0]])).is_coalesced()
    assert not GraphData(edge_index=mx.array([[0, 0, 1], [2, 1, 0]])).is_coalesced()
    assert not GraphData(edge_index=mx.array([[0, 0], [1, 1]])).is_coalesced()
    assert GraphData(edge_index=mx.zeros((2, 0), dtype=mx.int32)).is_coalesced()


# ---------------------------------------------------------------------------
# GraphDataBatch
# ---------------------------------------------------------------------------


def test_batch_and_ptr_aliases():
    graphs = [
        GraphData(edge_index=mx.array([[0], [1]]), node_features=mx.ones((2, 1))),
        GraphData(edge_index=mx.array([[0], [2]]), node_features=mx.ones((4, 1))),
        GraphData(edge_index=mx.array([[0], [1]]), node_features=mx.ones((3, 1))),
    ]
    batch = GraphDataBatch(graphs)
    assert batch.batch.tolist() == [0, 0, 1, 1, 1, 1, 2, 2, 2]
    assert mx.array_equal(batch.batch, batch.batch_indices)
    assert batch.ptr.tolist() == [0, 2, 6, 9]


def test_batch_and_ptr_match_pyg():
    pytest.importorskip("torch_geometric")
    from torch_geometric.data import Batch

    graphs = [make_graph(), make_graph(), make_graph()]
    batch = GraphDataBatch(graphs)
    expected = Batch.from_data_list([to_pyg(g) for g in graphs])
    assert np.array_equal(np.asarray(batch.batch), expected.batch.numpy())
    assert np.array_equal(np.asarray(batch.ptr), expected.ptr.numpy())


# ---------------------------------------------------------------------------
# HeteroGraphData
# ---------------------------------------------------------------------------


def make_hetero_graph() -> HeteroGraphData:
    return HeteroGraphData(
        edge_index_dict={
            ("author", "writes", "paper"): mx.array([[0, 1, 2], [0, 0, 1]]),
            ("paper", "cites", "paper"): mx.array([[0, 1], [1, 0]]),
            ("paper", "rev_writes", "author"): mx.array([[0, 0, 1], [0, 1, 2]]),
        },
        node_features_dict={
            "author": mx.arange(6).reshape(3, 2).astype(mx.float32),
            "paper": mx.arange(4).reshape(2, 2).astype(mx.float32) + 10,
        },
        edge_features_dict={
            ("author", "writes", "paper"): mx.ones((3, 1)),
            ("paper", "rev_writes", "author"): mx.zeros((3, 1)),
        },
        node_labels_dict={"paper": mx.array([1, 2])},
        graph_labels=mx.array([1]),
    )


def test_hetero_metadata_and_edge_types():
    data = make_hetero_graph()
    node_types, edge_types = data.metadata()
    assert node_types == ["author", "paper"]
    assert edge_types == [
        ("author", "writes", "paper"),
        ("paper", "cites", "paper"),
        ("paper", "rev_writes", "author"),
    ]
    assert data.edge_types == edge_types


def test_to_homogeneous():
    data = make_hetero_graph()
    graph = data.to_homogeneous()
    assert graph.num_nodes == 5
    assert graph.edge_index.tolist() == [
        [0, 1, 2, 3, 4, 3, 3, 4],
        [3, 3, 4, 4, 3, 0, 1, 2],
    ]
    assert graph.node_type.tolist() == [0, 0, 0, 1, 1]
    assert graph.edge_type.tolist() == [0, 0, 0, 1, 1, 2, 2, 2]
    assert graph.node_features.shape == (5, 2)
    # Missing values are filled with dummy values.
    assert graph.node_labels.tolist() == [-1, -1, -1, 1, 2]
    edge_features = np.asarray(graph.edge_features).reshape(-1)
    assert np.isnan(edge_features[3:5]).all()
    assert edge_features[[0, 1, 2]].tolist() == [1, 1, 1]
    assert graph.graph_labels.tolist() == [1]
    graph.validate()

    no_dummy = data.to_homogeneous(dummy_values=False, add_edge_type=False)
    assert no_dummy.node_labels is None and no_dummy.edge_features is None
    assert not hasattr(no_dummy, "edge_type")


def test_to_homogeneous_with_empty_edge_labels_index_dict():
    data = make_hetero_graph()
    expected = data.to_homogeneous()
    data.edge_labels_index_dict = {}
    graph = data.to_homogeneous()
    assert not hasattr(graph, "edge_labels_index")
    assert graph.edge_index.tolist() == expected.edge_index.tolist()


def test_to_homogeneous_matches_pyg():
    torch = pytest.importorskip("torch")
    pyg_data = pytest.importorskip("torch_geometric.data")
    data = make_hetero_graph()

    hetero = pyg_data.HeteroData()
    for node_type, x in data.node_features_dict.items():
        hetero[node_type].x = torch.from_numpy(np.asarray(x))
    hetero["paper"].y = torch.from_numpy(np.asarray(data.node_labels_dict["paper"]))
    for edge_type, edge_index in data.edge_index_dict.items():
        hetero[edge_type].edge_index = torch.from_numpy(
            np.asarray(edge_index).astype(np.int64)
        )
    for edge_type, edge_attr in data.edge_features_dict.items():
        hetero[edge_type].edge_attr = torch.from_numpy(np.asarray(edge_attr))
    expected = hetero.to_homogeneous()

    graph = data.to_homogeneous()
    assert np.array_equal(np.asarray(graph.edge_index), expected.edge_index.numpy())
    assert np.array_equal(np.asarray(graph.node_type), expected.node_type.numpy())
    assert np.array_equal(np.asarray(graph.edge_type), expected.edge_type.numpy())
    assert np.array_equal(np.asarray(graph.x), expected.x.numpy())
    assert np.array_equal(np.asarray(graph.node_labels), expected.y.numpy())
    assert np.allclose(
        np.asarray(graph.edge_attr), expected.edge_attr.numpy(), equal_nan=True
    )


def test_to_homogeneous_pads_feature_widths_and_edge_labels_index():
    data = HeteroGraphData(
        edge_index_dict={("a", "to", "b"): mx.array([[0], [1]])},
        node_features_dict={"a": mx.ones((1, 3)), "b": mx.ones((2, 1))},
        edge_labels_index_dict={("a", "to", "b"): mx.array([[0, 0], [0, 1]])},
        edge_labels_dict={("a", "to", "b"): mx.array([1, 0])},
    )
    graph = data.to_homogeneous()
    assert graph.node_features.tolist() == [[1, 1, 1], [1, 0, 0], [1, 0, 0]]
    assert graph.edge_labels_index.tolist() == [[0, 0], [1, 2]]
    assert graph.edge_labels.tolist() == [1, 0]


def test_to_homogeneous_type_vectors_are_int32():
    graph = make_hetero_graph().to_homogeneous()
    assert graph.node_type.dtype == mx.int32
    assert graph.edge_type.dtype == mx.int32
    empty = HeteroGraphData(
        edge_index_dict={}, node_features_dict={"a": mx.ones((2, 1))}
    ).to_homogeneous()
    assert empty.edge_index.dtype == mx.int32 and empty.edge_index.shape == (2, 0)
