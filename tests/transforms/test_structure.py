import mlx.core as mx
import numpy as np
import pytest

from mlx_graphs.data import GraphData
from mlx_graphs.transforms.structure import (
    AddSelfLoops,
    LargestConnectedComponents,
    RemoveIsolatedNodes,
    RemoveSelfLoops,
    ToUndirected,
    _attribute_names,
)
from mlx_graphs.utils.topology import is_undirected


def random_edges(seed: int, num_nodes: int = 12, num_edges: int = 30) -> np.ndarray:
    """Random edges with duplicates and self-loops."""
    return np.random.default_rng(seed).integers(0, num_nodes, size=(2, num_edges))


def random_graph(seed: int, num_nodes: int = 15, num_edges: int = 30) -> GraphData:
    """Random graph whose last 3 nodes are isolated, with node and edge
    attributes that identify each node and edge."""
    edges = random_edges(seed, num_nodes - 3, num_edges)
    rng = np.random.default_rng(1000 + seed)
    return GraphData(
        edge_index=mx.array(edges),
        node_features=mx.array(rng.normal(size=(num_nodes, 4)).astype(np.float32)),
        edge_features=mx.array(rng.normal(size=(num_edges, 2)).astype(np.float32)),
        node_labels=mx.arange(num_nodes),
        edge_labels=mx.arange(num_edges),
        train_mask=mx.array(rng.random(num_nodes) < 0.5),
        graph_labels=mx.array([1]),
    )


def test_repr():
    assert str(ToUndirected()) == "ToUndirected(reduce='add')"
    assert str(AddSelfLoops(2.0)) == "AddSelfLoops(fill_value=2.0)"
    assert str(RemoveSelfLoops()) == "RemoveSelfLoops()"
    assert str(RemoveIsolatedNodes()) == "RemoveIsolatedNodes()"
    assert str(LargestConnectedComponents(2)) == "LargestConnectedComponents(2)"


@pytest.mark.parametrize(
    "transform",
    [
        ToUndirected(),
        AddSelfLoops(),
        RemoveSelfLoops(),
        RemoveIsolatedNodes(),
        LargestConnectedComponents(),
    ],
)
def test_transforms_do_not_modify_their_input(transform):
    graph = random_graph(0)
    before = {key: value for key, value in graph.to_dict().items()}

    transformed = transform(graph)

    assert transformed is not graph
    after = graph.to_dict()
    assert list(after) == list(before)
    for key, value in before.items():
        assert after[key] is value, key


def test_attribute_names():
    graph = random_graph(0)
    graph.edge_label_index = mx.zeros((2, 30), dtype=mx.int32)
    graph.edge_label = mx.zeros((30,))
    graph.name = "toy"
    node_keys, edge_keys = _attribute_names(graph, graph.num_nodes)
    assert node_keys == ["node_features", "node_labels", "train_mask"]
    assert edge_keys == ["edge_features", "edge_labels"]

    # As many nodes as edges: only the names tell node and edge attributes apart.
    graph = GraphData(
        edge_index=mx.array([[0, 1, 2], [1, 2, 0]]),
        node_features=mx.ones((3, 2)),
        edge_features=mx.ones((3, 2)),
        train_mask=mx.array([True, False, True]),
        edge_weights=mx.ones((3,)),
    )
    node_keys, edge_keys = _attribute_names(graph, 3)
    assert node_keys == ["node_features", "train_mask"]
    assert edge_keys == ["edge_features", "edge_weights"]


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("reduce", ["add", "mean", "max", "min"])
def test_to_undirected_matches_pyg(seed, reduce):
    torch = pytest.importorskip("torch")
    pyg_data = pytest.importorskip("torch_geometric.data")
    pyg_transforms = pytest.importorskip("torch_geometric.transforms")

    graph = random_graph(seed)
    graph.edge_weights = mx.array(
        np.random.default_rng(seed).uniform(size=30).astype(np.float32)
    )
    del graph.edge_labels  # integer attributes do not support every reduction
    expected = pyg_transforms.ToUndirected(reduce=reduce)(
        pyg_data.Data(
            edge_index=torch.from_numpy(np.asarray(graph.edge_index)),
            edge_attr=torch.from_numpy(np.asarray(graph.edge_features)),
            edge_weight=torch.from_numpy(np.asarray(graph.edge_weights)),
            num_nodes=graph.num_nodes,
        )
    )

    undirected = ToUndirected(reduce=reduce)(graph)

    assert np.array_equal(np.asarray(undirected.edge_index), expected.edge_index)
    assert np.allclose(
        np.asarray(undirected.edge_features), expected.edge_attr.numpy(), atol=1e-6
    )
    assert np.allclose(
        np.asarray(undirected.edge_weights), expected.edge_weight.numpy(), atol=1e-6
    )
    assert is_undirected(undirected.edge_index)
    # Node-level attributes are untouched.
    assert undirected.node_features is graph.node_features
    assert undirected.num_nodes == graph.num_nodes


def test_to_undirected_basic():
    graph = GraphData(
        edge_index=mx.array([[0, 1, 1], [1, 0, 2]]),
        edge_features=mx.array([1.0, 2.0, 5.0]),
    )
    undirected = ToUndirected()(graph)
    assert undirected.edge_index.tolist() == [[0, 1, 1, 2], [1, 0, 2, 1]]
    # (0, 1) and (1, 0) both existed: their features are summed in each direction.
    assert undirected.edge_features.tolist() == [3.0, 3.0, 5.0, 5.0]

    undirected = ToUndirected(reduce="mean")(graph)
    assert undirected.edge_features.tolist() == [1.5, 1.5, 5.0, 5.0]


@pytest.mark.parametrize("dtype", [mx.uint32, mx.int64])
def test_to_undirected_empty_and_dtype(dtype):
    graph = GraphData(
        edge_index=mx.zeros((2, 0), dtype=dtype),
        node_features=mx.ones((3, 2)),
        edge_features=mx.zeros((0, 4)),
    )
    undirected = ToUndirected()(graph)
    assert undirected.edge_index.shape == (2, 0)
    assert undirected.edge_index.dtype == dtype
    assert undirected.edge_features.shape == (0, 4)

    graph = GraphData(edge_index=mx.array([[0, 5], [5, 2]], dtype=dtype))
    undirected = ToUndirected()(graph)
    assert undirected.edge_index.dtype == dtype
    assert undirected.edge_index.tolist() == [[0, 2, 5, 5], [5, 5, 0, 2]]


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("fill_value", [1.0, 0.5])
def test_add_self_loops_matches_pyg(seed, fill_value):
    torch = pytest.importorskip("torch")
    pyg_utils = pytest.importorskip("torch_geometric.utils")

    graph = random_graph(seed)
    expected_index, expected_features = pyg_utils.add_self_loops(
        torch.from_numpy(np.asarray(graph.edge_index)),
        torch.from_numpy(np.asarray(graph.edge_features)),
        fill_value=fill_value,
        num_nodes=graph.num_nodes,
    )

    with_loops = AddSelfLoops(fill_value)(graph)

    assert np.array_equal(np.asarray(with_loops.edge_index), expected_index.numpy())
    assert np.array_equal(
        np.asarray(with_loops.edge_features), expected_features.numpy()
    )
    # Isolated nodes get a self-loop too, and every per-edge attribute follows.
    assert with_loops.num_edges == graph.num_edges + graph.num_nodes
    assert with_loops.edge_labels.shape == (with_loops.num_edges,)
    assert with_loops.edge_labels.dtype == graph.edge_labels.dtype
    # The fill value is cast to the dtype of each attribute (integers here).
    loop_labels = with_loops.edge_labels[graph.num_edges :].tolist()
    assert loop_labels == [int(fill_value)] * graph.num_nodes
    assert with_loops.num_nodes == graph.num_nodes


def test_add_self_loops_without_features():
    graph = GraphData(edge_index=mx.array([[0, 1], [1, 2]], dtype=mx.uint32))
    with_loops = AddSelfLoops()(graph)
    assert with_loops.edge_index.tolist() == [[0, 1, 0, 1, 2], [1, 2, 0, 1, 2]]
    assert with_loops.edge_index.dtype == mx.uint32
    assert with_loops.edge_features is None

    # Skipped node ids and explicit node counts are covered.
    graph = GraphData(edge_index=mx.array([[0], [3]]))
    assert AddSelfLoops()(graph).edge_index.tolist() == [
        [0, 0, 1, 2, 3],
        [3, 0, 1, 2, 3],
    ]
    graph = GraphData(edge_index=mx.zeros((2, 0), dtype=mx.int32), num_nodes=2)
    assert AddSelfLoops()(graph).edge_index.tolist() == [[0, 1], [0, 1]]


@pytest.mark.parametrize("seed", range(4))
def test_remove_self_loops(seed):
    graph = random_graph(seed)
    edges = np.asarray(graph.edge_index)
    keep = edges[0] != edges[1]
    assert 0 < keep.sum() < graph.num_edges

    no_loops = RemoveSelfLoops()(graph)

    assert np.array_equal(np.asarray(no_loops.edge_index), edges[:, keep])
    assert np.array_equal(
        np.asarray(no_loops.edge_features), np.asarray(graph.edge_features)[keep]
    )
    assert no_loops.edge_labels.tolist() == np.nonzero(keep)[0].tolist()
    assert not no_loops.has_self_loops()
    assert no_loops.node_features is graph.node_features
    assert no_loops.num_nodes == graph.num_nodes


def test_remove_self_loops_edge_cases():
    # Node 2 only has a self-loop: it must still be counted afterwards.
    graph = GraphData(edge_index=mx.array([[0, 2], [1, 2]], dtype=mx.uint32))
    no_loops = RemoveSelfLoops()(graph)
    assert no_loops.edge_index.tolist() == [[0], [1]]
    assert no_loops.num_nodes == 3

    # Only self-loops.
    graph = GraphData(
        edge_index=mx.array([[0, 1], [0, 1]], dtype=mx.uint32),
        edge_features=mx.ones((2, 3)),
    )
    no_loops = RemoveSelfLoops()(graph)
    assert no_loops.edge_index.shape == (2, 0)
    assert no_loops.edge_index.dtype == mx.uint32
    assert no_loops.edge_features.shape == (0, 3)
    assert no_loops.num_nodes == 2

    # Nothing to remove.
    graph = GraphData(edge_index=mx.array([[0, 1], [1, 0]]))
    assert RemoveSelfLoops()(graph).edge_index.tolist() == [[0, 1], [1, 0]]


@pytest.mark.parametrize("seed", range(5))
def test_remove_isolated_nodes_matches_pyg(seed):
    torch = pytest.importorskip("torch")
    pyg_utils = pytest.importorskip("torch_geometric.utils")

    graph = random_graph(seed, num_nodes=25, num_edges=16)
    # PyG's `remove_isolated_nodes` utility shares our definition of an
    # isolated node (no edge to another node) but moves the self-loops of the
    # remaining nodes to the end: compare edges together with their features.
    expected_index, expected_features, expected_mask = pyg_utils.remove_isolated_nodes(
        torch.from_numpy(np.asarray(graph.edge_index)),
        torch.from_numpy(np.asarray(graph.edge_features)),
        num_nodes=graph.num_nodes,
    )
    expected_mask = expected_mask.numpy()
    assert 0 < expected_mask.sum() < graph.num_nodes - 3

    result = RemoveIsolatedNodes()(graph)

    def sorted_rows(edge_index, edge_features) -> np.ndarray:
        rows = np.concatenate(
            [np.asarray(edge_index).T, np.asarray(edge_features)], axis=1
        )
        return rows[np.lexsort(rows.T[::-1])]

    assert np.array_equal(
        sorted_rows(result.edge_index, result.edge_features),
        sorted_rows(expected_index.numpy(), expected_features.numpy()),
    )
    # Every per-node attribute is sliced with the same mask.
    assert result.num_nodes == expected_mask.sum()
    assert np.array_equal(
        np.asarray(result.node_features),
        np.asarray(graph.node_features)[expected_mask],
    )
    assert result.node_labels.tolist() == np.nonzero(expected_mask)[0].tolist()
    assert np.array_equal(
        np.asarray(result.train_mask), np.asarray(graph.train_mask)[expected_mask]
    )
    assert result.graph_labels.tolist() == [1]
    assert not result.has_isolated_nodes()


def test_remove_isolated_nodes_relabels_edges():
    graph = GraphData(
        edge_index=mx.array([[0, 3, 5, 2], [3, 0, 3, 2]], dtype=mx.uint32),
        node_features=mx.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]),
        edge_features=mx.array([10.0, 11.0, 12.0, 13.0]),
    )
    result = RemoveIsolatedNodes()(graph)

    # Nodes 1 and 4 have no edge, node 2 only has a self-loop.
    assert result.edge_index.tolist() == [[0, 1, 2], [1, 0, 1]]
    assert result.edge_index.dtype == mx.uint32
    assert result.node_features.tolist() == [[0.0], [3.0], [5.0]]
    assert result.edge_features.tolist() == [10.0, 11.0, 12.0]
    assert result.num_nodes == 3


def test_remove_isolated_nodes_edge_cases():
    # Nothing to remove.
    graph = GraphData(
        edge_index=mx.array([[0, 1], [1, 2]]), node_features=mx.ones((3, 2))
    )
    result = RemoveIsolatedNodes()(graph)
    assert result.edge_index.tolist() == [[0, 1], [1, 2]]
    assert result.node_features is graph.node_features
    assert result.num_nodes == 3

    # Only isolated nodes.
    graph = GraphData(
        edge_index=mx.array([[1], [1]], dtype=mx.uint32),
        node_features=mx.ones((3, 2)),
        edge_features=mx.ones((1, 4)),
    )
    result = RemoveIsolatedNodes()(graph)
    assert result.edge_index.shape == (2, 0)
    assert result.edge_index.dtype == mx.uint32
    assert result.node_features.shape == (0, 2)
    assert result.edge_features.shape == (0, 4)
    assert result.num_nodes == 0

    # Isolated nodes known only through an explicit number of nodes.
    graph = GraphData(edge_index=mx.array([[0, 4], [4, 0]]), num_nodes=7)
    result = RemoveIsolatedNodes()(graph)
    assert result.edge_index.tolist() == [[0, 1], [1, 0]]
    assert result.num_nodes == 2


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("num_components", [1, 2])
@pytest.mark.parametrize("connection", ["weak", "strong"])
def test_largest_connected_components_matches_pyg(seed, num_components, connection):
    torch = pytest.importorskip("torch")
    pyg_data = pytest.importorskip("torch_geometric.data")
    pyg_transforms = pytest.importorskip("torch_geometric.transforms")

    graph = random_graph(seed, num_nodes=40, num_edges=30)
    expected = pyg_transforms.LargestConnectedComponents(num_components, connection)(
        pyg_data.Data(
            x=torch.from_numpy(np.asarray(graph.node_features)),
            edge_index=torch.from_numpy(np.asarray(graph.edge_index)),
            edge_attr=torch.from_numpy(np.asarray(graph.edge_features)),
            y=torch.from_numpy(np.asarray(graph.node_labels)),
        )
    )
    assert expected.num_nodes < graph.num_nodes

    result = LargestConnectedComponents(num_components, connection)(graph)

    assert result.num_nodes == expected.num_nodes
    assert np.array_equal(np.asarray(result.edge_index), expected.edge_index.numpy())
    assert np.array_equal(np.asarray(result.node_features), expected.x.numpy())
    assert np.array_equal(np.asarray(result.edge_features), expected.edge_attr.numpy())
    assert np.array_equal(np.asarray(result.node_labels), expected.y.numpy())
    assert result.train_mask.shape == (expected.num_nodes,)


def test_largest_connected_components_basic():
    # Components {0, 1, 2}, {3, 4} and {5}.
    graph = GraphData(
        edge_index=mx.array([[0, 1, 3], [1, 2, 4]], dtype=mx.uint32),
        node_features=mx.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]),
    )

    largest = LargestConnectedComponents()(graph)
    assert largest.edge_index.tolist() == [[0, 1], [1, 2]]
    assert largest.edge_index.dtype == mx.uint32
    assert largest.node_features.tolist() == [[0.0], [1.0], [2.0]]
    assert largest.num_nodes == 3

    two_largest = LargestConnectedComponents(2)(graph)
    assert two_largest.edge_index.tolist() == [[0, 1, 3], [1, 2, 4]]
    assert two_largest.num_nodes == 5

    # Fewer components than requested: nothing changes.
    everything = LargestConnectedComponents(3)(graph)
    assert everything.edge_index.tolist() == [[0, 1, 3], [1, 2, 4]]
    assert everything.num_nodes == 6

    with pytest.raises(ValueError):
        LargestConnectedComponents(connection="other")


def test_largest_connected_components_empty_edges():
    graph = GraphData(edge_index=mx.zeros((2, 0), dtype=mx.int32), num_nodes=3)
    result = LargestConnectedComponents()(graph)
    assert result.edge_index.shape == (2, 0)
    assert result.num_nodes == 1


@pytest.mark.parametrize("dtype", [mx.int32, mx.uint32, mx.int64, mx.uint64])
@pytest.mark.parametrize(
    "transform, expected",
    [
        (ToUndirected(), [[0, 1, 1, 1, 2, 4, 5], [1, 0, 1, 2, 1, 5, 4]]),
        (
            AddSelfLoops(),
            [[0, 1, 1, 4, 0, 1, 2, 3, 4, 5], [1, 1, 2, 5, 0, 1, 2, 3, 4, 5]],
        ),
        (RemoveSelfLoops(), [[0, 1, 4], [1, 2, 5]]),
        (RemoveIsolatedNodes(), [[0, 1, 1, 3], [1, 1, 2, 4]]),
        (LargestConnectedComponents(), [[0, 1, 1], [1, 1, 2]]),
    ],
)
def test_transforms_preserve_edge_index_dtype(dtype, transform, expected):
    # Components {0, 1, 2}, {3} and {4, 5}, with a self-loop on node 1.
    graph = GraphData(edge_index=mx.array([[0, 1, 1, 4], [1, 1, 2, 5]], dtype=dtype))
    result = transform(graph)
    assert result.edge_index.dtype == dtype
    assert result.edge_index.tolist() == expected
