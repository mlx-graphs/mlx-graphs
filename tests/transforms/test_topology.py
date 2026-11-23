import mlx.core as mx
import numpy as np
import pytest

from mlx_graphs.data import GraphData
from mlx_graphs.transforms import LineGraph, ToDense, TwoHop, VirtualNode
from mlx_graphs.utils import to_pyg


def random_graph(seed: int, num_nodes: int = 10, num_edges: int = 25) -> GraphData:
    """Random directed graph with duplicates, self-loops and isolated nodes."""
    rng = np.random.default_rng(seed)
    edges = rng.integers(0, num_nodes - 2, size=(2, num_edges))
    return GraphData(edge_index=mx.array(edges), num_nodes=num_nodes)


def undirected_graph(seed: int, num_nodes: int = 10, num_pairs: int = 15):
    """Random undirected graph without self-loops nor duplicates."""
    rng = np.random.default_rng(seed)
    pairs = rng.integers(0, num_nodes, size=(2, num_pairs))
    pairs = np.unique(np.sort(pairs[:, pairs[0] != pairs[1]], axis=0), axis=1)
    edges = np.concatenate([pairs, pairs[::-1]], axis=1)
    return GraphData(edge_index=mx.array(edges), num_nodes=num_nodes)


def test_repr():
    assert str(VirtualNode()) == "VirtualNode()"
    assert str(TwoHop()) == "TwoHop()"
    assert str(LineGraph()) == "LineGraph(force_directed=False)"
    assert str(ToDense()) == "ToDense()"
    assert str(ToDense(5)) == "ToDense(num_nodes=5)"


def test_virtual_node_example():
    graph = GraphData(
        edge_index=mx.array([[0, 1], [1, 2]]),
        node_features=mx.ones((3, 2)),
    )
    graph = VirtualNode()(graph)
    assert graph.edge_index.tolist() == [
        [0, 1, 0, 1, 2, 3, 3, 3],
        [1, 2, 3, 3, 3, 0, 1, 2],
    ]
    assert graph.edge_type.tolist() == [0, 0, 1, 1, 1, 2, 2, 2]
    assert graph.node_features.tolist() == [[1, 1], [1, 1], [1, 1], [0, 0]]
    assert graph.num_nodes == 4


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("edge_type", [False, True])
def test_virtual_node_matches_pyg(seed, edge_type):
    pyg_transforms = pytest.importorskip("torch_geometric.transforms")
    rng = np.random.default_rng(seed)
    graph = random_graph(seed)
    graph.node_features = mx.array(rng.normal(size=(10, 3)).astype(np.float32))
    graph.edge_features = mx.array(rng.normal(size=(25, 2)).astype(np.float32))
    graph.edge_weights = mx.array(rng.uniform(size=25).astype(np.float32))
    graph.node_labels = mx.array(rng.integers(0, 3, size=10))
    if edge_type:
        graph.edge_type = mx.array(rng.integers(0, 4, size=25))

    expected = pyg_transforms.VirtualNode()(to_pyg(graph))
    out = VirtualNode()(graph)

    assert out.num_nodes == expected.num_nodes == 11
    assert np.array_equal(np.asarray(out.edge_index), expected.edge_index.numpy())
    assert np.array_equal(np.asarray(out.edge_type), expected.edge_type.numpy())
    assert np.allclose(np.asarray(out.node_features), expected.x.numpy())
    assert np.allclose(np.asarray(out.edge_features), expected.edge_attr.numpy())
    assert np.allclose(np.asarray(out.edge_weights), expected.edge_weight.numpy())
    assert np.array_equal(np.asarray(out.node_labels), expected.y.numpy())


def test_virtual_node_without_node_features():
    graph = GraphData(edge_index=mx.array([[0], [1]]), num_nodes=3)
    out = VirtualNode()(graph)
    assert out.num_nodes == 4
    assert graph.num_nodes == 3


def test_two_hop_example():
    graph = TwoHop()(GraphData(edge_index=mx.array([[0, 1], [1, 2]])))
    assert graph.edge_index.tolist() == [[0, 0, 1], [1, 2, 2]]


def test_two_hop_keeps_edge_features_aligned():
    graph = GraphData(
        edge_index=mx.array([[1, 0], [2, 1]]),
        edge_features=mx.array([[1.0], [2.0]]),
    )
    graph = TwoHop()(graph)
    assert graph.edge_index.tolist() == [[0, 0, 1], [1, 2, 2]]
    assert graph.edge_features.tolist() == [[2.0], [0.0], [1.0]]


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("edge_features", [False, True])
def test_two_hop_matches_pyg(seed, edge_features, monkeypatch):
    pyg_utils = pytest.importorskip("torch_geometric.utils")
    pyg_transforms = pytest.importorskip("torch_geometric.transforms")
    graph = random_graph(seed)
    if edge_features:
        # PyG sorts the edges by source without permuting `edge_attr`, so the
        # comparison needs edges that are already sorted, and a stable sort:
        # x86 builds of torch sort ties in an unspecified order.
        graph = graph.coalesce()
        index_sort = pyg_utils.index_sort

        def stable_index_sort(inputs, max_value=None, stable=False):
            return index_sort(inputs, max_value, stable=True)

        monkeypatch.setattr(pyg_utils, "index_sort", stable_index_sort)
        graph.edge_features = mx.random.normal((graph.num_edges, 2))

    expected = pyg_transforms.TwoHop()(to_pyg(graph))
    out = TwoHop()(graph)

    assert np.array_equal(np.asarray(out.edge_index), expected.edge_index.numpy())
    if edge_features:
        assert np.allclose(
            np.asarray(out.edge_features), expected.edge_attr.numpy(), atol=1e-6
        )
    assert out.num_nodes == 10


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("edge_features", [False, True])
def test_line_graph_directed_matches_pyg(seed, edge_features):
    pyg_transforms = pytest.importorskip("torch_geometric.transforms")
    graph = random_graph(seed)
    # PyG uses the original (not coalesced) edge features as node features,
    # so the input is coalesced first for the comparison.
    graph = graph.coalesce()
    if edge_features:
        graph.edge_features = mx.random.normal((graph.num_edges, 2))

    expected = pyg_transforms.LineGraph(force_directed=True)(to_pyg(graph))
    out = LineGraph(force_directed=True)(graph)

    assert out.num_nodes == expected.num_nodes
    assert np.array_equal(np.asarray(out.edge_index), expected.edge_index.numpy())
    assert out.edge_features is None
    if edge_features:
        assert np.allclose(np.asarray(out.node_features), expected.x.numpy())
    else:
        assert out.node_features is None


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("edge_features", [False, True])
def test_line_graph_undirected_matches_pyg(seed, edge_features):
    pyg_transforms = pytest.importorskip("torch_geometric.transforms")
    graph = undirected_graph(seed)
    if edge_features:
        # Symmetric features, so that the graph is undirected.
        num_pairs = graph.num_edges // 2
        features = mx.random.normal((num_pairs, 2))
        graph.edge_features = mx.concatenate([features, features])

    expected = pyg_transforms.LineGraph()(to_pyg(graph))
    out = LineGraph()(graph)

    assert out.num_nodes == expected.num_nodes
    assert np.array_equal(np.asarray(out.edge_index), expected.edge_index.numpy())
    if edge_features:
        assert np.allclose(np.asarray(out.node_features), expected.x.numpy())


def test_line_graph_undirected_path():
    graph = GraphData(edge_index=mx.array([[0, 1, 1, 2], [1, 0, 2, 1]]))
    out = LineGraph()(graph)
    assert out.edge_index.tolist() == [[0, 1], [1, 0]]
    assert out.num_nodes == 2


@pytest.mark.parametrize("num_nodes", [None, 12])
@pytest.mark.parametrize("edge_features", [False, True])
def test_to_dense_matches_pyg(num_nodes, edge_features):
    pyg_transforms = pytest.importorskip("torch_geometric.transforms")
    rng = np.random.default_rng(0)
    graph = random_graph(0)
    graph.node_features = mx.array(rng.normal(size=(10, 3)).astype(np.float32))
    graph.node_labels = mx.array(rng.integers(0, 3, size=10))
    graph.pos = mx.array(rng.normal(size=(10, 2)).astype(np.float32))
    if edge_features:
        graph.edge_features = mx.random.normal((graph.num_edges, 2))

    expected = pyg_transforms.ToDense(num_nodes)(to_pyg(graph))
    out = ToDense(num_nodes)(graph)

    assert out.edge_index is None and out.edge_features is None
    assert out.num_nodes == (num_nodes or 10)
    assert np.allclose(np.asarray(out.adj), expected.adj.numpy(), atol=1e-6)
    assert np.array_equal(np.asarray(out.mask), expected.mask.numpy())
    assert np.allclose(np.asarray(out.node_features), expected.x.numpy())
    assert np.allclose(np.asarray(out.pos), expected.pos.numpy())
    assert np.array_equal(np.asarray(out.node_labels), expected.y.numpy())


def test_to_dense_too_many_nodes():
    with pytest.raises(ValueError):
        ToDense(5)(random_graph(0))
