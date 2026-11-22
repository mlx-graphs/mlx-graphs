import mlx.core as mx
import numpy as np
import pytest

from mlx_graphs.data import GraphData
from mlx_graphs.transforms import LocalDegreeProfile, TargetIndegree
from mlx_graphs.utils import to_pyg


def random_graph(seed: int, num_nodes: int = 10, num_edges: int = 25, **kwargs):
    """Random graph with duplicates, self-loops and isolated nodes."""
    rng = np.random.default_rng(seed)
    edges = rng.integers(0, num_nodes - 2, size=(2, num_edges))
    return GraphData(edge_index=mx.array(edges), num_nodes=num_nodes, **kwargs)


def test_repr():
    assert str(TargetIndegree()) == "TargetIndegree(norm=True, max_value=None)"
    assert str(LocalDegreeProfile()) == "LocalDegreeProfile()"


def test_target_indegree_example():
    graph = GraphData(edge_index=mx.array([[0, 1, 2], [1, 2, 1]]))
    graph = TargetIndegree()(graph)
    assert graph.edge_features.tolist() == [[1.0], [0.5], [1.0]]


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("norm", [False, True])
@pytest.mark.parametrize("max_value", [None, 10.0])
@pytest.mark.parametrize("cat", [False, True])
@pytest.mark.parametrize("edge_features", [None, "1d", "2d"])
def test_target_indegree_matches_pyg(seed, norm, max_value, cat, edge_features):
    pyg_transforms = pytest.importorskip("torch_geometric.transforms")
    graph = random_graph(seed)
    if edge_features == "1d":
        graph.edge_features = mx.arange(graph.num_edges).astype(mx.float32)
    elif edge_features == "2d":
        graph.edge_features = mx.random.normal((graph.num_edges, 3))

    expected = pyg_transforms.TargetIndegree(norm, max_value, cat)(to_pyg(graph))
    out = TargetIndegree(norm, max_value, cat)(graph)
    assert np.allclose(np.asarray(out.edge_features), expected.edge_attr.numpy())


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("node_features", [None, "1d", "2d"])
def test_local_degree_profile_matches_pyg(seed, node_features):
    pyg_transforms = pytest.importorskip("torch_geometric.transforms")
    graph = random_graph(seed)
    if node_features == "1d":
        graph.node_features = mx.arange(graph.num_nodes).astype(mx.float32)
    elif node_features == "2d":
        graph.node_features = mx.random.normal((graph.num_nodes, 2))

    expected = pyg_transforms.LocalDegreeProfile()(to_pyg(graph))
    out = LocalDegreeProfile()(graph)
    assert out.node_features.shape == tuple(expected.x.shape)
    assert np.allclose(np.asarray(out.node_features), expected.x.numpy(), atol=1e-5)


def test_local_degree_profile_does_not_modify_input():
    graph = random_graph(0)
    LocalDegreeProfile()(graph)
    assert graph.node_features is None
