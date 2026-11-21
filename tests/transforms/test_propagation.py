import mlx.core as mx
import numpy as np
import pytest

from mlx_graphs.data import GraphData
from mlx_graphs.transforms import SIGN, FeaturePropagation
from mlx_graphs.utils import to_pyg


def random_graph(seed: int, num_nodes: int = 10, num_edges: int = 30, weights=False):
    """Random graph with duplicates, self-loops and isolated nodes."""
    rng = np.random.default_rng(seed)
    edges = rng.integers(0, num_nodes - 2, size=(2, num_edges))
    graph = GraphData(
        edge_index=mx.array(edges),
        node_features=mx.array(rng.normal(size=(num_nodes, 3)).astype(np.float32)),
    )
    if weights:
        graph.edge_weights = mx.array(
            rng.uniform(0.5, 2.0, size=num_edges).astype(np.float32)
        )
    return graph


def test_repr():
    assert str(SIGN(3)) == "SIGN(K=3)"
    mask = mx.array([[True, False], [False, False]])
    assert (
        str(FeaturePropagation(mask, 10))
        == "FeaturePropagation(missing_features=25.0%, num_iterations=10)"
    )


def test_sign_example():
    graph = GraphData(
        edge_index=mx.array([[0, 1], [1, 0]]),
        node_features=mx.array([[1.0], [2.0]]),
    )
    graph = SIGN(K=2)(graph)
    assert graph.x1.tolist() == [[2.0], [1.0]]
    assert graph.x2.tolist() == [[1.0], [2.0]]


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("weights", [False, True])
def test_sign_matches_pyg(seed, weights):
    pyg_transforms = pytest.importorskip("torch_geometric.transforms")
    graph = random_graph(seed, weights=weights)
    expected = pyg_transforms.SIGN(3)(to_pyg(graph))
    out = SIGN(3)(graph)
    for i in range(1, 4):
        actual = np.asarray(getattr(out, f"x{i}"))
        assert np.allclose(actual, expected[f"x{i}"].numpy(), atol=1e-5)


def test_feature_propagation_example():
    node_features = mx.array([[1.0], [float("nan")], [3.0]])
    graph = GraphData(
        edge_index=mx.array([[0, 1, 1, 2], [1, 0, 2, 1]]),
        node_features=node_features,
    )
    graph = FeaturePropagation(mx.isnan(node_features))(graph)
    expected = [[1.0], [4 / np.sqrt(2)], [3.0]]
    assert np.allclose(np.asarray(graph.node_features), expected)


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("weights", [None, "edge_weights", "edge_features"])
def test_feature_propagation_matches_pyg(seed, weights):
    torch = pytest.importorskip("torch")
    pyg_transforms = pytest.importorskip("torch_geometric.transforms")
    # Without weights, PyG collapses duplicate edges instead of summing them.
    graph = random_graph(seed, weights=weights == "edge_weights").coalesce()
    if weights == "edge_features":
        graph.edge_features = mx.random.uniform(0.5, 2.0, (graph.num_edges,))
    rng = np.random.default_rng(seed)
    missing = rng.random(graph.node_features.shape) < 0.4
    graph.node_features = mx.where(
        mx.array(missing), mx.array(float("nan")), graph.node_features
    )

    expected = pyg_transforms.FeaturePropagation(
        torch.from_numpy(missing), num_iterations=15
    )(to_pyg(graph)).x
    out = FeaturePropagation(mx.array(missing), num_iterations=15)(graph)
    assert np.allclose(np.asarray(out.node_features), expected.numpy(), atol=1e-5)


def test_feature_propagation_checks_mask_shape():
    graph = random_graph(0)
    with pytest.raises(ValueError):
        FeaturePropagation(mx.zeros((2, 2), dtype=mx.bool_))(graph)
