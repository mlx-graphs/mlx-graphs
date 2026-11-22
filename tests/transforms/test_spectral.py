import mlx.core as mx
import numpy as np
import pytest

from mlx_graphs.data import GraphData
from mlx_graphs.transforms import AddLaplacianEigenvectorPE, SVDFeatureReduction
from mlx_graphs.utils import to_pyg


def undirected_graph(seed: int, num_nodes: int, num_pairs: int) -> GraphData:
    """Random connected undirected graph: a path plus random chords."""
    rng = np.random.default_rng(seed)
    src = np.arange(num_nodes - 1)
    dst = src + 1
    extra = rng.integers(0, num_nodes, size=(2, num_pairs))
    extra = extra[:, extra[0] != extra[1]]
    pairs = np.concatenate([np.stack([src, dst]), extra], axis=1)
    pairs = np.unique(np.sort(pairs, axis=0), axis=1)
    edges = np.concatenate([pairs, pairs[::-1]], axis=1)
    weights = rng.uniform(0.5, 2.0, size=pairs.shape[1])
    return GraphData(
        edge_index=mx.array(edges),
        edge_weights=mx.array(np.concatenate([weights, weights]).astype(np.float32)),
    )


def align_signs(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """Flips the columns of ``actual`` to the sign of those of ``expected``."""
    signs = np.sign(np.sum(actual * expected, axis=0))
    signs[signs == 0] = 1
    return actual * signs


def test_repr():
    assert str(AddLaplacianEigenvectorPE(3)) == "AddLaplacianEigenvectorPE(k=3)"
    assert str(SVDFeatureReduction(4)) == "SVDFeatureReduction(4)"


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("is_undirected", [False, True])
@pytest.mark.parametrize("num_nodes", [12, 120])
def test_laplacian_pe_matches_pyg(seed, is_undirected, num_nodes):
    torch = pytest.importorskip("torch")
    pyg_transforms = pytest.importorskip("torch_geometric.transforms")
    graph = undirected_graph(seed, num_nodes, num_pairs=num_nodes)
    k = 3

    torch.manual_seed(seed)
    expected = pyg_transforms.AddLaplacianEigenvectorPE(k, is_undirected=is_undirected)(
        to_pyg(graph)
    ).laplacian_eigenvector_pe.numpy()
    out = AddLaplacianEigenvectorPE(k, is_undirected=is_undirected)(graph)
    actual = np.asarray(out.laplacian_eigenvector_pe)

    assert actual.shape == (num_nodes, k)
    assert actual.dtype == np.float32
    assert np.allclose(align_signs(actual, expected), expected, atol=1e-4)


def test_laplacian_pe_eigenvectors():
    graph = undirected_graph(0, 10, 10)
    out = AddLaplacianEigenvectorPE(4, is_undirected=True)(graph)
    pe = np.asarray(out.laplacian_eigenvector_pe)
    # Orthonormal eigenvectors.
    assert np.allclose(pe.T @ pe, np.eye(4), atol=1e-5)


def test_laplacian_pe_concatenates_to_node_features():
    graph = undirected_graph(0, 10, 10)
    graph.node_features = mx.ones((10, 2))
    out = AddLaplacianEigenvectorPE(3, attr_name=None, is_undirected=True)(graph)
    assert out.node_features.shape == (10, 5)
    assert not hasattr(out, "laplacian_eigenvector_pe")


def test_laplacian_pe_random_sign_is_seeded():
    graph = undirected_graph(0, 10, 10)
    transform = AddLaplacianEigenvectorPE(3, is_undirected=True)
    mx.random.seed(3)
    first = transform(graph).laplacian_eigenvector_pe
    mx.random.seed(3)
    second = transform(graph).laplacian_eigenvector_pe
    assert mx.array_equal(first, second)


def test_laplacian_pe_requires_k_smaller_than_num_nodes():
    graph = undirected_graph(0, 4, 2)
    with pytest.raises(ValueError):
        AddLaplacianEigenvectorPE(4)(graph)


@pytest.mark.parametrize("out_channels", [2, 5, 8])
def test_svd_feature_reduction_matches_pyg(out_channels):
    pytest.importorskip("torch")
    pyg_transforms = pytest.importorskip("torch_geometric.transforms")
    rng = np.random.default_rng(0)
    graph = GraphData(
        edge_index=mx.array([[0, 1], [1, 2]]),
        node_features=mx.array(rng.normal(size=(10, 6)).astype(np.float32)),
    )
    expected = pyg_transforms.SVDFeatureReduction(out_channels)(to_pyg(graph)).x
    out = SVDFeatureReduction(out_channels)(graph)
    actual = np.asarray(out.node_features)

    assert actual.shape == tuple(expected.shape)
    assert np.allclose(align_signs(actual, expected.numpy()), expected, atol=1e-4)


def test_laplacian_pe_sparse_with_many_components():
    """Regression: on graphs with many connected components (e.g. Cora) the
    sparse eigensolver did not converge (ArpackNoConvergence) in float64,
    while PyG's, which works in float32, does. Here: 30 small random trees
    and one random connected graph of 1000 nodes."""
    pytest.importorskip("scipy")
    rng = np.random.default_rng(0)
    blocks, offset = [], 0
    for _ in range(30):
        size = rng.integers(2, 6)
        src = np.arange(1, size)
        blocks.append(np.stack([src, rng.integers(0, src)]) + offset)
        offset += size
    src = np.arange(1, 1000)
    extra = rng.integers(0, 1000, (2, 1000))
    extra = extra[:, extra[0] != extra[1]]
    blocks.append(np.concatenate([np.stack([src, rng.integers(0, src)]), extra], 1))
    blocks[-1] += offset
    num_nodes = offset + 1000
    pairs = np.unique(np.sort(np.concatenate(blocks, 1), 0), axis=1)
    edges = np.concatenate([pairs, pairs[::-1]], axis=1)
    graph = GraphData(edge_index=mx.array(edges), node_features=mx.ones((num_nodes, 2)))

    out = AddLaplacianEigenvectorPE(4, attr_name=None, is_undirected=True)(graph)
    assert out.node_features.shape == (num_nodes, 6)
    assert mx.all(mx.isfinite(out.node_features)).item()
