import math

import mlx.core as mx
import numpy as np
import pytest

from mlx_graphs.utils.laplacian import get_laplacian, homophily


def random_edges(seed: int, num_nodes: int = 12, num_edges: int = 40) -> np.ndarray:
    """Random edges with duplicates and self-loops."""
    return np.random.default_rng(seed).integers(0, num_nodes, size=(2, num_edges))


def to_dense(edge_index, weights, num_nodes: int) -> np.ndarray:
    dense = np.zeros((num_nodes, num_nodes))
    np.add.at(dense, (np.asarray(edge_index[0]), np.asarray(edge_index[1])), weights)
    return dense


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("normalization", [None, "sym", "rw"])
@pytest.mark.parametrize("weighted", [False, True])
def test_get_laplacian_matches_pyg(seed, normalization, weighted):
    torch = pytest.importorskip("torch")
    pyg_utils = pytest.importorskip("torch_geometric.utils")

    num_nodes = 14  # nodes 12 and 13 are isolated
    edges = random_edges(seed)
    weights = None
    if weighted:
        weights = np.random.default_rng(seed).uniform(0.5, 2.0, edges.shape[1])
        weights = weights.astype(np.float32)

    expected_index, expected_weights = pyg_utils.get_laplacian(
        torch.from_numpy(edges),
        None if weights is None else torch.from_numpy(weights),
        normalization=normalization,
        num_nodes=num_nodes,
    )
    edge_index, edge_weights = get_laplacian(
        mx.array(edges),
        None if weights is None else mx.array(weights),
        normalization=normalization,
        num_nodes=num_nodes,
    )

    assert np.array_equal(np.asarray(edge_index), expected_index.numpy())
    assert edge_weights.dtype == mx.float32
    assert np.allclose(
        np.asarray(edge_weights), expected_weights.numpy(), rtol=1e-5, atol=1e-6
    )


def test_get_laplacian_basic():
    edge_index = mx.array([[0, 1, 1, 2], [1, 0, 2, 1]])

    lap_index, lap_weights = get_laplacian(edge_index)
    assert lap_index.tolist() == [[0, 1, 1, 2, 0, 1, 2], [1, 0, 2, 1, 0, 1, 2]]
    assert lap_weights.tolist() == [-1, -1, -1, -1, 1, 2, 1]

    _, lap_weights = get_laplacian(edge_index, normalization="rw")
    assert lap_weights.tolist() == [-1, -0.5, -0.5, -1, 1, 1, 1]

    _, lap_weights = get_laplacian(edge_index, normalization="sym")
    inv_sqrt_2 = 1 / math.sqrt(2)
    assert np.allclose(
        lap_weights.tolist(),
        [-inv_sqrt_2, -inv_sqrt_2, -inv_sqrt_2, -inv_sqrt_2, 1, 1, 1],
    )


@pytest.mark.parametrize("seed", range(3))
def test_get_laplacian_is_degree_minus_adjacency(seed):
    num_nodes = 12
    edges = random_edges(seed)
    edges = edges[:, edges[0] != edges[1]]
    weights = np.random.default_rng(seed).uniform(0.5, 2.0, edges.shape[1])

    adjacency = to_dense(edges, weights, num_nodes)
    degree = np.diag(adjacency.sum(axis=1))

    lap_index, lap_weights = get_laplacian(
        mx.array(edges), mx.array(weights.astype(np.float32)), num_nodes=num_nodes
    )
    laplacian = to_dense(lap_index, np.asarray(lap_weights), num_nodes)
    assert np.allclose(laplacian, degree - adjacency, atol=1e-5)
    # Rows of L = D - A sum to zero.
    assert np.allclose(laplacian.sum(axis=1), 0, atol=1e-5)

    # Rows of I - D^-1 A sum to zero for the nodes that have neighbors.
    lap_index, lap_weights = get_laplacian(
        mx.array(edges),
        mx.array(weights.astype(np.float32)),
        normalization="rw",
        num_nodes=num_nodes,
    )
    laplacian = to_dense(lap_index, np.asarray(lap_weights), num_nodes)
    has_neighbors = adjacency.sum(axis=1) > 0
    assert np.allclose(laplacian.sum(axis=1)[has_neighbors], 0, atol=1e-5)


def test_get_laplacian_removes_self_loops_and_handles_isolated_nodes():
    # Node 0 only has a self-loop and node 3 has no edge: both end up with a
    # degree of 0, and a diagonal entry of 1 once normalized.
    edge_index = mx.array([[0, 1, 2], [0, 2, 1]])
    lap_index, lap_weights = get_laplacian(edge_index, num_nodes=4)
    assert lap_index.tolist() == [[1, 2, 0, 1, 2, 3], [2, 1, 0, 1, 2, 3]]
    assert lap_weights.tolist() == [-1, -1, 0, 1, 1, 0]

    for normalization in ("sym", "rw"):
        _, lap_weights = get_laplacian(
            edge_index, normalization=normalization, num_nodes=4
        )
        assert lap_weights.tolist() == [-1, -1, 1, 1, 1, 1]
        assert not np.isnan(np.asarray(lap_weights)).any()


@pytest.mark.parametrize("dtype", [mx.uint32, mx.int64, mx.uint64])
def test_get_laplacian_empty_edges_and_dtype(dtype):
    no_edges = mx.zeros((2, 0), dtype=dtype)
    lap_index, lap_weights = get_laplacian(no_edges, num_nodes=3)
    assert lap_index.dtype == dtype
    assert lap_index.tolist() == [[0, 1, 2], [0, 1, 2]]
    assert lap_weights.tolist() == [0, 0, 0]
    for normalization in ("sym", "rw"):
        lap_index, lap_weights = get_laplacian(
            no_edges, normalization=normalization, num_nodes=3
        )
        assert lap_index.tolist() == [[0, 1, 2], [0, 1, 2]]
        assert lap_weights.tolist() == [1, 1, 1]

    lap_index, lap_weights = get_laplacian(no_edges)
    assert lap_index.shape == (2, 0)
    assert lap_weights.shape == (0,)

    edge_index = mx.array([[0, 1], [1, 0]], dtype=dtype)
    lap_index, lap_weights = get_laplacian(edge_index, normalization="sym")
    assert lap_index.dtype == dtype
    assert lap_weights.tolist() == [-1, -1, 1, 1]
    assert homophily(edge_index, mx.array([1, 1])) == 1.0
    assert homophily(edge_index, mx.array([1, 1]), method="node") == 1.0


def test_get_laplacian_keeps_float_weight_dtype():
    edge_index = mx.array([[0, 1], [1, 0]])
    weights = mx.array([2.0, 2.0], dtype=mx.float16)
    assert get_laplacian(edge_index, weights)[1].dtype == mx.float16
    # Integer weights give a float32 Laplacian.
    _, lap_weights = get_laplacian(edge_index, mx.array([2, 2]), normalization="rw")
    assert lap_weights.dtype == mx.float32
    assert lap_weights.tolist() == [-1, -1, 1, 1]


def test_get_laplacian_invalid_inputs():
    edge_index = mx.array([[0, 1], [1, 0]])
    with pytest.raises(ValueError):
        get_laplacian(edge_index, normalization="other")
    with pytest.raises(ValueError):
        get_laplacian(edge_index, mx.ones((3,)))
    with pytest.raises(ValueError):
        get_laplacian(edge_index, num_nodes=1)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("method", ["edge", "node"])
def test_homophily_matches_pyg(seed, method):
    torch = pytest.importorskip("torch")
    pyg_utils = pytest.importorskip("torch_geometric.utils")

    num_nodes = 14  # nodes 12 and 13 are isolated
    edges = random_edges(seed)
    labels = np.random.default_rng(seed).integers(0, 3, size=num_nodes)

    expected = pyg_utils.homophily(
        torch.from_numpy(edges), torch.from_numpy(labels), method=method
    )
    result = homophily(mx.array(edges), mx.array(labels), method=method)

    assert isinstance(result, float)
    assert result == pytest.approx(expected, abs=1e-6)
    # Labels of shape [num_nodes, 1] are accepted too.
    column = homophily(mx.array(edges), mx.array(labels[:, None]), method=method)
    assert column == result


def test_homophily_basic():
    edge_index = mx.array([[0, 1, 2, 3], [1, 2, 0, 4]])
    node_labels = mx.array([0, 0, 0, 0, 1])
    assert homophily(edge_index, node_labels) == pytest.approx(0.75)
    assert homophily(edge_index, node_labels, method="node") == pytest.approx(0.6)

    same = mx.array([1, 1, 1, 1, 1])
    assert homophily(edge_index, same) == 1.0
    different = mx.array([0, 1, 2, 3, 4])
    assert homophily(edge_index, different) == 0.0
    assert homophily(edge_index, different, method="node") == 0.0


def test_homophily_empty_and_invalid():
    no_edges = mx.zeros((2, 0), dtype=mx.int32)
    labels = mx.array([0, 1, 0])
    assert math.isnan(homophily(no_edges, labels))
    # No node has an incoming edge.
    assert homophily(no_edges, labels, method="node") == 0.0

    edge_index = mx.array([[0, 1], [1, 2]])
    with pytest.raises(ValueError):
        homophily(edge_index, labels, method="other")
    with pytest.raises(ValueError):
        homophily(edge_index, mx.array([0, 1]))  # node 2 has no label
    with pytest.raises(ValueError):
        homophily(edge_index, mx.ones((3, 2)))  # one-hot labels
