import mlx.core as mx
import numpy as np
import pytest

from mlx_graphs.data import GraphData
from mlx_graphs.transforms.features import (
    AddRandomWalkPE,
    Constant,
    GCNNorm,
    OneHotDegree,
)


def random_edges(seed: int, num_nodes: int = 12, num_edges: int = 30) -> np.ndarray:
    """Random edges with duplicates and self-loops."""
    return np.random.default_rng(seed).integers(0, num_nodes, size=(2, num_edges))


def simple_edges(seed: int, num_nodes: int = 12, num_edges: int = 40) -> np.ndarray:
    """Random edges without duplicates (self-loops are possible)."""
    pairs = np.random.default_rng(seed).choice(num_nodes**2, num_edges, replace=False)
    return np.stack([pairs // num_nodes, pairs % num_nodes])


def to_pyg(graph: GraphData):
    torch = pytest.importorskip("torch")
    pyg_data = pytest.importorskip("torch_geometric.data")
    data = pyg_data.Data(
        edge_index=torch.from_numpy(np.asarray(graph.edge_index).astype(np.int64)),
        num_nodes=graph.num_nodes,
    )
    if graph.node_features is not None:
        data.x = torch.from_numpy(np.asarray(graph.node_features))
    if getattr(graph, "edge_weights", None) is not None:
        data.edge_weight = torch.from_numpy(np.asarray(graph.edge_weights))
    return data


def test_repr():
    assert str(OneHotDegree(5)) == "OneHotDegree(5)"
    assert str(Constant(2.0)) == "Constant(value=2.0)"
    assert str(AddRandomWalkPE(8)) == "AddRandomWalkPE(walk_length=8)"
    assert str(GCNNorm()) == "GCNNorm(add_self_loops=True)"


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("in_degree", [False, True])
@pytest.mark.parametrize("cat", [False, True])
@pytest.mark.parametrize("features", [None, "1d", "2d"])
def test_one_hot_degree_matches_pyg(seed, in_degree, cat, features):
    pyg_transforms = pytest.importorskip("torch_geometric.transforms")

    num_nodes = 15  # the last 3 nodes are isolated
    node_features = None
    if features == "1d":
        node_features = mx.arange(num_nodes).astype(mx.float32)
    elif features == "2d":
        node_features = mx.random.normal((num_nodes, 3))
    graph = GraphData(
        edge_index=mx.array(random_edges(seed)),
        node_features=node_features,
        num_nodes=num_nodes,
    )
    expected = pyg_transforms.OneHotDegree(10, in_degree=in_degree, cat=cat)(
        to_pyg(graph)
    )

    result = OneHotDegree(10, in_degree=in_degree, cat=cat)(graph)

    assert result.node_features.dtype == mx.float32
    assert np.array_equal(np.asarray(result.node_features), expected.x.numpy())
    assert graph.node_features is node_features  # the input is untouched


def test_one_hot_degree_basic():
    graph = GraphData(edge_index=mx.array([[0, 0, 1], [1, 2, 2]], dtype=mx.uint32))
    out_degree = OneHotDegree(2)(graph)
    assert out_degree.node_features.tolist() == [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    in_degree = OneHotDegree(2, in_degree=True)(graph)
    assert in_degree.node_features.tolist() == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    # The encoding takes the dtype of the features it is appended to.
    graph.node_features = mx.array([[7], [8], [9]], dtype=mx.int32)
    result = OneHotDegree(2)(graph)
    assert result.node_features.dtype == mx.int32
    assert result.node_features.tolist() == [[7, 0, 0, 1], [8, 0, 1, 0], [9, 1, 0, 0]]


def test_one_hot_degree_edge_cases():
    graph = GraphData(edge_index=mx.zeros((2, 0), dtype=mx.int32), num_nodes=2)
    assert OneHotDegree(1)(graph).node_features.tolist() == [[1, 0], [1, 0]]

    graph = GraphData(edge_index=mx.array([[0, 0, 0], [1, 2, 3]]))
    with pytest.raises(ValueError, match="max_degree"):
        OneHotDegree(2)(graph)
    assert OneHotDegree(2, in_degree=True)(graph).node_features.shape == (4, 3)


@pytest.mark.parametrize("cat", [False, True])
@pytest.mark.parametrize("features", [None, "1d", "2d"])
def test_constant_matches_pyg(cat, features):
    pyg_transforms = pytest.importorskip("torch_geometric.transforms")

    num_nodes = 6
    node_features = None
    if features == "1d":
        node_features = mx.arange(num_nodes).astype(mx.float32)
    elif features == "2d":
        node_features = mx.random.normal((num_nodes, 3))
    graph = GraphData(
        edge_index=mx.array([[0, 1], [1, 2]]),
        node_features=node_features,
        num_nodes=num_nodes,
    )
    expected = pyg_transforms.Constant(2.5, cat=cat)(to_pyg(graph))

    result = Constant(2.5, cat=cat)(graph)

    assert result.node_features.dtype == mx.float32
    assert np.array_equal(np.asarray(result.node_features), expected.x.numpy())
    assert graph.node_features is node_features


def test_constant_basic():
    graph = GraphData(edge_index=mx.array([[0, 1], [1, 2]]))
    assert Constant()(graph).node_features.tolist() == [[1.0], [1.0], [1.0]]
    # Skipped node ids are covered.
    graph = GraphData(edge_index=mx.array([[0], [4]]))
    assert Constant()(graph).node_features.shape == (5, 1)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("dtype", [mx.int64, mx.uint32])
def test_add_random_walk_pe_matches_pyg(seed, dtype):
    pyg_transforms = pytest.importorskip("torch_geometric.transforms")

    num_nodes = 14  # the last 2 nodes are isolated
    graph = GraphData(
        edge_index=mx.array(simple_edges(seed), dtype=dtype), num_nodes=num_nodes
    )
    expected = pyg_transforms.AddRandomWalkPE(6, attr_name="pe")(to_pyg(graph))

    result = AddRandomWalkPE(6, attr_name="pe")(graph)

    assert result.pe.shape == (num_nodes, 6)
    assert result.pe.dtype == mx.float32
    # Computed on the host (no GPU matmul): tight tolerance.
    assert np.allclose(np.asarray(result.pe), expected.pe.numpy(), atol=1e-5)
    assert result.node_features is None


@pytest.mark.parametrize("seed", range(4))
def test_add_random_walk_pe_weighted(seed):
    # Diagonals of the powers of D^-1 A, with A the weighted adjacency matrix.
    num_nodes = 14  # the last 2 nodes are isolated
    edges = simple_edges(seed)
    weights = np.random.default_rng(seed).uniform(0.1, 3.0, size=edges.shape[1])
    graph = GraphData(
        edge_index=mx.array(edges),
        edge_weights=mx.array(weights.astype(np.float32)),
        num_nodes=num_nodes,
    )

    adjacency = np.zeros((num_nodes, num_nodes))
    adjacency[edges[0], edges[1]] = weights
    degree = adjacency.sum(axis=1, keepdims=True)
    transition = np.divide(
        adjacency, degree, out=np.zeros_like(adjacency), where=degree > 0
    )
    # Every row with edges is a probability distribution.
    assert np.allclose(transition.sum(axis=1)[:12][degree[:12, 0] > 0], 1.0)
    expected = np.stack(
        [np.diagonal(np.linalg.matrix_power(transition, k)) for k in range(1, 7)],
        axis=1,
    )

    result = AddRandomWalkPE(6)(graph).random_walk_pe
    assert np.allclose(np.asarray(result), expected, atol=1e-5)


def test_add_random_walk_pe_basic():
    # Path 0 - 1 - 2: after 2 steps, the walk is back with probability 0.5 from
    # the ends and 1 from the middle. It can never be back after 1 or 3 steps.
    graph = GraphData(edge_index=mx.array([[0, 1, 1, 2], [1, 0, 2, 1]]))
    result = AddRandomWalkPE(walk_length=3)(graph)
    assert result.random_walk_pe.tolist() == [[0, 0.5, 0], [0, 1, 0], [0, 0.5, 0]]

    # A self-loop on an otherwise isolated node always comes back.
    graph = GraphData(edge_index=mx.array([[0], [0]]), num_nodes=2)
    assert AddRandomWalkPE(2)(graph).random_walk_pe.tolist() == [[1, 1], [0, 0]]


def test_add_random_walk_pe_duplicate_edges_are_summed():
    # Two parallel edges 0 -> 1 and one edge 0 -> 2: the walk goes to node 1
    # with probability 2/3. Node 1 always comes back to node 0.
    graph = GraphData(edge_index=mx.array([[0, 0, 0, 1], [1, 1, 2, 0]]))
    result = AddRandomWalkPE(2)(graph).random_walk_pe
    assert np.allclose(result.tolist(), [[0, 2 / 3], [0, 2 / 3], [0, 0]], atol=1e-6)


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("block_entries", [1, 60, 2**23])
def test_add_random_walk_pe_sparse_path_matches_dense(seed, block_entries):
    pytest.importorskip("scipy")

    class SparseRandomWalkPE(AddRandomWalkPE):
        DENSE_THRESHOLD = 0
        BLOCK_ENTRIES = block_entries

    graph = GraphData(edge_index=mx.array(random_edges(seed, 20, 70)), num_nodes=23)
    graph.edge_weights = mx.random.uniform(0.5, 2.0, (70,))

    dense = AddRandomWalkPE(5)(graph).random_walk_pe
    sparse = SparseRandomWalkPE(5)(graph).random_walk_pe

    assert sparse.shape == (23, 5)
    assert np.allclose(np.asarray(sparse), np.asarray(dense), atol=1e-6)


def test_add_random_walk_pe_as_node_features_and_empty():
    graph = GraphData(
        edge_index=mx.array([[0, 1, 1, 2], [1, 0, 2, 1]]),
        node_features=mx.ones((3, 2)),
    )
    result = AddRandomWalkPE(2, attr_name=None)(graph)
    assert result.node_features.tolist() == [
        [1, 1, 0, 0.5],
        [1, 1, 0, 1],
        [1, 1, 0, 0.5],
    ]
    assert not hasattr(result, "random_walk_pe")
    assert graph.node_features.shape == (3, 2)

    # Without node features, the encoding becomes the node features.
    graph = GraphData(edge_index=mx.array([[0, 1, 1, 2], [1, 0, 2, 1]]))
    assert AddRandomWalkPE(2, attr_name=None)(graph).node_features.shape == (3, 2)

    graph = GraphData(edge_index=mx.zeros((2, 0), dtype=mx.int32), num_nodes=3)
    assert AddRandomWalkPE(4)(graph).random_walk_pe.tolist() == [[0.0] * 4] * 3


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("add_self_loops", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_gcn_norm_matches_pyg(seed, add_self_loops, weighted):
    pyg_transforms = pytest.importorskip("torch_geometric.transforms")

    num_nodes = 15  # the last 3 nodes are isolated
    graph = GraphData(
        edge_index=mx.array(random_edges(seed), dtype=mx.uint32), num_nodes=num_nodes
    )
    if weighted:
        weights = np.random.default_rng(seed).uniform(0.5, 3.0, size=graph.num_edges)
        graph.edge_weights = mx.array(weights.astype(np.float32))
    expected = pyg_transforms.GCNNorm(add_self_loops=add_self_loops)(to_pyg(graph))

    result = GCNNorm(add_self_loops=add_self_loops)(graph)

    assert result.edge_index.dtype == mx.uint32
    assert np.array_equal(np.asarray(result.edge_index), expected.edge_index.numpy())
    assert result.edge_weights.dtype == mx.float32
    assert np.allclose(
        np.asarray(result.edge_weights),
        expected.edge_weight.numpy(),
        rtol=1e-5,
        atol=1e-6,
    )
    assert result.num_nodes == num_nodes
    # The input graph is untouched.
    assert graph.num_edges == 30
    assert weighted or not hasattr(graph, "edge_weights")


def test_gcn_norm_basic():
    graph = GraphData(edge_index=mx.array([[0, 1], [1, 0]]))
    result = GCNNorm()(graph)
    assert result.edge_index.tolist() == [[0, 1, 0, 1], [1, 0, 0, 1]]
    assert result.edge_weights.tolist() == [0.5, 0.5, 0.5, 0.5]

    result = GCNNorm(add_self_loops=False)(graph)
    assert result.edge_index.tolist() == [[0, 1], [1, 0]]
    assert result.edge_weights.tolist() == [1.0, 1.0]


def test_gcn_norm_is_symmetric_normalization():
    # Dense check of D^-1/2 (A + I) D^-1/2 on an undirected graph.
    edges = np.array([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    result = GCNNorm()(GraphData(edge_index=mx.array(edges)))

    adjacency = np.eye(4)
    adjacency[edges[0], edges[1]] = 1
    inv_sqrt = 1 / np.sqrt(adjacency.sum(axis=0))
    expected = inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]

    dense = np.zeros((4, 4))
    index = np.asarray(result.edge_index)
    dense[index[0], index[1]] = np.asarray(result.edge_weights)
    assert np.allclose(dense, expected, atol=1e-6)


def test_gcn_norm_keeps_edge_attributes_aligned():
    # Edge 1 is a self-loop on node 2: it moves to the self-loop block and
    # keeps its weight and features. New self-loops get zeros as features.
    graph = GraphData(
        edge_index=mx.array([[0, 2, 1], [1, 2, 0]]),
        edge_features=mx.array([[10.0], [20.0], [30.0]]),
        edge_weights=mx.array([1.0, 4.0, 1.0]),
        node_features=mx.ones((4, 1)),
    )
    result = GCNNorm()(graph)

    assert result.edge_index.tolist() == [[0, 1, 0, 1, 2, 3], [1, 0, 0, 1, 2, 3]]
    assert result.edge_features.tolist() == [
        [10.0],
        [30.0],
        [0.0],
        [0.0],
        [20.0],
        [0.0],
    ]
    # In-degrees with self-loops: 2, 2, 4 and 1.
    assert np.allclose(
        result.edge_weights.tolist(), [0.5, 0.5, 0.5, 0.5, 1.0, 1.0], atol=1e-6
    )

    # Without self-loops, nothing moves.
    result = GCNNorm(add_self_loops=False)(graph)
    assert result.edge_index is graph.edge_index
    assert result.edge_features is graph.edge_features
    assert np.allclose(result.edge_weights.tolist(), [1.0, 1.0, 1.0], atol=1e-6)


def test_gcn_norm_empty_edges_and_invalid_weights():
    graph = GraphData(edge_index=mx.zeros((2, 0), dtype=mx.int32), num_nodes=2)
    result = GCNNorm()(graph)
    assert result.edge_index.tolist() == [[0, 1], [0, 1]]
    assert result.edge_weights.tolist() == [1.0, 1.0]

    result = GCNNorm(add_self_loops=False)(graph)
    assert result.edge_index.shape == (2, 0)
    assert result.edge_weights.shape == (0,)

    graph = GraphData(
        edge_index=mx.array([[0, 1], [1, 0]]), edge_weights=mx.ones((2, 3))
    )
    with pytest.raises(ValueError):
        GCNNorm()(graph)


@pytest.mark.parametrize("dtype", [mx.int32, mx.uint32, mx.int64, mx.uint64])
def test_transforms_accept_any_edge_index_dtype(dtype):
    graph = GraphData(edge_index=mx.array([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=dtype))

    assert OneHotDegree(2)(graph).node_features.tolist() == [
        [0, 1, 0],
        [0, 0, 1],
        [0, 1, 0],
    ]
    assert Constant()(graph).node_features.shape == (3, 1)
    assert AddRandomWalkPE(2)(graph).random_walk_pe.tolist() == [
        [0, 0.5],
        [0, 1],
        [0, 0.5],
    ]
    normalized = GCNNorm()(graph)
    assert normalized.edge_index.dtype == dtype
    assert normalized.edge_index.tolist() == [
        [0, 1, 1, 2, 0, 1, 2],
        [1, 0, 2, 1, 0, 1, 2],
    ]
    assert normalized.edge_weights.shape == (7,)
