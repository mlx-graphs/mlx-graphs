import mlx.core as mx
import numpy as np
import pytest

from mlx_graphs.utils.dense import to_dense_adj, to_dense_batch


def random_batch(seed: int):
    """Node features, sorted batch indices and edges of a batch of 4 graphs,
    the third of which is empty. Edges have duplicates and self-loops."""
    rng = np.random.default_rng(seed)
    sizes = [5, 3, 0, 7]
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    batch = np.repeat(np.arange(len(sizes)), sizes)
    features = rng.normal(size=(sum(sizes), 4)).astype(np.float32)
    edges = np.concatenate(
        [
            rng.integers(0, size, size=(2, 3 * size)) + offset
            for size, offset in zip(sizes, offsets)
        ],
        axis=1,
    )
    return features, batch, edges


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("max_num_nodes", [None, 4, 7, 10])
@pytest.mark.parametrize("fill_value", [0.0, -1.5])
def test_to_dense_batch_matches_pyg(seed, max_num_nodes, fill_value):
    torch = pytest.importorskip("torch")
    pyg_utils = pytest.importorskip("torch_geometric.utils")

    features, batch, _ = random_batch(seed)
    expected_dense, expected_mask = pyg_utils.to_dense_batch(
        torch.from_numpy(features),
        torch.from_numpy(batch),
        fill_value=fill_value,
        max_num_nodes=max_num_nodes,
    )
    dense, mask = to_dense_batch(
        mx.array(features),
        mx.array(batch),
        fill_value=fill_value,
        max_num_nodes=max_num_nodes,
    )

    assert dense.shape == tuple(expected_dense.shape)
    assert np.array_equal(np.asarray(dense), expected_dense.numpy())
    assert mask.dtype == mx.bool_
    assert np.array_equal(np.asarray(mask), expected_mask.numpy())


def test_to_dense_batch_basic():
    node_features = mx.array([[1.0], [2.0], [3.0]])
    dense, mask = to_dense_batch(node_features, mx.array([0, 0, 1]))
    assert dense.tolist() == [[[1.0], [2.0]], [[3.0], [0.0]]]
    assert mask.tolist() == [[True, True], [True, False]]


def test_to_dense_batch_single_graph_and_empty():
    torch = pytest.importorskip("torch")
    pyg_utils = pytest.importorskip("torch_geometric.utils")

    features = np.arange(12, dtype=np.float32).reshape(4, 3)
    dense, mask = to_dense_batch(mx.array(features))
    expected_dense, expected_mask = pyg_utils.to_dense_batch(torch.from_numpy(features))
    assert np.array_equal(np.asarray(dense), expected_dense.numpy())
    assert np.array_equal(np.asarray(mask), expected_mask.numpy())

    dense, mask = to_dense_batch(mx.zeros((0, 3)))
    assert dense.shape == (1, 0, 3)
    assert mask.shape == (1, 0)

    dense, mask = to_dense_batch(mx.zeros((0, 3)), max_num_nodes=2)
    assert dense.shape == (1, 2, 3)
    assert mask.tolist() == [[False, False]]


def test_to_dense_batch_keeps_dtype_and_feature_shape():
    # Integer features, 1-dimensional.
    labels = mx.array([3, 1, 4, 1, 5], dtype=mx.int32)
    dense, mask = to_dense_batch(labels, mx.array([0, 0, 0, 1, 1]), fill_value=-1)
    assert dense.dtype == mx.int32
    assert dense.tolist() == [[3, 1, 4], [1, 5, -1]]
    assert mask.tolist() == [[True, True, True], [True, True, False]]
    # The default (float) fill value does not change the dtype either.
    dense, _ = to_dense_batch(labels, mx.array([0, 0, 0, 1, 1]))
    assert dense.dtype == mx.int32
    assert dense.tolist() == [[3, 1, 4], [1, 5, 0]]

    # Features with several trailing dimensions.
    features = mx.arange(24).reshape(3, 2, 4)
    dense, _ = to_dense_batch(features, mx.array([0, 1, 1]))
    assert dense.shape == (2, 2, 2, 4)
    assert mx.array_equal(dense[0, 0], features[0])
    assert mx.array_equal(dense[1], features[1:])


def test_to_dense_batch_unsorted_batch_indices():
    # The nodes of a graph do not have to be contiguous: they keep their
    # relative order within the graph.
    node_features = mx.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    batch_indices = mx.array([1, 0, 1, 0, 1])
    dense, mask = to_dense_batch(node_features, batch_indices)
    assert dense.squeeze(-1).tolist() == [[1.0, 3.0, 0.0], [0.0, 2.0, 4.0]]
    assert mask.tolist() == [[True, True, False], [True, True, True]]


def test_to_dense_batch_invalid_batch_indices():
    with pytest.raises(ValueError):
        to_dense_batch(mx.ones((3, 2)), mx.array([0, 1]))


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("max_num_nodes", [None, 4, 7, 10])
@pytest.mark.parametrize("feature_shape", [None, (), (3,)])
def test_to_dense_adj_matches_pyg(seed, max_num_nodes, feature_shape):
    torch = pytest.importorskip("torch")
    pyg_utils = pytest.importorskip("torch_geometric.utils")

    _, batch, edges = random_batch(seed)
    features = None
    if feature_shape is not None:
        features = np.random.default_rng(seed).normal(
            size=(edges.shape[1], *feature_shape)
        )
        features = features.astype(np.float32)

    expected = pyg_utils.to_dense_adj(
        torch.from_numpy(edges),
        torch.from_numpy(batch),
        None if features is None else torch.from_numpy(features),
        max_num_nodes=max_num_nodes,
    )
    adj = to_dense_adj(
        mx.array(edges),
        mx.array(batch),
        None if features is None else mx.array(features),
        max_num_nodes=max_num_nodes,
    )

    assert adj.shape == tuple(expected.shape)
    assert adj.dtype == mx.float32
    # Duplicate edges are summed, in an unspecified order.
    assert np.allclose(np.asarray(adj), expected.numpy(), rtol=1e-5, atol=1e-5)


def test_to_dense_adj_basic():
    edge_index = mx.array([[0, 0, 1, 2, 3], [0, 1, 0, 3, 2]])
    adj = to_dense_adj(edge_index, mx.array([0, 0, 1, 1]))
    assert adj.tolist() == [[[1, 1], [1, 0]], [[0, 1], [1, 0]]]


def test_to_dense_adj_single_graph():
    torch = pytest.importorskip("torch")
    pyg_utils = pytest.importorskip("torch_geometric.utils")

    edges = np.array([[0, 0, 2, 2, 3], [1, 1, 2, 0, 1]])  # duplicate + self-loop
    adj = to_dense_adj(mx.array(edges))
    expected = pyg_utils.to_dense_adj(torch.from_numpy(edges))
    assert adj.shape == (1, 4, 4)
    assert np.array_equal(np.asarray(adj), expected.numpy())
    assert adj[0, 0, 1].item() == 2.0

    # Isolated trailing nodes are only known through `max_num_nodes`.
    adj = to_dense_adj(mx.array(edges), max_num_nodes=6)
    assert adj.shape == (1, 6, 6)
    assert adj.sum().item() == 5.0


def test_to_dense_adj_empty_edges():
    no_edges = mx.zeros((2, 0), dtype=mx.int32)
    assert to_dense_adj(no_edges).shape == (1, 0, 0)

    adj = to_dense_adj(no_edges, mx.array([0, 0, 1]))
    assert adj.shape == (2, 2, 2)
    assert adj.sum().item() == 0.0

    adj = to_dense_adj(no_edges, mx.array([0, 0, 1]), mx.zeros((0, 5)))
    assert adj.shape == (2, 2, 2, 5)


def test_to_dense_adj_keeps_feature_dtype():
    edge_index = mx.array([[0, 1, 1], [1, 0, 0]])
    features = mx.array([2, 3, 4], dtype=mx.int32)
    adj = to_dense_adj(edge_index, edge_features=features)
    assert adj.dtype == mx.int32
    assert adj.tolist() == [[[0, 2], [7, 0]]]


def test_to_dense_adj_unsorted_batch_indices():
    # Graph 0 holds nodes 1 and 3, graph 1 holds nodes 0, 2 and 4.
    batch_indices = mx.array([1, 0, 1, 0, 1])
    edge_index = mx.array([[1, 0, 4], [3, 2, 0]])
    adj = to_dense_adj(edge_index, batch_indices)
    expected = np.zeros((2, 3, 3), dtype=np.float32)
    expected[0, 0, 1] = 1  # 1 -> 3
    expected[1, 0, 1] = 1  # 0 -> 2
    expected[1, 2, 0] = 1  # 4 -> 0
    assert np.array_equal(np.asarray(adj), expected)


def test_to_dense_adj_invalid_inputs():
    edge_index = mx.array([[0, 1], [1, 2]])
    with pytest.raises(ValueError):
        to_dense_adj(edge_index, mx.array([0, 0]))  # node 2 has no graph
    with pytest.raises(ValueError):
        to_dense_adj(edge_index, edge_features=mx.ones((3,)))
    with pytest.raises(ValueError):
        to_dense_adj(mx.array([0, 1, 2]))
