import mlx.core as mx
import numpy as np
import pytest

from mlx_graphs.utils.subgraph import k_hop_subgraph, subgraph


def random_edges(seed: int, num_nodes: int = 12, num_edges: int = 40) -> np.ndarray:
    """Random edges with duplicates and self-loops."""
    return np.random.default_rng(seed).integers(0, num_nodes, size=(2, num_edges))


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("relabel_nodes", [False, True])
@pytest.mark.parametrize("as_mask", [False, True])
def test_subgraph_matches_pyg(seed, relabel_nodes, as_mask):
    torch = pytest.importorskip("torch")
    pyg_utils = pytest.importorskip("torch_geometric.utils")

    rng = np.random.default_rng(100 + seed)
    num_nodes = 12
    edges = random_edges(seed, num_nodes)
    features = rng.normal(size=(edges.shape[1], 3)).astype(np.float32)
    # Unsorted node indices: relabeling follows the order of `subset`.
    subset = rng.permutation(num_nodes)[:7]
    if as_mask:
        mask = np.zeros(num_nodes, dtype=bool)
        mask[subset] = True
        subset = mask

    expected_index, expected_features, expected_mask = pyg_utils.subgraph(
        torch.from_numpy(subset),
        torch.from_numpy(edges),
        torch.from_numpy(features),
        relabel_nodes=relabel_nodes,
        num_nodes=num_nodes,
        return_edge_mask=True,
    )
    edge_index, edge_features, edge_mask = subgraph(
        mx.array(subset),
        mx.array(edges),
        mx.array(features),
        relabel_nodes=relabel_nodes,
        num_nodes=num_nodes,
        return_edge_mask=True,
    )

    assert np.array_equal(np.asarray(edge_index), expected_index.numpy())
    assert np.array_equal(np.asarray(edge_features), expected_features.numpy())
    assert np.array_equal(np.asarray(edge_mask), expected_mask.numpy())
    assert edge_mask.dtype == mx.bool_


def test_subgraph_basic():
    edge_index = mx.array([[0, 1, 2, 3], [1, 2, 3, 0]])

    sub_index, sub_features = subgraph(mx.array([1, 2, 3]), edge_index)
    assert sub_index.tolist() == [[1, 2], [2, 3]]
    assert sub_features is None

    mask = mx.array([False, True, True, True])
    sub_index, _ = subgraph(mask, edge_index, relabel_nodes=True)
    assert sub_index.tolist() == [[0, 1], [1, 2]]


def test_subgraph_index_list_and_mask_agree():
    edges = mx.array(random_edges(0))
    index = [2, 5, 7, 9, 11]
    mask = np.zeros(12, dtype=bool)
    mask[index] = True

    from_list, _ = subgraph(index, edges, relabel_nodes=True)
    from_array, _ = subgraph(mx.array(index), edges, relabel_nodes=True)
    from_mask, _ = subgraph(mx.array(mask), edges, relabel_nodes=True)
    from_numpy_mask, _ = subgraph(mask, edges, relabel_nodes=True)

    assert mx.array_equal(from_list, from_array)
    assert mx.array_equal(from_list, from_mask)
    assert mx.array_equal(from_list, from_numpy_mask)


def test_subgraph_keeps_duplicates_and_self_loops():
    edge_index = mx.array([[0, 0, 1, 1, 2], [1, 1, 1, 2, 0]])
    features = mx.array([1.0, 2.0, 3.0, 4.0, 5.0])

    sub_index, sub_features, edge_mask = subgraph(
        [0, 1], edge_index, features, return_edge_mask=True
    )
    assert sub_index.tolist() == [[0, 0, 1], [1, 1, 1]]
    assert sub_features.tolist() == [1.0, 2.0, 3.0]
    assert edge_mask.tolist() == [True, True, True, False, False]


@pytest.mark.parametrize("dtype", [mx.uint32, mx.int32, mx.int64, mx.uint64])
@pytest.mark.parametrize("relabel_nodes", [False, True])
def test_subgraph_preserves_dtype(dtype, relabel_nodes):
    edge_index = mx.array([[0, 1, 2], [1, 2, 0]], dtype=dtype)
    sub_index, _ = subgraph([0, 1], edge_index, relabel_nodes=relabel_nodes)
    assert sub_index.dtype == dtype
    assert sub_index.tolist() == [[0], [1]]

    empty, _ = subgraph([2], edge_index, relabel_nodes=relabel_nodes)
    assert empty.dtype == dtype
    assert empty.shape == (2, 0)


def test_subgraph_empty_inputs():
    edge_index = mx.array([[0, 1, 2], [1, 2, 0]])
    features = mx.ones((3, 4))

    # Empty subset.
    sub_index, sub_features, edge_mask = subgraph(
        mx.zeros((0,), dtype=mx.int32), edge_index, features, return_edge_mask=True
    )
    assert sub_index.shape == (2, 0)
    assert sub_features.shape == (0, 4)
    assert edge_mask.tolist() == [False, False, False]

    # Empty edge set.
    no_edges = mx.zeros((2, 0), dtype=mx.int32)
    sub_index, sub_features, edge_mask = subgraph(
        [0, 3], no_edges, mx.zeros((0, 4)), relabel_nodes=True, return_edge_mask=True
    )
    assert sub_index.shape == (2, 0)
    assert sub_features.shape == (0, 4)
    assert edge_mask.shape == (0,)


def test_subgraph_isolated_nodes_in_subset():
    # Nodes 4 and 5 have no edge; node 5 is beyond the ids of `edge_index`.
    edge_index = mx.array([[0, 1], [1, 2]])
    sub_index, _ = subgraph([5, 1, 4, 2], edge_index, relabel_nodes=True)
    assert sub_index.tolist() == [[1], [3]]


def test_subgraph_invalid_inputs():
    edge_index = mx.array([[0, 1, 2], [1, 2, 0]])
    with pytest.raises(ValueError):
        subgraph(mx.array([True, False]), edge_index)  # mask too short
    with pytest.raises(ValueError):
        subgraph(mx.array([True, False, True]), edge_index, num_nodes=5)
    with pytest.raises(ValueError):
        subgraph([0, 1], edge_index, mx.ones((2, 1)))  # one feature per edge
    with pytest.raises(ValueError):
        subgraph([0, 1], mx.array([0, 1, 2]))  # not an edge index


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("num_hops", [0, 1, 2, 3])
@pytest.mark.parametrize("relabel_nodes", [False, True])
@pytest.mark.parametrize("directed", [False, True])
def test_k_hop_subgraph_matches_pyg(seed, num_hops, relabel_nodes, directed):
    torch = pytest.importorskip("torch")
    pyg_utils = pytest.importorskip("torch_geometric.utils")

    num_nodes = 30  # sparse enough for the hops to differ; node 29 may be isolated
    edges = random_edges(seed, num_nodes - 1, 45)
    node_idx = [int(edges[1, 0]), int(edges[0, 3]), 29]

    expected = pyg_utils.k_hop_subgraph(
        node_idx,
        num_hops,
        torch.from_numpy(edges),
        relabel_nodes=relabel_nodes,
        num_nodes=num_nodes,
        directed=directed,
    )
    result = k_hop_subgraph(
        mx.array(node_idx),
        num_hops,
        mx.array(edges),
        relabel_nodes=relabel_nodes,
        num_nodes=num_nodes,
        directed=directed,
    )

    for name, ours, theirs in zip(
        ("subset", "edge_index", "mapping", "edge_mask"), result, expected
    ):
        assert np.array_equal(np.asarray(ours), theirs.numpy()), name


def test_k_hop_subgraph_follows_edges_backwards():
    # Path 0 -> 1 -> 2 -> 3 -> 4: the messages reaching node 3 within two hops
    # come from nodes 1 and 2, never from node 4.
    edge_index = mx.array([[0, 1, 2, 3], [1, 2, 3, 4]])

    subset, sub_index, mapping, edge_mask = k_hop_subgraph(3, 2, edge_index)
    assert subset.tolist() == [1, 2, 3]
    assert sub_index.tolist() == [[1, 2], [2, 3]]
    assert mapping.tolist() == [2]
    assert edge_mask.tolist() == [False, True, True, False]

    subset, sub_index, mapping, _ = k_hop_subgraph(3, 2, edge_index, relabel_nodes=True)
    assert sub_index.tolist() == [[0, 1], [1, 2]]
    assert subset[mapping].tolist() == [3]


def test_k_hop_subgraph_node_idx_types():
    edge_index = mx.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    from_int = k_hop_subgraph(2, 1, edge_index)
    from_list = k_hop_subgraph([2], 1, edge_index)
    from_array = k_hop_subgraph(mx.array([2]), 1, edge_index)
    for a, b, c in zip(from_int, from_list, from_array):
        assert mx.array_equal(a, b)
        assert mx.array_equal(a, c)


def test_k_hop_subgraph_directed_keeps_only_followed_edges():
    # 0 -> 2, 1 -> 2 and 0 -> 1: the edge 0 -> 1 links two nodes of the 1-hop
    # subgraph of node 2 but is not on a path of length 1 to node 2.
    edge_index = mx.array([[0, 1, 0], [2, 2, 1]])

    _, undirected_index, _, undirected_mask = k_hop_subgraph(2, 1, edge_index)
    assert undirected_mask.tolist() == [True, True, True]
    assert undirected_index.shape == (2, 3)

    _, directed_index, _, directed_mask = k_hop_subgraph(
        2, 1, edge_index, directed=True
    )
    assert directed_mask.tolist() == [True, True, False]
    assert directed_index.tolist() == [[0, 1], [2, 2]]


@pytest.mark.parametrize("dtype", [mx.uint32, mx.int64, mx.uint64])
def test_k_hop_subgraph_empty_edges_and_dtype(dtype):
    no_edges = mx.zeros((2, 0), dtype=dtype)
    subset, sub_index, mapping, edge_mask = k_hop_subgraph(
        [3, 1], 2, no_edges, relabel_nodes=True, num_nodes=5
    )
    assert subset.tolist() == [1, 3]
    assert sub_index.shape == (2, 0)
    assert sub_index.dtype == dtype
    assert mapping.tolist() == [1, 0]
    assert edge_mask.shape == (0,)

    edge_index = mx.array([[0, 1], [1, 2]], dtype=dtype)
    subset, sub_index, _, _ = k_hop_subgraph(2, 1, edge_index)
    assert subset.dtype == dtype
    assert sub_index.dtype == dtype
