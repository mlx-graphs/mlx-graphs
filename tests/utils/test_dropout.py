import mlx.core as mx
import numpy as np
import pytest

from mlx_graphs.utils.dropout import dropout_edge, dropout_node


def random_edges(seed: int, num_nodes: int, num_edges: int) -> np.ndarray:
    """Random edges with duplicates and self-loops."""
    return np.random.default_rng(seed).integers(0, num_nodes, size=(2, num_edges))


def test_dropout_edge_identity_cases():
    edge_index = mx.array(random_edges(0, 10, 30))

    for kwargs in ({"p": 0.0}, {"training": False}, {"p": 1.0, "training": False}):
        out, edge_mask = dropout_edge(edge_index, **kwargs)
        assert mx.array_equal(out, edge_index)
        assert edge_mask.dtype == mx.bool_
        assert edge_mask.tolist() == [True] * 30

    # As in PyG, the mask stays a boolean mask when nothing is dropped.
    out, edge_mask = dropout_edge(edge_index, force_undirected=True, training=False)
    assert mx.array_equal(out, edge_index)
    assert edge_mask.tolist() == [True] * 30


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("p", [0.2, 0.5, 0.9])
def test_dropout_edge_mask_is_consistent(seed, p):
    edges = random_edges(seed, 50, 5000)
    mx.random.seed(seed)

    out, edge_mask = dropout_edge(mx.array(edges), p=p)
    mask = np.asarray(edge_mask)

    assert mask.shape == (5000,) and mask.dtype == bool
    assert np.array_equal(np.asarray(out), edges[:, mask])
    assert abs(mask.mean() - (1 - p)) < 0.05


def test_dropout_edge_drop_everything():
    edge_index = mx.array(random_edges(0, 10, 30), dtype=mx.uint32)
    out, edge_mask = dropout_edge(edge_index, p=1.0)
    assert out.shape == (2, 0)
    assert out.dtype == mx.uint32
    assert edge_mask.tolist() == [False] * 30

    out, edge_id = dropout_edge(edge_index, p=1.0, force_undirected=True)
    assert out.shape == (2, 0)
    assert out.dtype == mx.uint32
    assert edge_id.shape == (0,)


@pytest.mark.parametrize("seed", range(3))
def test_dropout_edge_force_undirected(seed):
    edges = random_edges(seed, 20, 200)
    mx.random.seed(seed)

    out, edge_id = dropout_edge(mx.array(edges), p=0.4, force_undirected=True)
    out_np, edge_id_np = np.asarray(out), np.asarray(edge_id)

    num_kept = out_np.shape[1] // 2
    assert out_np.shape[1] == 2 * num_kept and num_kept > 0
    # The second half is the reverse of the first half.
    assert np.array_equal(out_np[:, num_kept:], out_np[::-1, :num_kept])
    # `edge_id` is the position of each returned edge in the input.
    assert edge_id_np.shape == (2 * num_kept,)
    assert np.array_equal(edge_id_np[:num_kept], edge_id_np[num_kept:])
    assert np.array_equal(edges[:, edge_id_np[:num_kept]], out_np[:, :num_kept])
    # Only one direction is sampled, and each input edge at most once.
    assert np.all(out_np[0, :num_kept] <= out_np[1, :num_kept])
    assert np.unique(edge_id_np[:num_kept]).size == num_kept


def test_dropout_edge_reproducible_with_seed():
    edge_index = mx.array(random_edges(0, 20, 200))
    mx.random.seed(3)
    first, _ = dropout_edge(edge_index)
    mx.random.seed(3)
    second, _ = dropout_edge(edge_index)
    mx.random.seed(4)
    third, _ = dropout_edge(edge_index)
    assert mx.array_equal(first, second)
    assert first.shape != third.shape or not mx.array_equal(first, third)


def test_dropout_edge_empty_and_invalid():
    no_edges = mx.zeros((2, 0), dtype=mx.int32)
    out, edge_mask = dropout_edge(no_edges, p=0.5)
    assert out.shape == (2, 0)
    assert edge_mask.shape == (0,)

    edge_index = mx.array([[0, 1], [1, 0]])
    for p in (-0.1, 1.5):
        with pytest.raises(ValueError):
            dropout_edge(edge_index, p=p)
        with pytest.raises(ValueError):
            dropout_node(edge_index, p=p)


def test_dropout_node_identity_cases():
    edge_index = mx.array(random_edges(0, 10, 30))

    for kwargs in ({"p": 0.0}, {"training": False}):
        out, edge_mask, node_mask = dropout_node(edge_index, num_nodes=12, **kwargs)
        assert mx.array_equal(out, edge_index)
        assert edge_mask.tolist() == [True] * 30
        assert node_mask.tolist() == [True] * 12


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("p", [0.2, 0.5, 0.9])
def test_dropout_node_masks_are_consistent(seed, p):
    num_nodes = 2000
    edges = random_edges(seed, num_nodes - 100, 3000)  # 100 isolated nodes
    mx.random.seed(seed)

    out, edge_mask, node_mask = dropout_node(mx.array(edges), p=p, num_nodes=num_nodes)
    edge_mask_np, node_mask_np = np.asarray(edge_mask), np.asarray(node_mask)

    assert node_mask_np.shape == (num_nodes,) and node_mask_np.dtype == bool
    assert abs(node_mask_np.mean() - (1 - p)) < 0.05
    # An edge is kept if and only if both its nodes are kept.
    expected = node_mask_np[edges[0]] & node_mask_np[edges[1]]
    assert np.array_equal(edge_mask_np, expected)
    # Edges are not relabeled.
    assert np.array_equal(np.asarray(out), edges[:, expected])


def test_dropout_node_drop_everything_and_empty():
    edge_index = mx.array(random_edges(0, 10, 30), dtype=mx.uint32)
    out, edge_mask, node_mask = dropout_node(edge_index, p=1.0)
    assert out.shape == (2, 0)
    assert out.dtype == mx.uint32
    assert not any(edge_mask.tolist())
    assert not any(node_mask.tolist())

    no_edges = mx.zeros((2, 0), dtype=mx.int32)
    out, edge_mask, node_mask = dropout_node(no_edges, p=0.5, num_nodes=4)
    assert out.shape == (2, 0)
    assert edge_mask.shape == (0,)
    assert node_mask.shape == (4,)

    out, edge_mask, node_mask = dropout_node(no_edges, p=0.5)
    assert out.shape == (2, 0)
    assert node_mask.shape == (0,)


def test_dropout_node_reproducible_with_seed():
    edge_index = mx.array(random_edges(0, 50, 200))
    mx.random.seed(3)
    first = dropout_node(edge_index)[2]
    mx.random.seed(3)
    second = dropout_node(edge_index)[2]
    mx.random.seed(4)
    third = dropout_node(edge_index)[2]
    assert mx.array_equal(first, second)
    assert not mx.array_equal(first, third)
