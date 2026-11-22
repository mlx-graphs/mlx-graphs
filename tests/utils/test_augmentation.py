import mlx.core as mx
import numpy as np
import pytest

from mlx_graphs.utils.augmentation import add_random_edge, mask_feature, shuffle_node
from mlx_graphs.utils.dropout import dropout_path


def random_edges(seed: int, num_nodes: int, num_edges: int) -> np.ndarray:
    edges = np.random.default_rng(seed).integers(0, num_nodes, size=(2, num_edges))
    return edges.astype(np.int32)


def edge_set(edges: np.ndarray) -> set:
    return set(map(tuple, np.asarray(edges).T.tolist()))


def test_shuffle_node():
    x = mx.arange(12).reshape(6, 2)
    out, perm = shuffle_node(x, training=False)
    assert mx.array_equal(out, x) and perm.tolist() == list(range(6))

    mx.random.seed(0)
    out, perm = shuffle_node(x)
    assert sorted(perm.tolist()) == list(range(6))
    assert mx.array_equal(out, x[perm])

    batch = mx.array([0, 0, 0, 1, 1, 2])
    for seed in range(5):
        mx.random.seed(seed)
        out, perm = shuffle_node(x, batch)
        perm_np = np.asarray(perm)
        assert mx.array_equal(out, x[perm])
        # Nodes stay within their graph.
        assert sorted(perm_np[:3].tolist()) == [0, 1, 2]
        assert sorted(perm_np[3:5].tolist()) == [3, 4]
        assert perm_np[5] == 5


def test_shuffle_node_is_random():
    x = mx.arange(100)
    batch = mx.array(np.repeat(np.arange(4), 25))
    mx.random.seed(0)
    _, perm_a = shuffle_node(x, batch)
    _, perm_b = shuffle_node(x, batch)
    assert not mx.array_equal(perm_a, perm_b)
    assert not mx.array_equal(perm_a, mx.arange(100))


@pytest.mark.parametrize("mode", ["row", "col", "all"])
def test_mask_feature(mode):
    x = mx.random.normal((200, 50)) + 10.0
    mx.random.seed(0)
    out, mask = mask_feature(x, p=0.3, mode=mode, fill_value=-1.0)
    expected_shape = {"row": (200, 1), "col": (1, 50), "all": (200, 50)}[mode]
    assert mask.shape == expected_shape and mask.dtype == mx.bool_
    full_mask = np.broadcast_to(np.asarray(mask), x.shape)
    assert np.array_equal(np.asarray(out)[full_mask], np.asarray(x)[full_mask])
    assert np.all(np.asarray(out)[~full_mask] == -1.0)
    assert abs(1 - full_mask.mean() - 0.3) < 0.15

    out, mask = mask_feature(x, p=0.3, mode=mode, training=False)
    assert mx.array_equal(out, x) and mask.shape == x.shape


def test_mask_feature_errors():
    with pytest.raises(ValueError):
        mask_feature(mx.ones((2, 2)), p=1.5)
    with pytest.raises(ValueError):
        mask_feature(mx.ones((2, 2)), mode="cols")
    out, _ = mask_feature(mx.ones((2, 2), mx.int32), p=1.0)
    assert out.dtype == mx.int32 and out.tolist() == [[0, 0], [0, 0]]


@pytest.mark.parametrize("force_undirected", [False, True])
def test_add_random_edge(force_undirected):
    edges = random_edges(0, 20, 40)
    mx.random.seed(0)
    out, added = add_random_edge(
        mx.array(edges), p=0.5, force_undirected=force_undirected
    )
    added_np = np.asarray(added)
    assert added.dtype == mx.int32
    assert np.array_equal(np.asarray(out)[:, :40], edges)
    assert np.array_equal(np.asarray(out)[:, 40:], added_np)
    assert added_np.shape[1] == 20
    assert not (edge_set(added_np) & edge_set(edges))
    assert np.all(added_np[0] != added_np[1])
    if force_undirected:
        assert edge_set(added_np) == edge_set(added_np[::-1])

    out, added = add_random_edge(mx.array(edges), p=0.5, training=False)
    assert mx.array_equal(out, mx.array(edges)) and added.shape == (2, 0)


def test_add_random_edge_bipartite():
    edges = np.array([[0, 1, 2, 3], [0, 0, 1, 1]], dtype=np.int32)
    out, added = add_random_edge(mx.array(edges), p=1.0, num_nodes=(4, 2))
    added_np = np.asarray(added)
    assert added_np.shape == (2, 4)
    assert np.all(added_np[1] < 2)
    assert edge_set(added_np) | edge_set(edges) == {
        (i, j) for i in range(4) for j in range(2)
    }
    with pytest.raises(ValueError):
        add_random_edge(mx.array(edges), force_undirected=True, num_nodes=(4, 2))
    with pytest.raises(ValueError):
        add_random_edge(mx.array(edges), p=-0.1)


def test_dropout_path_identity_cases():
    edge_index = mx.array(random_edges(0, 10, 30))
    for kwargs in ({"p": 0.0}, {"training": False}):
        out, mask = dropout_path(edge_index, **kwargs)
        assert mx.array_equal(out, edge_index)
        assert mask.tolist() == [True] * 30
    with pytest.raises(ValueError):
        dropout_path(edge_index, p=2.0)
    out, mask = dropout_path(mx.zeros((2, 0), dtype=mx.int32))
    assert out.shape == (2, 0) and mask.shape == (0,)


@pytest.mark.parametrize("seed", range(3))
def test_dropout_path_drops_walks(seed):
    edges = random_edges(seed, 30, 120)
    mx.random.seed(seed)
    out, mask = dropout_path(mx.array(edges), p=0.2, walk_length=3)
    mask_np = np.asarray(mask)
    assert mask_np.dtype == bool and mask_np.shape == (120,)
    assert np.array_equal(np.asarray(out), edges[:, mask_np])
    assert 0 < (~mask_np).sum() < 120
    # Duplicated edges are dropped together.
    dropped = edge_set(edges[:, ~mask_np])
    assert not (dropped & edge_set(edges[:, mask_np]))


def test_dropout_path_follows_paths():
    # On a directed path, a walk from i drops the edges i -> i+1 -> ... in a row.
    num_nodes = 50
    edges = np.stack([np.arange(num_nodes - 1), np.arange(1, num_nodes)])
    edges = edges.astype(np.int32)
    for seed in range(5):
        mx.random.seed(seed)
        _, mask = dropout_path(mx.array(edges), p=0.05, walk_length=4)
        dropped = np.nonzero(~np.asarray(mask))[0]
        if dropped.size == 0:
            continue
        # Each run of dropped edges has length >= min(4, edges until the end).
        runs = np.split(dropped, np.nonzero(np.diff(dropped) > 1)[0] + 1)
        for run in runs:
            assert run.size >= min(4, num_nodes - 1 - run[0])


def test_dropout_path_sorted_input_and_reproducibility():
    edges = np.unique(random_edges(1, 20, 60), axis=1)
    mx.random.seed(7)
    a = dropout_path(mx.array(edges), p=0.3, is_sorted=True)[1]
    mx.random.seed(7)
    b = dropout_path(mx.array(edges), p=0.3)[1]
    assert mx.array_equal(a, b)
