import mlx.core as mx
import numpy as np
import pytest

from mlx_graphs.utils.loop import (
    add_remaining_self_loops,
    contains_isolated_nodes,
    contains_self_loops,
    remove_isolated_nodes,
    segregate_self_loops,
)


def random_edges(seed: int, num_nodes: int = 10, num_edges: int = 30) -> np.ndarray:
    """Random edges with duplicates and self-loops (several per node)."""
    return np.random.default_rng(seed).integers(0, num_nodes, size=(2, num_edges))


def random_features(seed: int, num_edges: int, shape=(3,)) -> np.ndarray:
    rng = np.random.default_rng(1000 + seed)
    return rng.normal(size=(num_edges, *shape)).astype(np.float32)


def test_contains_self_loops_and_isolated_nodes():
    pyg_utils = pytest.importorskip("torch_geometric.utils")
    torch = pytest.importorskip("torch")

    for seed in range(5):
        edges = random_edges(seed, 12, 8)
        for num_nodes in (None, 15):
            assert contains_isolated_nodes(
                mx.array(edges), num_nodes
            ) == pyg_utils.contains_isolated_nodes(torch.tensor(edges), num_nodes)
        assert contains_self_loops(mx.array(edges)) == pyg_utils.contains_self_loops(
            torch.tensor(edges)
        )

    assert not contains_self_loops(mx.array([[0, 1], [1, 0]]))
    assert contains_self_loops(mx.array([[0, 1], [1, 1]]))
    assert not contains_self_loops(mx.zeros((2, 0), dtype=mx.int32))
    assert contains_isolated_nodes(mx.array([[0, 1, 2], [1, 0, 2]]))
    assert not contains_isolated_nodes(mx.array([[0, 1, 2], [1, 2, 0]]))


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize(
    "fill_value", [None, 2.5, "add", "sum", "mean", "max", "min", "array"]
)
@pytest.mark.parametrize("feature_shape", [(), (3,)])
def test_add_remaining_self_loops_matches_pyg(seed, fill_value, feature_shape):
    torch = pytest.importorskip("torch")
    pyg_utils = pytest.importorskip("torch_geometric.utils")

    edges = random_edges(seed)
    features = random_features(seed, edges.shape[1], feature_shape)
    num_nodes = 12  # nodes 10 and 11 have no edge

    if fill_value == "array":
        fill = np.arange(1, 1 + int(np.prod(feature_shape)), dtype=np.float32)
        fill = fill.reshape(feature_shape)
        fill_mx, fill_pt = mx.array(fill), torch.tensor(fill)
    else:
        fill_mx = fill_pt = fill_value

    out_index, out_features = add_remaining_self_loops(
        mx.array(edges), mx.array(features), fill_mx, num_nodes
    )
    exp_index, exp_features = pyg_utils.add_remaining_self_loops(
        torch.tensor(edges), torch.tensor(features), fill_pt, num_nodes
    )
    assert np.array_equal(np.asarray(out_index), exp_index.numpy())
    assert np.allclose(np.asarray(out_features), exp_features.numpy(), atol=1e-6)

    out_index, none = add_remaining_self_loops(mx.array(edges), num_nodes=num_nodes)
    assert none is None
    assert np.array_equal(np.asarray(out_index), exp_index.numpy())


def test_add_remaining_self_loops_dtypes_and_errors():
    edge_index = mx.array([[0, 1, 1], [1, 0, 1]], dtype=mx.uint32)
    weights = mx.array([1, 2, 3], dtype=mx.int32)
    out_index, out_weights = add_remaining_self_loops(edge_index, weights, 7)
    assert out_index.dtype == mx.uint32
    assert out_weights.dtype == mx.int32
    assert out_index.tolist() == [[0, 1, 0, 1], [1, 0, 0, 1]]
    assert out_weights.tolist() == [1, 2, 7, 3]

    empty, _ = add_remaining_self_loops(mx.zeros((2, 0), dtype=mx.int32), num_nodes=2)
    assert empty.tolist() == [[0, 1], [0, 1]]

    with pytest.raises(ValueError):
        add_remaining_self_loops(edge_index, num_nodes=1)
    with pytest.raises(ValueError):
        add_remaining_self_loops(edge_index, weights, fill_value=object())


@pytest.mark.parametrize("seed", range(4))
def test_segregate_self_loops_matches_pyg(seed):
    torch = pytest.importorskip("torch")
    pyg_utils = pytest.importorskip("torch_geometric.utils")

    edges = random_edges(seed)
    features = random_features(seed, edges.shape[1])
    out = segregate_self_loops(mx.array(edges), mx.array(features))
    expected = pyg_utils.segregate_self_loops(
        torch.tensor(edges), torch.tensor(features)
    )
    for a, b in zip(out, expected):
        assert np.array_equal(np.asarray(a), b.numpy())

    out = segregate_self_loops(mx.array(edges))
    assert out[1] is None and out[3] is None
    assert np.array_equal(np.asarray(out[2]), expected[2].numpy())


def test_segregate_self_loops_without_loops():
    edge_index, _, loop_index, loop_features = segregate_self_loops(
        mx.array([[0, 1], [1, 0]]), mx.array([1.0, 2.0])
    )
    assert edge_index.tolist() == [[0, 1], [1, 0]]
    assert loop_index.shape == (2, 0)
    assert loop_features.shape == (0,)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("num_nodes", [None, 14])
def test_remove_isolated_nodes_matches_pyg(seed, num_nodes):
    torch = pytest.importorskip("torch")
    pyg_utils = pytest.importorskip("torch_geometric.utils")

    # Few edges so that some nodes are isolated or only have self-loops.
    edges = random_edges(seed, 12, 9)
    edges[:, :3] = edges[0, :3]  # a few self-loops, possibly on isolated nodes
    features = random_features(seed, edges.shape[1])

    out_index, out_features, mask = remove_isolated_nodes(
        mx.array(edges), mx.array(features), num_nodes
    )
    exp_index, exp_features, exp_mask = pyg_utils.remove_isolated_nodes(
        torch.tensor(edges), torch.tensor(features), num_nodes
    )
    assert np.array_equal(np.asarray(mask), exp_mask.numpy())
    assert np.array_equal(np.asarray(out_index), exp_index.numpy())
    assert np.allclose(np.asarray(out_features), exp_features.numpy())

    out_index, none, _ = remove_isolated_nodes(mx.array(edges), num_nodes=num_nodes)
    assert none is None
    assert np.array_equal(np.asarray(out_index), exp_index.numpy())


def test_remove_isolated_nodes_keeps_last_self_loop():
    edge_index = mx.array([[0, 1, 1, 1, 2], [1, 0, 1, 1, 2]])
    features = mx.array([0.0, 1.0, 2.0, 3.0, 4.0])
    out_index, out_features, mask = remove_isolated_nodes(edge_index, features, 4)
    assert out_index.tolist() == [[0, 1, 1], [1, 0, 1]]
    assert out_features.tolist() == [0.0, 1.0, 3.0]
    assert mask.tolist() == [True, True, False, False]

    out_index, _, mask = remove_isolated_nodes(mx.array([[0], [0]]), num_nodes=2)
    assert out_index.shape == (2, 0)
    assert mask.tolist() == [False, False]
