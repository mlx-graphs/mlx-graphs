import mlx.core as mx
import numpy as np
import pytest

import mlx_graphs.nn.pooling as pooling
import mlx_graphs.utils as utils
import mlx_graphs.utils.spatial as spatial
from mlx_graphs.utils import knn, knn_graph, radius, radius_graph

# PyG's knn and radius functions need pyg-lib (or torch_cluster), which is not
# a test dependency: the results are compared with a brute-force reference.


def _distances(x, y, cosine=False):
    if cosine:
        x = x / np.linalg.norm(x, axis=-1, keepdims=True)
        y = y / np.linalg.norm(y, axis=-1, keepdims=True)
        return 1.0 - y @ x.T
    return np.linalg.norm(y[:, None, :] - x[None, :, :], axis=-1)


def _reference_knn(x, y, k, bx, by, cosine=False, exclude_self=False):
    dist = _distances(x, y, cosine)
    pairs = []
    for i in range(y.shape[0]):
        candidates = [
            j for j in np.nonzero(bx == by[i])[0] if not (exclude_self and i == j)
        ]
        candidates = sorted(candidates, key=lambda j: dist[i, j])[:k]
        pairs += [(i, j) for j in candidates]
    return np.array(pairs, dtype=np.int64).reshape(-1, 2).T


def _reference_radius(x, y, r, bx, by, max_num_neighbors=None, exclude_self=False):
    dist = _distances(x, y)
    pairs = []
    for i in range(y.shape[0]):
        candidates = [
            j
            for j in np.nonzero((bx == by[i]) & (dist[i] < r))[0]
            if not (exclude_self and i == j)
        ]
        if max_num_neighbors is not None:
            candidates = sorted(candidates, key=lambda j: dist[i, j])
            candidates = candidates[:max_num_neighbors]
        pairs += [(i, j) for j in sorted(candidates)]
    return np.array(pairs, dtype=np.int64).reshape(-1, 2).T


def _points(rng, sizes, dim=3):
    batch = np.repeat(np.arange(len(sizes)), sizes).astype(np.int64)
    return rng.normal(size=(len(batch), dim)).astype(np.float32), batch


def test_exported_from_utils_and_pooling():
    for name in ("knn", "knn_graph", "radius", "radius_graph"):
        assert getattr(utils, name) is getattr(spatial, name)
        assert getattr(pooling, name) is getattr(spatial, name)


@pytest.mark.parametrize("k", [1, 3, 8])
@pytest.mark.parametrize("cosine", [False, True])
def test_knn(k, cosine):
    rng = np.random.default_rng(0)
    x, bx = _points(rng, [6, 2, 9])
    y, by = _points(rng, [4, 3, 5])

    out = knn(mx.array(x), mx.array(y), k, mx.array(bx), mx.array(by), cosine=cosine)
    expected = _reference_knn(x, y, k, bx, by, cosine=cosine)
    np.testing.assert_array_equal(np.array(out), expected)
    assert out.dtype == mx.int32

    # without batch vectors: a single graph
    out = knn(mx.array(x), mx.array(y), k)
    expected = _reference_knn(x, y, k, np.zeros_like(bx), np.zeros_like(by))
    np.testing.assert_array_equal(np.array(out), expected)


def test_knn_example():
    x = mx.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    y = mx.array([[-1.0, 0.0], [1.0, 0.0]])
    out = knn(x, y, 2, mx.array([0, 0, 0, 0]), mx.array([0, 0]))
    assert out[0].tolist() == [0, 0, 1, 1]
    assert sorted(out[1, :2].tolist()) == [0, 1]
    assert sorted(out[1, 2:].tolist()) == [2, 3]


@pytest.mark.parametrize("loop", [False, True])
@pytest.mark.parametrize("flow", ["source_to_target", "target_to_source"])
def test_knn_graph(loop, flow):
    rng = np.random.default_rng(1)
    x, batch = _points(rng, [5, 1, 7, 3])
    # nodes of the graphs do not need to be contiguous
    perm = rng.permutation(len(batch))
    x, batch = x[perm], batch[perm]

    out = knn_graph(mx.array(x), 3, mx.array(batch), loop=loop, flow=flow)
    expected = _reference_knn(x, x, 3, batch, batch, exclude_self=not loop)
    if flow == "source_to_target":
        expected = expected[::-1]
    np.testing.assert_array_equal(np.array(out), expected)

    # each node has min(k, graph size - 1) neighbors (+ itself with loops)
    centers = np.array(out[1] if flow == "source_to_target" else out[0])
    sizes = np.bincount(batch)
    expected_degree = np.minimum(3, sizes[batch] - (0 if loop else 1))
    np.testing.assert_array_equal(
        np.bincount(centers, minlength=len(batch)), expected_degree
    )


def test_knn_graph_invalid_flow():
    with pytest.raises(ValueError):
        knn_graph(mx.zeros((3, 2)), 2, flow="both")


@pytest.mark.parametrize("max_num_neighbors", [None, 2, 32])
def test_radius(max_num_neighbors):
    rng = np.random.default_rng(2)
    x, bx = _points(rng, [8, 3, 10], dim=2)
    y, by = _points(rng, [5, 2, 6], dim=2)

    out = radius(
        mx.array(x),
        mx.array(y),
        1.2,
        mx.array(bx),
        mx.array(by),
        max_num_neighbors=max_num_neighbors,
    )
    expected = _reference_radius(x, y, 1.2, bx, by, max_num_neighbors)
    np.testing.assert_array_equal(np.array(out), expected)


@pytest.mark.parametrize("loop", [False, True])
@pytest.mark.parametrize("max_num_neighbors", [None, 3])
def test_radius_graph(loop, max_num_neighbors):
    rng = np.random.default_rng(3)
    x, batch = _points(rng, [9, 4, 7], dim=2)

    out = radius_graph(
        mx.array(x),
        1.0,
        mx.array(batch),
        loop=loop,
        max_num_neighbors=max_num_neighbors,
        flow="target_to_source",
    )
    expected = _reference_radius(
        x, x, 1.0, batch, batch, max_num_neighbors, exclude_self=not loop
    )
    np.testing.assert_array_equal(np.array(out), expected)

    flipped = radius_graph(
        mx.array(x), 1.0, mx.array(batch), loop, max_num_neighbors=max_num_neighbors
    )
    np.testing.assert_array_equal(np.array(flipped), expected[::-1])


def test_radius_graph_example():
    x = mx.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    out = radius_graph(x, r=2.5)
    assert out.tolist() == [[1, 2, 0, 3, 0, 3, 1, 2], [0, 0, 1, 1, 2, 2, 3, 3]]


def test_blocks(monkeypatch):
    """Results do not depend on the size of the blocks of distances."""
    rng = np.random.default_rng(4)
    x, batch = _points(rng, [12, 3, 9, 7])
    reference_knn = np.array(knn_graph(mx.array(x), 4, mx.array(batch)))
    reference_radius = np.array(radius_graph(mx.array(x), 1.0, mx.array(batch)))

    for block_size in (1, 20, 150):
        monkeypatch.setattr(spatial, "_BLOCK_SIZE", block_size)
        out = knn_graph(mx.array(x), 4, mx.array(batch))
        np.testing.assert_array_equal(np.array(out), reference_knn)
        out = radius_graph(mx.array(x), 1.0, mx.array(batch))
        np.testing.assert_array_equal(np.array(out), reference_radius)


def test_points_far_from_origin():
    """Close points with large coordinates are still told apart."""
    rng = np.random.default_rng(5)
    x = (1000.0 + 0.01 * rng.normal(size=(30, 3))).astype(np.float32)
    batch = np.zeros(30, dtype=np.int64)
    out = knn_graph(mx.array(x), 3, flow="target_to_source")
    np.testing.assert_array_equal(
        np.array(out), _reference_knn(x, x, 3, batch, batch, exclude_self=True)
    )


def test_empty_and_missing_graphs():
    empty = knn(mx.zeros((0, 2)), mx.zeros((3, 2)), 2)
    assert empty.shape == (2, 0)
    assert knn_graph(mx.zeros((1, 2)), 2).shape == (2, 0)
    assert radius_graph(mx.zeros((0, 2)), 1.0).shape == (2, 0)

    # queries of graphs without points of x have no neighbors
    x = mx.array([[0.0], [1.0]])
    y = mx.array([[0.0], [1.0]])
    out = knn(x, y, 1, mx.array([0, 0]), mx.array([0, 1]))
    assert out.tolist() == [[0], [0]]


def test_large_batched_clouds_match_kdtree():
    """knn / radius on a few thousand points in unsorted batches, against
    scipy's exact KD-tree (float64)."""
    cKDTree = pytest.importorskip("scipy.spatial").cKDTree
    rng = np.random.default_rng(7)
    batch = rng.permutation(np.repeat(np.arange(5), [700, 50, 400, 300, 550]))
    x = rng.random((batch.size, 3)).astype(np.float32)
    by = rng.integers(0, 5, 400)
    y = rng.random((400, 3)).astype(np.float32)
    trees = {g: (np.nonzero(batch == g)[0], cKDTree(x[batch == g])) for g in range(5)}

    def neighbors(out):
        result = {}
        for q, n in np.asarray(out).T.tolist():
            result.setdefault(q, []).append(n)
        return result

    k = 16
    got = neighbors(knn(mx.array(x), mx.array(y), k, mx.array(batch), mx.array(by)))
    for i in range(y.shape[0]):
        ids, tree = trees[by[i]]
        dist, j = tree.query(y[i], k=k)
        found = np.linalg.norm(x[got[i]].astype(np.float64) - y[i], axis=-1)
        # Same distances, closest first (equal ids unless float32 ties)
        np.testing.assert_allclose(found, dist, atol=1e-5)
        assert set(got[i]) <= set(ids.tolist())

    got = neighbors(knn_graph(mx.array(x), k, mx.array(batch), flow="target_to_source"))
    for i in range(0, x.shape[0], 7):
        ids, tree = trees[batch[i]]
        _, j = tree.query(x[i], k=k + 1)
        assert set(got[i]) == set(ids[j].tolist()) - {i}

    r = 0.12
    got = neighbors(
        radius_graph(
            mx.array(x),
            r,
            mx.array(batch),
            max_num_neighbors=None,
            flow="target_to_source",
        )
    )
    for i in range(x.shape[0]):
        ids, tree = trees[batch[i]]
        expected = set(ids[tree.query_ball_point(x[i], r)].tolist()) - {i}
        assert set(got.get(i, [])) == expected

    # max_num_neighbors keeps the closest neighbors
    capped = neighbors(
        radius_graph(
            mx.array(x),
            r,
            mx.array(batch),
            max_num_neighbors=3,
            flow="target_to_source",
        )
    )
    for i, nbrs in got.items():
        dist = np.linalg.norm(x[nbrs] - x[i], axis=-1)
        closest = set(np.asarray(nbrs)[np.argsort(dist)[:3]].tolist())
        assert set(capped.get(i, [])) == closest
