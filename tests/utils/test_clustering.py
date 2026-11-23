"""Sampling and clustering ops, compared with numpy ports of PyG's
implementations (pyg-lib / torch-cluster, not installable here), and with
torch-cluster itself when it is importable."""

import mlx.core as mx
import numpy as np
import pytest
import torch

from mlx_graphs.nn import avg_pool, avg_pool_x, max_pool, max_pool_x
from mlx_graphs.utils import consecutive_cluster, fps, graclus, nearest, voxel_grid

try:
    import torch_cluster
except ImportError:  # pragma: no cover - optional reference
    torch_cluster = None


@pytest.fixture(params=["gpu", "cpu"])
def device(request):
    """Runs the test on the GPU (Metal kernels) and on the CPU."""
    previous = mx.default_device()
    if request.param == "gpu" and not mx.metal.is_available():
        pytest.skip("no Metal device")
    mx.set_default_device(mx.gpu if request.param == "gpu" else mx.cpu)
    yield request.param
    mx.set_default_device(previous)


# ------------------------------------------------------------------------ fps


def fps_reference(x, batch, ratio, start=None):
    """Port of torch-cluster's / pyg-lib's CPU farthest point sampling."""
    sizes = np.bincount(batch)
    ptr = np.concatenate([[0], np.cumsum(sizes)])
    counts = np.ceil(sizes.astype(np.float32) * np.float32(ratio)).astype(np.int64)
    out = []
    for b in range(len(sizes)):
        y = x[ptr[b] : ptr[b + 1]]
        idx = 0 if start is None else start[b]
        selected = [idx]
        dist = ((y - y[idx]) ** 2).sum(1)
        for _ in range(1, counts[b]):
            idx = int(dist.argmax())
            selected.append(idx)
            dist = np.minimum(dist, ((y - y[idx]) ** 2).sum(1))
        out.append(ptr[b] + np.array(selected, dtype=np.int64))
    return np.concatenate(out)


@pytest.mark.parametrize(
    "num_graphs,max_size,dim,ratio",
    [(1, 10, 3, 0.5), (4, 1000, 3, 0.25), (3, 37, 2, 0.3), (2, 3000, 6, 0.5)],
)
def test_fps_matches_pyg(device, num_graphs, max_size, dim, ratio):
    rng = np.random.default_rng(max_size)
    sizes = rng.integers(max_size // 2, max_size + 1, num_graphs)
    batch = np.repeat(np.arange(num_graphs), sizes)
    x = rng.normal(size=(batch.size, dim)).astype(np.float32)

    out = fps(mx.array(x), mx.array(batch), ratio, random_start=False)
    expected = fps_reference(x, batch, ratio)
    np.testing.assert_array_equal(np.array(out), expected)
    assert out.dtype == mx.int32

    if torch_cluster is not None:
        ref = torch_cluster.fps(
            torch.tensor(x), torch.tensor(batch), ratio, random_start=False
        )
        np.testing.assert_array_equal(ref.numpy(), expected)


def test_fps_example_and_defaults(device):
    x = mx.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    assert fps(x, mx.array([0, 0, 0, 0]), 0.5, random_start=False).tolist() == [0, 3]
    assert fps(x, None, 0.5, random_start=False).tolist() == [0, 3]
    assert fps(x, None, 1.0, random_start=False).tolist() == [0, 3, 1, 2]


def test_fps_random_start(device):
    rng = np.random.default_rng(0)
    batch = np.repeat([0, 1, 2], [50, 30, 40])
    x = rng.normal(size=(120, 3)).astype(np.float32)
    mx.random.seed(3)
    out = np.array(fps(mx.array(x), mx.array(batch), 0.2))
    starts = out[[0, 10, 16]] - np.array([0, 50, 80])
    # the rest is farthest point sampling from the random starting points
    np.testing.assert_array_equal(out, fps_reference(x, batch, 0.2, starts))
    seen = {
        tuple(np.array(fps(mx.array(x), mx.array(batch), 0.2))[[0, 10, 16]])
        for _ in range(5)
    }
    assert len(seen) > 1


def test_fps_errors():
    with pytest.raises(ValueError, match="ratio"):
        fps(mx.zeros((4, 3)), ratio=0.0)
    with pytest.raises(ValueError, match="sorted"):
        fps(mx.zeros((4, 3)), mx.array([1, 1, 0, 0]))


def test_fps_with_empty_graph(device):
    x = mx.random.normal((6, 3))
    out = fps(x, mx.array([0, 0, 0, 2, 2, 2]), 0.5, random_start=False, batch_size=3)
    assert out.tolist()[0] == 0 and out.tolist()[2] == 3 and out.shape == (4,)
    # an empty last graph
    out = fps(x, mx.zeros((6,), dtype=mx.int32), 0.5, random_start=False, batch_size=2)
    np.testing.assert_array_equal(
        np.array(out), fps_reference(np.array(x), np.zeros(6, np.int64), 0.5)
    )


# ----------------------------------------------------------------- voxel_grid


def grid_cluster_reference(pos, size, start=None, end=None):
    """Port of torch-cluster's ``grid_cluster`` (float32, truncation)."""
    pos = pos.astype(np.float32)
    size = np.asarray(size, dtype=np.float32)
    start = pos.min(0) if start is None else np.asarray(start, dtype=np.float32)
    end = pos.max(0) if end is None else np.asarray(end, dtype=np.float32)
    pos = pos - start[None]
    num_voxels = ((end - start) / size).astype(np.int64) + 1
    num_voxels = np.concatenate([[1], np.cumprod(num_voxels)])[: size.size]
    return ((pos / size[None]).astype(np.int64) * num_voxels[None]).sum(1)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(size=0.7),
        dict(size=[0.5, 0.3, 1.0]),
        dict(size=0.5, start=-2.0, end=3.0),
        dict(size=[0.5, 0.3, 1.0], start=[-3.0, -2.5, -2.0], end=[3.0, 3.0, 4.0]),
    ],
)
@pytest.mark.parametrize("with_batch", [True, False])
def test_voxel_grid_matches_pyg(kwargs, with_batch):
    rng = np.random.default_rng(0)
    pos = rng.uniform(-2, 3, size=(500, 3)).astype(np.float32)
    batch = np.repeat(np.arange(5), 100) if with_batch else np.zeros(500, np.int64)

    # PyG's voxel_grid adds the batch as a dimension of voxel size 1
    size = np.broadcast_to(np.asarray(kwargs["size"], np.float32), 3)
    start = kwargs.get("start")
    end = kwargs.get("end")
    expected = grid_cluster_reference(
        np.concatenate([pos, batch[:, None]], 1),
        np.concatenate([size, [1.0]]),
        None if start is None else np.append(np.broadcast_to(start, 3), 0.0),
        None if end is None else np.append(np.broadcast_to(end, 3), batch.max()),
    )
    out = voxel_grid(
        mx.array(pos), batch=mx.array(batch) if with_batch else None, **kwargs
    )
    assert out.dtype == mx.int64
    np.testing.assert_array_equal(np.array(out), expected)

    if torch_cluster is not None:
        ref = torch_cluster.grid_cluster(
            torch.tensor(np.concatenate([pos, batch[:, None]], 1).astype(np.float32)),
            torch.tensor(np.concatenate([size, [1.0]])),
            None
            if start is None
            else torch.tensor(
                np.append(np.broadcast_to(start, 3), 0.0), dtype=torch.float
            ),
            None
            if end is None
            else torch.tensor(
                np.append(np.broadcast_to(end, 3), batch.max()), dtype=torch.float
            ),
        )
        np.testing.assert_array_equal(ref.numpy(), expected)


def test_voxel_grid_example():
    pos = mx.array([[0.0, 0.0], [0.4, 0.2], [1.5, 0.1], [0.2, 1.7]])
    assert voxel_grid(pos, size=1.0).tolist() == [0, 0, 1, 2]
    assert voxel_grid(mx.array([0.1, 0.2, 2.5]), size=1.0).tolist() == [0, 0, 2]


# -------------------------------------------------------------------- nearest


@pytest.mark.parametrize("with_batch", [True, False])
def test_nearest_matches_pyg(with_batch):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(300, 3)).astype(np.float32)
    y = rng.normal(size=(40, 3)).astype(np.float32)
    bx = np.repeat(np.arange(4), 75) if with_batch else np.zeros(300, np.int64)
    by = np.repeat(np.arange(4), 10) if with_batch else np.zeros(40, np.int64)
    d = ((x[:, None] - y[None]) ** 2).sum(-1)
    d[bx[:, None] != by[None]] = np.inf
    expected = d.argmin(1)
    args = (mx.array(bx), mx.array(by)) if with_batch else ()
    out = nearest(mx.array(x), mx.array(y), *args)
    np.testing.assert_array_equal(np.array(out), expected)

    if torch_cluster is not None:
        ref = torch_cluster.nearest(
            torch.tensor(x), torch.tensor(y), torch.tensor(bx), torch.tensor(by)
        )
        np.testing.assert_array_equal(ref.numpy(), expected)


def test_nearest_example_and_errors():
    x = mx.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    y = mx.array([[-1.0, 0.0], [1.0, 0.0]])
    assert nearest(x, y).tolist() == [0, 0, 1, 1]
    with pytest.raises(ValueError, match="same graph"):
        nearest(x, y, mx.array([0, 0, 1, 1]), mx.array([0, 0]))


# -------------------------------------------------------------------- graclus


def graclus_reference(edge_index, num_nodes, weight=None, perm=None):
    """Port of torch-cluster's / pyg-lib's CPU graclus, for a node order."""
    order = np.argsort(edge_index[0], kind="stable")
    row, col = edge_index[0][order], edge_index[1][order]
    w = None if weight is None else weight[order]
    rowptr = np.concatenate([[0], np.cumsum(np.bincount(row, minlength=num_nodes))])
    out = np.full(num_nodes, -1)
    for u in perm if perm is not None else range(num_nodes):
        if out[u] >= 0:
            continue
        if w is None:
            out[u] = u
            for e in range(rowptr[u], rowptr[u + 1]):
                v = col[e]
                if out[v] >= 0:
                    continue
                out[u] = out[v] = min(u, v)
                break
        else:
            v_max, w_max = u, 0.0
            for e in range(rowptr[u], rowptr[u + 1]):
                v = col[e]
                if out[v] >= 0:
                    continue
                if w[e] >= w_max:
                    v_max, w_max = v, w[e]
            out[u] = out[v_max] = min(u, v_max)
    return out


def check_matching(cluster, edge_index, num_nodes):
    """Invariants of PyG's graclus: clusters of 1 or 2 nodes named after their
    smallest node, pairs linked by an edge, and a maximal matching."""
    cluster = np.array(cluster)
    assert cluster.shape == (num_nodes,)
    sizes = np.bincount(cluster, minlength=num_nodes)
    assert set(np.unique(sizes[cluster])) <= {1, 2}
    assert (cluster <= np.arange(num_nodes)).all()
    assert (cluster[cluster] == cluster).all()  # the id is a member
    edges = set(map(tuple, edge_index.T.tolist()))
    for v in np.nonzero(cluster != np.arange(num_nodes))[0]:
        assert (cluster[v], v) in edges
    single = sizes[cluster] == 1
    row, col = edge_index
    assert not np.any(single[row] & single[col] & (row != col))


def random_undirected(rng, num_nodes, num_edges):
    e = rng.integers(0, num_nodes, (2, num_edges))
    e = np.unique(np.concatenate([e, e[::-1]], 1), axis=1)
    return e[:, e[0] != e[1]]


@pytest.mark.parametrize("weighted", [False, True])
def test_graclus_invariants(device, weighted):
    rng = np.random.default_rng(2)
    edges = random_undirected(rng, 200, 400)
    weight = None
    if weighted:
        w = rng.uniform(size=edges.shape[1]).astype(np.float32)
        # symmetric weights
        key = np.minimum(edges[0], edges[1]) * 1000 + np.maximum(edges[0], edges[1])
        weight = mx.array(w[np.unique(key, return_inverse=True)[1]])
    mx.random.seed(0)
    cluster = graclus(mx.array(edges), weight, num_nodes=210)  # 10 isolated
    assert cluster.dtype == mx.int64
    check_matching(cluster, edges, 210)
    assert (np.array(cluster)[200:] == np.arange(200, 210)).all()
    # the PyG reference satisfies the same invariants
    ref = graclus_reference(edges, 210, None if weight is None else np.array(weight))
    check_matching(ref, edges, 210)
    if torch_cluster is not None:
        out = torch_cluster.graclus_cluster(
            torch.tensor(edges[0]),
            torch.tensor(edges[1]),
            None if weight is None else torch.tensor(np.array(weight)),
            num_nodes=210,
        )
        check_matching(out.numpy(), edges, 210)


def test_graclus_matches_pyg_when_unique():
    """Graphs whose greedy matching does not depend on the node order."""
    # path with weights 1, 0.5, 2: every order matches (0, 1) and (2, 3)
    edges = np.array([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    weight = np.array([1.0, 1.0, 0.5, 0.5, 2.0, 2.0], dtype=np.float32)
    out = np.array(graclus(mx.array(edges), mx.array(weight)))
    for perm in [[0, 1, 2, 3], [3, 2, 1, 0], [1, 2, 0, 3], [2, 1, 3, 0]]:
        np.testing.assert_array_equal(out, graclus_reference(edges, 4, weight, perm))

    # disjoint edges, a triangle-free perfect matching, and isolated nodes
    rng = np.random.default_rng(3)
    perm = rng.permutation(20)
    pairs = perm.reshape(2, 10)
    edges = np.concatenate([pairs, pairs[::-1]], 1)
    out = np.array(graclus(mx.array(edges), num_nodes=23))
    expected = graclus_reference(edges, 23, perm=rng.permutation(23))
    np.testing.assert_array_equal(out, expected)


def test_graclus_negative_weights_and_self_loops():
    edges = np.array([[0, 1, 2, 3, 2], [1, 0, 3, 2, 2]])
    weight = mx.array([-1.0, -1.0, 1.0, 1.0, 5.0])
    assert graclus(mx.array(edges), weight).tolist() == [0, 1, 2, 2]


# -------------------------------------------------------------------- pooling


def test_consecutive_cluster():
    cluster, perm = consecutive_cluster(mx.array([5, 2, 5, 9, 2]))
    assert cluster.tolist() == [1, 0, 1, 2, 0]
    assert perm.tolist() == [4, 2, 3]


@pytest.mark.parametrize("fn,reduce", [(avg_pool_x, "mean"), (max_pool_x, "max")])
def test_pool_x_matches_pyg(fn, reduce):
    from torch_geometric.nn import avg_pool_x as pyg_avg
    from torch_geometric.nn import max_pool_x as pyg_max

    pyg_fn = pyg_avg if reduce == "mean" else pyg_max
    rng = np.random.default_rng(4)
    x = rng.normal(size=(50, 4)).astype(np.float32)
    batch = np.repeat([0, 1], 25)
    cluster = rng.integers(0, 6, 50) + 10 * batch
    out, out_batch = fn(mx.array(cluster), mx.array(x), mx.array(batch))
    ref, ref_batch = pyg_fn(torch.tensor(cluster), torch.tensor(x), torch.tensor(batch))
    np.testing.assert_allclose(np.array(out), ref.numpy(), rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(np.array(out_batch), ref_batch.numpy())

    out, none = fn(mx.array(cluster), mx.array(x), mx.array(batch), size=10)
    ref, _ = pyg_fn(
        torch.tensor(cluster), torch.tensor(x), torch.tensor(batch), size=10
    )
    assert none is None
    np.testing.assert_allclose(np.array(out), ref.numpy(), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("fn,name", [(avg_pool, "avg_pool"), (max_pool, "max_pool")])
def test_pool_matches_pyg(fn, name):
    import torch_geometric.nn as pyg_nn
    from torch_geometric.data import Batch, Data

    from mlx_graphs.data import GraphData, GraphDataBatch

    rng = np.random.default_rng(5)
    graphs, pyg_graphs = [], []
    for n in (20, 15):
        pos = rng.uniform(size=(n, 2)).astype(np.float32)
        x = rng.normal(size=(n, 3)).astype(np.float32)
        e = random_undirected(rng, n, 30)
        attr = rng.normal(size=(e.shape[1], 2)).astype(np.float32)
        graphs.append(
            GraphData(mx.array(e), mx.array(x), mx.array(attr), pos=mx.array(pos))
        )
        pyg_graphs.append(
            Data(
                torch.tensor(x),
                torch.tensor(e),
                torch.tensor(attr),
                pos=torch.tensor(pos),
            )
        )
    batch, pyg_batch = GraphDataBatch(graphs), Batch.from_data_list(pyg_graphs)
    b = np.array(batch.batch_indices)
    cluster = np.array(voxel_grid(batch.pos, 0.3, batch.batch_indices))

    out = fn(mx.array(cluster), batch)
    ref = getattr(pyg_nn, name)(torch.tensor(cluster), pyg_batch)
    np.testing.assert_allclose(
        np.array(out.node_features), ref.x.numpy(), rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(np.array(out.pos), ref.pos.numpy(), rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(np.array(out.edge_index), ref.edge_index.numpy())
    np.testing.assert_allclose(
        np.array(out.edge_features), ref.edge_attr.numpy(), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_array_equal(np.array(out.batch_indices), ref.batch.numpy())
    assert out.num_nodes == ref.num_nodes
    assert b.shape == (35,)
