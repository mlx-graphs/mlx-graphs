import mlx.core as mx
import numpy as np
import pytest

from mlx_graphs.data import GraphData
from mlx_graphs.data.batch import GraphDataBatch
from mlx_graphs.transforms import KNNGraph, RadiusGraph
from mlx_graphs.utils import knn_graph, radius_graph


def _graph(rng, num_nodes=8):
    pos = mx.array(rng.normal(size=(num_nodes, 2)).astype(np.float32))
    return GraphData(
        edge_index=mx.array([[0, 1], [1, 2]]),
        node_features=mx.ones((num_nodes, 1)),
        edge_features=mx.ones((2, 3)),
        edge_weights=mx.ones((2,)),
        pos=pos,
    )


@pytest.mark.parametrize("loop", [False, True])
def test_knn_graph_transform(loop):
    rng = np.random.default_rng(0)
    graph = _graph(rng)
    out = KNNGraph(k=3, loop=loop)(graph)

    expected = knn_graph(graph.pos, 3, loop=loop)
    assert out.edge_index.tolist() == expected.tolist()
    assert out.edge_features is None
    assert out.edge_weights is None
    assert out.num_nodes == 8
    # the input graph is not modified
    assert graph.edge_index.shape == (2, 2)
    assert repr(KNNGraph(k=3)) == "KNNGraph(k=3)"


def test_knn_graph_transform_undirected():
    rng = np.random.default_rng(1)
    graph = _graph(rng)
    out = KNNGraph(k=2, force_undirected=True)(graph)
    edges = set(map(tuple, np.array(out.edge_index).T.tolist()))
    directed = set(map(tuple, np.array(knn_graph(graph.pos, 2)).T.tolist()))
    assert edges == directed | {(j, i) for i, j in directed}
    assert len(edges) == out.edge_index.shape[1]


def test_radius_graph_transform():
    graph = GraphData(
        edge_index=mx.zeros((2, 0), dtype=mx.int64),
        pos=mx.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]]),
    )
    out = RadiusGraph(r=1.5)(graph)
    assert out.edge_index.tolist() == [[1, 0], [0, 1]]
    # the isolated node is kept
    assert out.num_nodes == 3

    out = RadiusGraph(r=1.5, loop=True, flow="target_to_source")(graph)
    assert out.edge_index.tolist() == [[0, 0, 1, 1, 2], [0, 1, 0, 1, 2]]
    assert repr(RadiusGraph(r=1.5)) == "RadiusGraph(r=1.5)"


def test_transforms_on_batch():
    rng = np.random.default_rng(2)
    batch = GraphDataBatch([_graph(rng, 5), _graph(rng, 4)])
    out = KNNGraph(k=2)(batch)
    expected = knn_graph(batch.pos, 2, batch.batch_indices)
    assert out.edge_index.tolist() == expected.tolist()
    # edges stay inside each graph
    src, dst = np.array(out.edge_index)
    graph_of = np.array(batch.batch_indices)
    assert (graph_of[src] == graph_of[dst]).all()

    out = RadiusGraph(r=1.0)(batch)
    expected = radius_graph(batch.pos, 1.0, batch.batch_indices)
    assert out.edge_index.tolist() == expected.tolist()


def test_missing_positions():
    with pytest.raises(ValueError):
        KNNGraph()(GraphData(edge_index=mx.array([[0], [1]])))


# PyG's KNNGraph / RadiusGraph call torch_cluster's knn_graph / radius_graph,
# which are not installed: they are replaced by brute-force versions with
# torch_cluster's documented semantics (k + 1 / max_num_neighbors + 1
# neighbors including the point itself, then self-loops removed), so that the
# transforms' own logic (batch vector, loops, flow, force_undirected, dropped
# edge attributes) is compared.


def _torch_cluster_pairs(pos, batch, num, loop, radius=None):
    pos = pos.numpy().astype(np.float64)
    batch = np.zeros(len(pos), np.int64) if batch is None else batch.numpy()
    rows, cols = [], []
    for i in range(len(pos)):
        candidates = np.nonzero(batch == batch[i])[0]
        dist = np.linalg.norm(pos[candidates] - pos[i], axis=-1)
        order = np.argsort(dist, kind="stable")
        if radius is not None:
            order = order[dist[order] < radius]
        nearest = candidates[order[: num if loop else num + 1]]
        if not loop:
            nearest = nearest[nearest != i][:num]
        rows += [i] * len(nearest)
        cols += nearest.tolist()
    return rows, cols


def _fake_knn_graph(x, k, batch=None, loop=False, flow="source_to_target", **kwargs):
    import torch

    row, col = _torch_cluster_pairs(x, batch, k, loop)
    row, col = (col, row) if flow == "source_to_target" else (row, col)
    return torch.tensor([row, col], dtype=torch.long)


def _fake_radius_graph(
    x, r, batch=None, loop=False, max_num_neighbors=32, flow="source_to_target", **kw
):
    import torch

    row, col = _torch_cluster_pairs(x, batch, max_num_neighbors, loop, radius=r)
    row, col = (col, row) if flow == "source_to_target" else (row, col)
    return torch.tensor([row, col], dtype=torch.long)


def _edge_list(edge_index):
    return sorted(map(tuple, np.asarray(edge_index).T.tolist()))


@pytest.mark.parametrize("loop", [False, True])
@pytest.mark.parametrize("force_undirected", [False, True])
@pytest.mark.parametrize("flow", ["source_to_target", "target_to_source"])
def test_transforms_match_pyg(monkeypatch, loop, force_undirected, flow):
    torch = pytest.importorskip("torch")
    pyg_nn = pytest.importorskip("torch_geometric.nn")
    pyg_transforms = pytest.importorskip("torch_geometric.transforms")
    from torch_geometric.data import Batch, Data

    monkeypatch.setattr(pyg_nn, "knn_graph", _fake_knn_graph)
    monkeypatch.setattr(pyg_nn, "radius_graph", _fake_radius_graph)

    rng = np.random.default_rng(6)
    graphs = [_graph(rng, n) for n in (12, 5, 9)]
    batch = GraphDataBatch(graphs)
    pyg_batch = Batch.from_data_list(
        [
            Data(
                pos=torch.from_numpy(np.asarray(g.pos)),
                edge_index=torch.zeros(2, 0, dtype=torch.long),
                edge_attr=torch.ones(0, 3),
            )
            for g in graphs
        ]
    )

    for k in (1, 3, 20):
        ours = KNNGraph(k, loop=loop, force_undirected=force_undirected, flow=flow)
        pyg = pyg_transforms.KNNGraph(
            k, loop=loop, force_undirected=force_undirected, flow=flow
        )
        out, expected = ours(batch), pyg(pyg_batch.clone())
        assert _edge_list(out.edge_index) == _edge_list(expected.edge_index)
        assert out.edge_features is None and expected.edge_attr is None

    for r, max_num in ((0.8, 32), (1.5, None), (2.0, 2)):
        ours = RadiusGraph(r, loop=loop, max_num_neighbors=max_num, flow=flow)
        pyg = pyg_transforms.RadiusGraph(
            r, loop=loop, max_num_neighbors=max_num or 10**6, flow=flow
        )
        out, expected = ours(batch), pyg(pyg_batch.clone())
        assert _edge_list(out.edge_index) == _edge_list(expected.edge_index)
