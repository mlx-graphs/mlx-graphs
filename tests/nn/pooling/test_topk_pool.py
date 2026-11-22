import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch
from mlx.utils import tree_flatten
from torch_geometric.nn import SAGPooling as PyGSAGPooling
from torch_geometric.nn import TopKPooling as PyGTopKPooling

from mlx_graphs.nn import GCNConv, SAGPooling, TopKPooling, filter_adj, topk

# The SAGPooling scores use a matmul, which is only accurate to ~1e-3 relative
# on the GPU of some machines.
TOL = dict(rtol=2e-3, atol=2e-3)

DIM = 4
# 3 graphs with 6, 3 and 5 nodes
BATCH = np.repeat(np.arange(3), [6, 3, 5]).astype(np.int64)
NUM_NODES = len(BATCH)


def _random_batch(rng, num_edges_per_graph=8):
    """Random edges inside each graph of BATCH, with edge features."""
    edges = []
    for g in range(BATCH.max() + 1):
        nodes = np.nonzero(BATCH == g)[0]
        edges.append(rng.choice(nodes, size=(2, num_edges_per_graph)))
    edge_index = np.concatenate(edges, axis=1).astype(np.int64)
    x = rng.normal(size=(NUM_NODES, DIM)).astype(np.float32)
    edge_attr = rng.normal(size=(edge_index.shape[1], 3)).astype(np.float32)
    return x, edge_index, edge_attr


def _assert_same_output(out, expected):
    names = ["x", "edge_index", "edge_attr", "batch", "perm", "score"]
    for name, a, b in zip(names, out, expected):
        if b is None:
            assert a is None, name
            continue
        b = b.detach().numpy()
        assert a.shape == b.shape, name
        if name in ("x", "edge_attr", "score"):
            np.testing.assert_allclose(np.array(a), b, err_msg=name, **TOL)
        else:
            np.testing.assert_array_equal(np.array(a), b, err_msg=name)


POOL_KWARGS = [
    dict(ratio=0.5),
    dict(ratio=0.3, multiplier=2.0),
    dict(ratio=2),
    dict(ratio=10),
    dict(min_score=0.2),
    dict(ratio=0.5, nonlinearity="sigmoid"),
]


@pytest.mark.parametrize("kwargs", POOL_KWARGS)
def test_topk_pooling_matches_pyg(kwargs):
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    x, edge_index, edge_attr = _random_batch(rng)

    pyg_pool = PyGTopKPooling(DIM, **kwargs)
    expected = pyg_pool(
        torch.from_numpy(x),
        torch.from_numpy(edge_index),
        torch.from_numpy(edge_attr),
        torch.from_numpy(BATCH),
    )

    pool = TopKPooling(DIM, **kwargs)
    pool.select.weight = mx.array(pyg_pool.select.weight.detach().numpy())
    out = pool(
        mx.array(edge_index),
        mx.array(x),
        mx.array(edge_attr),
        mx.array(BATCH),
    )
    _assert_same_output(out, expected)


def test_topk_pooling_single_graph_and_attn():
    rng = np.random.default_rng(1)
    torch.manual_seed(1)
    x, edge_index, _ = _random_batch(rng)
    attn = rng.normal(size=(NUM_NODES, DIM)).astype(np.float32)

    pyg_pool = PyGTopKPooling(DIM, ratio=0.4)
    expected = pyg_pool(
        torch.from_numpy(x), torch.from_numpy(edge_index), attn=torch.from_numpy(attn)
    )

    pool = TopKPooling(DIM, ratio=0.4)
    pool.select.weight = mx.array(pyg_pool.select.weight.detach().numpy())
    out = pool(mx.array(edge_index), mx.array(x), attn=mx.array(attn))
    _assert_same_output(out, expected)


@pytest.mark.parametrize("kwargs", POOL_KWARGS[:3] + POOL_KWARGS[4:5])
def test_sag_pooling_matches_pyg(kwargs):
    rng = np.random.default_rng(2)
    torch.manual_seed(2)
    x, edge_index, edge_attr = _random_batch(rng)

    pyg_pool = PyGSAGPooling(DIM, **kwargs)
    expected = pyg_pool(
        torch.from_numpy(x),
        torch.from_numpy(edge_index),
        torch.from_numpy(edge_attr),
        torch.from_numpy(BATCH),
    )

    pool = SAGPooling(DIM, **kwargs)
    gnn = pyg_pool.gnn
    pool.gnn.neigh_proj.weight = mx.array(gnn.lin_rel.weight.detach().numpy())
    pool.gnn.neigh_proj.bias = mx.array(gnn.lin_rel.bias.detach().numpy())
    pool.gnn.self_proj.weight = mx.array(gnn.lin_root.weight.detach().numpy())
    pool.select.weight = mx.array(pyg_pool.select.weight.detach().numpy())

    out = pool(
        mx.array(edge_index),
        mx.array(x),
        mx.array(edge_attr),
        mx.array(BATCH),
    )
    _assert_same_output(out, expected)


def test_sag_pooling_other_gnn():
    pool = SAGPooling(DIM, ratio=0.5, GNN=GCNConv)
    assert isinstance(pool.gnn, GCNConv)
    rng = np.random.default_rng(3)
    x, edge_index, _ = _random_batch(rng)
    out_x, out_edges, _, out_batch, perm, score = pool(
        mx.array(edge_index), mx.array(x), batch_indices=mx.array(BATCH)
    )
    assert out_x.shape == (3 + 2 + 3, DIM)
    assert perm.shape == score.shape == out_batch.shape == (8,)
    assert repr(pool) == "SAGPooling(GCNConv, 4, ratio=0.5, multiplier=1.0)"


@pytest.mark.parametrize("pool_cls", [TopKPooling, SAGPooling])
def test_pooling_gradients(pool_cls):
    rng = np.random.default_rng(4)
    x, edge_index, _ = _random_batch(rng)
    pool = pool_cls(DIM, ratio=0.5)

    def loss_fn():
        out = pool(mx.array(edge_index), mx.array(x), batch_indices=mx.array(BATCH))
        return (out[0] ** 2).sum()

    _, grads = nn.value_and_grad(pool, loss_fn)()
    params = dict(tree_flatten(pool.trainable_parameters()))
    grads = dict(tree_flatten(grads))
    assert set(grads) == set(params)
    assert "select.weight" in grads
    for name, grad in grads.items():
        assert mx.all(mx.isfinite(grad)).item(), name
        assert mx.any(grad != 0).item(), name


def test_topk_function():
    scores = mx.array([0.1, 0.9, 0.5, 0.3, 0.7])
    batch = mx.array([0, 0, 0, 1, 1])
    assert topk(scores, 0.5, batch).tolist() == [1, 2, 4]
    assert topk(scores, 1, batch).tolist() == [1, 4]
    assert topk(scores, 5, batch).tolist() == [1, 2, 0, 4, 3]
    # with min_score, the best node of each graph is always kept
    assert topk(scores, None, batch, min_score=0.6).tolist() == [1, 4]
    assert topk(scores, None, batch, min_score=0.95).tolist() == [1, 4]
    with pytest.raises(ValueError):
        topk(scores, None, batch)


def test_filter_adj():
    edge_index = mx.array([[0, 1, 2, 3], [1, 2, 0, 0]])
    edge_features = mx.array([[1.0], [2.0], [3.0], [4.0]])
    new_edges, new_features = filter_adj(edge_index, edge_features, mx.array([2, 0]))
    assert new_edges.tolist() == [[0], [1]]
    assert new_features.tolist() == [[3.0]]


@pytest.mark.parametrize("pool_cls", [TopKPooling, SAGPooling])
@pytest.mark.parametrize("kwargs", [dict(), dict(min_score=0.1)])
def test_pooling_without_nodes(pool_cls, kwargs):
    # min_score read the number of graphs with max() of an empty array, and
    # SAGPooling's GNN failed on 0 nodes.
    pool = pool_cls(4, **kwargs)
    out = pool(
        mx.zeros((2, 0), dtype=mx.int32),
        mx.zeros((0, 4)),
        batch_indices=mx.zeros((0,), dtype=mx.int32),
    )
    x, edge_index, _, batch, perm, score = out
    mx.eval(x, edge_index, batch, perm, score)
    assert x.shape == (0, 4) and edge_index.shape == (2, 0)
    assert batch.shape == perm.shape == score.shape == (0,)
