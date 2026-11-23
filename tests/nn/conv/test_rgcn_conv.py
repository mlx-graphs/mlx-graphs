import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch
from mlx.utils import tree_flatten
from torch_geometric.nn import RGCNConv as PyGRGCNConv

from mlx_graphs.nn.conv.rgcn_conv import RGCNConv

# MLX's GPU matmul is only accurate to ~1e-3 relative on some machines, while a
# wrong formula gives O(1) errors with these O(1) inputs.
TOL = dict(rtol=2e-2, atol=2e-2)

NUM_RELATIONS = 5


def _random_graph(rng, num_src=12, num_dst=12, num_edges=60):
    """Random multi-relational graph with duplicate edges (same and different
    relation), isolated nodes (the last 3 nodes of each side never appear in an
    edge) and one relation without any edge (relation 2)."""
    src = rng.integers(0, num_src - 3, size=num_edges)
    dst = rng.integers(0, num_dst - 3, size=num_edges)
    edge_type = rng.choice([0, 1, 3, 4], size=num_edges)
    # duplicate the first 5 edges with the same relation and the next 5 with
    # another one
    src = np.concatenate([src, src[:10]])
    dst = np.concatenate([dst, dst[:10]])
    edge_type = np.concatenate([edge_type, edge_type[:5], (edge_type[5:10] + 1) % 2])
    return np.stack([src, dst]).astype(np.int64), edge_type.astype(np.int64)


def _copy_weights(conv, pyg_conv, rng):
    for name in ("weight", "comp", "root"):
        if name in conv:
            setattr(conv, name, mx.array(getattr(pyg_conv, name).detach().numpy()))
    if "bias" in conv:
        # PyG initializes the bias to zeros: use a random one in both layers
        bias = rng.normal(size=pyg_conv.bias.shape).astype(np.float32)
        with torch.no_grad():
            pyg_conv.bias.copy_(torch.from_numpy(bias))
        conv.bias = mx.array(bias)


def _force_path(monkeypatch, path):
    """Forces the dense or the per-relation code path of RGCNConv."""
    if path is not None:
        monkeypatch.setattr(
            RGCNConv, "_use_dense_path", lambda self, *args: path == "dense"
        )


def _assert_finite_grads(model, loss_fn):
    _, grads = nn.value_and_grad(model, loss_fn)()
    params = dict(tree_flatten(model.trainable_parameters()))
    grads = dict(tree_flatten(grads))
    assert set(grads) == set(params)
    for name, grad in grads.items():
        assert grad.shape == params[name].shape, name
        assert mx.all(mx.isfinite(grad)).item(), f"non-finite gradient for {name}"
        assert mx.any(grad != 0).item(), f"zero gradient for {name}"


@pytest.mark.parametrize("path", ["dense", "per_relation"])
@pytest.mark.parametrize(
    "kwargs",
    [
        dict(),
        dict(num_bases=2),
        dict(root_weight=False, bias=False),
        dict(aggr="add"),
        dict(aggr="max"),
    ],
)
def test_rgcn_conv_shapes(monkeypatch, path, kwargs):
    _force_path(monkeypatch, path)
    rng = np.random.default_rng(0)
    edge_index, edge_type = _random_graph(rng)
    node_features = mx.array(rng.normal(size=(12, 6)).astype(np.float32))

    conv = RGCNConv(6, 8, NUM_RELATIONS, **kwargs)
    out = conv(mx.array(edge_index), node_features, mx.array(edge_type))

    assert out.shape == (12, 8)
    assert mx.all(mx.isfinite(out)).item()


def test_rgcn_conv_parameters():
    conv = RGCNConv(6, 8, NUM_RELATIONS)
    assert conv.weight.shape == (NUM_RELATIONS, 6, 8)
    assert conv.root.shape == (6, 8)
    assert conv.bias.shape == (8,)
    assert "comp" not in conv
    assert conv.aggr == "mean"

    conv = RGCNConv(6, 8, NUM_RELATIONS, num_bases=2, root_weight=False, bias=False)
    assert conv.weight.shape == (2, 6, 8)
    assert conv.comp.shape == (NUM_RELATIONS, 2)
    assert "root" not in conv and "bias" not in conv

    with pytest.raises(ValueError):
        RGCNConv(6, 8, NUM_RELATIONS, aggr="softmax")
    with pytest.raises(ValueError):
        RGCNConv(6, 8, 0)


@pytest.mark.parametrize("path", [None, "dense", "per_relation"])  # None: auto
@pytest.mark.parametrize("aggr", ["mean", "add", "max", "min"])
@pytest.mark.parametrize("num_bases", [None, 2])
def test_rgcn_conv_matches_pyg(monkeypatch, path, aggr, num_bases):
    _force_path(monkeypatch, path)
    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    edge_index, edge_type = _random_graph(rng)
    x = rng.normal(size=(12, 6)).astype(np.float32)

    pyg_conv = PyGRGCNConv(6, 8, NUM_RELATIONS, num_bases=num_bases, aggr=aggr)
    conv = RGCNConv(6, 8, NUM_RELATIONS, num_bases=num_bases, aggr=aggr)
    _copy_weights(conv, pyg_conv, rng)

    with torch.no_grad():
        expected = pyg_conv(
            torch.from_numpy(x),
            torch.from_numpy(edge_index),
            torch.from_numpy(edge_type),
        ).numpy()

    out = conv(mx.array(edge_index), mx.array(x), mx.array(edge_type))

    assert out.shape == expected.shape
    np.testing.assert_allclose(np.array(out), expected, **TOL)
    # sanity check: the comparison is not trivially satisfied
    assert np.abs(expected).max() > 0.2


@pytest.mark.parametrize("path", ["dense", "per_relation"])
@pytest.mark.parametrize("aggr", ["mean", "add", "max"])
def test_rgcn_conv_shared_destinations_matches_pyg(monkeypatch, path, aggr):
    """Several relations send messages to the same node: the neighbors of each
    (node, relation) pair are normalized separately, also when they are next to
    each other once the edges are sorted by relation and destination."""
    _force_path(monkeypatch, path)
    torch.manual_seed(0)
    rng = np.random.default_rng(6)
    # relation 0: {0, 1} -> 5 and 2 -> 3, relation 1: 2 -> 5,
    # relation 3: {3, 4, 0, 0} -> 5 and 1 -> 6, relation 4: {1, 2} -> 6
    edge_index = np.array(
        [[3, 0, 2, 4, 1, 0, 2, 0, 1, 1, 2], [5, 5, 5, 5, 5, 5, 3, 5, 6, 6, 6]]
    )
    edge_type = np.array([3, 0, 1, 3, 0, 3, 0, 3, 3, 4, 4])
    x = rng.normal(size=(7, 6)).astype(np.float32)

    pyg_conv = PyGRGCNConv(6, 8, NUM_RELATIONS, aggr=aggr, root_weight=False)
    conv = RGCNConv(6, 8, NUM_RELATIONS, aggr=aggr, root_weight=False)
    _copy_weights(conv, pyg_conv, rng)

    with torch.no_grad():
        expected = pyg_conv(
            torch.from_numpy(x),
            torch.from_numpy(edge_index),
            torch.from_numpy(edge_type),
        ).numpy()

    out = conv(mx.array(edge_index), mx.array(x), mx.array(edge_type))
    np.testing.assert_allclose(np.array(out), expected, **TOL)

    if aggr == "mean":
        # node 5, by hand: mean of the neighbors of each relation
        weight = np.array(conv.weight)
        by_hand = (
            x[[0, 1]].mean(0) @ weight[0]
            + x[2] @ weight[1]
            + x[[3, 4, 0, 0]].mean(0) @ weight[3]
            + np.array(conv.bias)
        )
        np.testing.assert_allclose(np.array(out[5]), by_hand, **TOL)


@pytest.mark.parametrize("path", ["dense", "per_relation"])
@pytest.mark.parametrize("kwargs", [dict(root_weight=False), dict(bias=False)])
def test_rgcn_conv_no_root_or_bias_matches_pyg(monkeypatch, path, kwargs):
    _force_path(monkeypatch, path)
    torch.manual_seed(0)
    rng = np.random.default_rng(2)
    edge_index, edge_type = _random_graph(rng)
    x = rng.normal(size=(12, 6)).astype(np.float32)

    pyg_conv = PyGRGCNConv(6, 8, NUM_RELATIONS, **kwargs)
    conv = RGCNConv(6, 8, NUM_RELATIONS, **kwargs)
    _copy_weights(conv, pyg_conv, rng)

    with torch.no_grad():
        expected = pyg_conv(
            torch.from_numpy(x),
            torch.from_numpy(edge_index),
            torch.from_numpy(edge_type),
        ).numpy()

    out = conv(mx.array(edge_index), mx.array(x), mx.array(edge_type))
    np.testing.assert_allclose(np.array(out), expected, **TOL)

    if not kwargs.get("root_weight", True):
        # nodes 9, 10, 11 have no incoming edge: only the bias is left
        np.testing.assert_allclose(
            np.array(out[9:]), np.broadcast_to(np.array(conv.bias), (3, 8)), **TOL
        )


@pytest.mark.parametrize("path", ["dense", "per_relation"])
@pytest.mark.parametrize("num_bases", [None, 2])
def test_rgcn_conv_bipartite_matches_pyg(monkeypatch, path, num_bases):
    _force_path(monkeypatch, path)
    torch.manual_seed(0)
    rng = np.random.default_rng(3)
    edge_index, edge_type = _random_graph(rng, num_src=10, num_dst=7, num_edges=40)
    x_src = rng.normal(size=(10, 6)).astype(np.float32)
    x_dst = rng.normal(size=(7, 4)).astype(np.float32)

    pyg_conv = PyGRGCNConv((6, 4), 8, NUM_RELATIONS, num_bases=num_bases)
    conv = RGCNConv((6, 4), 8, NUM_RELATIONS, num_bases=num_bases)
    _copy_weights(conv, pyg_conv, rng)

    with torch.no_grad():
        expected = pyg_conv(
            (torch.from_numpy(x_src), torch.from_numpy(x_dst)),
            torch.from_numpy(edge_index),
            torch.from_numpy(edge_type),
        ).numpy()

    out = conv(
        mx.array(edge_index), (mx.array(x_src), mx.array(x_dst)), mx.array(edge_type)
    )

    assert out.shape == (7, 8)
    np.testing.assert_allclose(np.array(out), expected, **TOL)


@pytest.mark.parametrize("aggr", ["mean", "add", "max", "min"])
def test_rgcn_conv_paths_agree(monkeypatch, aggr):
    mx.random.seed(0)
    rng = np.random.default_rng(4)
    edge_index, edge_type = _random_graph(rng, num_src=50, num_dst=50, num_edges=400)
    edge_index, edge_type = mx.array(edge_index), mx.array(edge_type)
    node_features = mx.array(rng.normal(size=(50, 6)).astype(np.float32))

    conv = RGCNConv(6, 8, NUM_RELATIONS, aggr=aggr)

    _force_path(monkeypatch, "dense")
    dense = np.array(conv(edge_index, node_features, edge_type))
    _force_path(monkeypatch, "per_relation")
    per_relation = np.array(conv(edge_index, node_features, edge_type))

    np.testing.assert_allclose(dense, per_relation, **TOL)


def test_rgcn_conv_path_selection():
    conv = RGCNConv(16, 8, num_relations=10)

    # small graphs: always dense
    assert conv._use_dense_path(num_nodes=100, num_edges=50)
    # the dense [num_nodes * num_relations, dim] array is not larger than the
    # per-edge features: dense
    assert conv._use_dense_path(num_nodes=1_000_000, num_edges=10_000_000)
    # many more (node, relation) pairs than edges: one relation at a time
    assert not conv._use_dense_path(num_nodes=1_000_000, num_edges=2_000_000)


@pytest.mark.parametrize("path", ["dense", "per_relation"])
def test_rgcn_conv_no_edges(monkeypatch, path):
    _force_path(monkeypatch, path)
    mx.random.seed(0)
    node_features = mx.random.normal((4, 6))
    edge_index = mx.zeros((2, 0), dtype=mx.int32)
    edge_type = mx.zeros((0,), dtype=mx.int32)

    conv = RGCNConv(6, 8, NUM_RELATIONS)
    out = conv(edge_index, node_features, edge_type)

    expected = node_features @ conv.root + conv.bias
    np.testing.assert_allclose(np.array(out), np.array(expected), **TOL)


def test_rgcn_conv_edge_type_validation():
    conv = RGCNConv(6, 8, NUM_RELATIONS)
    edge_index = mx.array([[0, 1, 2], [1, 2, 0]])
    node_features = mx.random.normal((3, 6))

    with pytest.raises(ValueError):
        conv(edge_index, node_features, mx.array([0, 1]))
    with pytest.raises(ValueError):
        conv(edge_index, node_features, mx.array([[0], [1], [2]]))


@pytest.mark.parametrize("path", ["dense", "per_relation"])
@pytest.mark.parametrize(
    "kwargs",
    [dict(), dict(num_bases=2), dict(aggr="add"), dict(aggr="max")],
)
def test_rgcn_conv_gradients(monkeypatch, path, kwargs):
    _force_path(monkeypatch, path)
    mx.random.seed(0)
    rng = np.random.default_rng(5)
    edge_index, edge_type = _random_graph(rng)
    edge_index, edge_type = mx.array(edge_index), mx.array(edge_type)
    node_features = mx.array(rng.normal(size=(12, 6)).astype(np.float32))

    conv = RGCNConv(6, 8, NUM_RELATIONS, **kwargs)

    def loss_fn():
        return (conv(edge_index, node_features, edge_type) ** 2).sum()

    _assert_finite_grads(conv, loss_fn)


@pytest.mark.parametrize("aggr", [["mean", "max"], "powermean"])
def test_rgcn_conv_rejects_non_scatter_aggregations(aggr):
    with pytest.raises(ValueError, match="single scatter aggregation"):
        RGCNConv(4, 4, 2, aggr=aggr)
