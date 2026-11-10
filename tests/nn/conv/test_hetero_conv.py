import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_graphs.data import HeteroGraphData
from mlx_graphs.loaders import NeighborLoader
from mlx_graphs.nn import HeteroConv, SAGEConv

WRITES = ("author", "writes", "paper")
REV_WRITES = ("paper", "rev_writes", "author")
CITES = ("paper", "cites", "paper")


def _inputs():
    x_dict = {"author": mx.random.normal((4, 6)), "paper": mx.random.normal((5, 3))}
    edge_index_dict = {
        WRITES: mx.array([[0, 1, 3, 3], [0, 0, 3, 4]]),
        CITES: mx.array([[1, 2, 4], [0, 0, 2]]),
        REV_WRITES: mx.array([[0, 0, 3, 4], [0, 1, 3, 3]]),
    }
    return x_dict, edge_index_dict


def _convs(out_dim=8):
    return {
        WRITES: SAGEConv((6, 3), out_dim),
        CITES: SAGEConv(3, out_dim),
        REV_WRITES: SAGEConv((3, 6), out_dim),
    }


def _set_scalar_sage(conv, w_self, w_neigh):
    conv.self_proj.weight = mx.array([[w_self]])
    conv.self_proj.bias = mx.array([0.0])
    conv.neigh_proj.weight = mx.array([[w_neigh]])


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------


def test_output_types_and_shapes():
    x_dict, edge_index_dict = _inputs()
    conv = HeteroConv(_convs(8))
    out = conv(x_dict, edge_index_dict)
    assert set(out) == {"author", "paper"}
    assert out["author"].shape == (4, 8)
    assert out["paper"].shape == (5, 8)


def test_node_type_without_incoming_edges_is_not_in_output():
    x_dict, edge_index_dict = _inputs()
    convs = _convs()
    del convs[REV_WRITES]  # nothing points to authors anymore
    out = HeteroConv(convs)(x_dict, edge_index_dict)
    assert set(out) == {"paper"}


def test_skips_missing_edge_types_and_features():
    x_dict, edge_index_dict = _inputs()
    conv = HeteroConv(_convs())

    # edge type missing from edge_index_dict
    partial = {k: v for k, v in edge_index_dict.items() if k != REV_WRITES}
    assert set(conv(x_dict, partial)) == {"paper"}

    # source features missing: author -> paper is skipped, cites still runs
    out = conv({"paper": x_dict["paper"]}, edge_index_dict)
    expected = conv.convs["paper__cites__paper"](
        edge_index_dict[CITES], x_dict["paper"]
    )
    assert set(out) == {"paper"}
    assert mx.allclose(out["paper"], expected)


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------


def test_worked_example():
    """Alice(1), Bob(3) write papers P0(10), P1(20); P1 cites P0.

    writes (w_self=0.5, w_neigh=2): P0 = 0.5*10 + 2*mean(1,3) = 9
                                    P1 = 0.5*20 + 2*3         = 16
    cites  (w_self=0.1, w_neigh=1): P0 = 0.1*10 + 1*20 = 21
                                    P1 = 0.1*20 + 0    = 2
    sum:                            P0 = 30, P1 = 18
    """
    writes_conv, cites_conv = SAGEConv((1, 1), 1), SAGEConv(1, 1)
    _set_scalar_sage(writes_conv, 0.5, 2.0)
    _set_scalar_sage(cites_conv, 0.1, 1.0)
    conv = HeteroConv({WRITES: writes_conv, CITES: cites_conv}, aggr="sum")

    x_dict = {"author": mx.array([[1.0], [3.0]]), "paper": mx.array([[10.0], [20.0]])}
    edge_index_dict = {
        WRITES: mx.array([[0, 1, 1], [0, 0, 1]]),
        CITES: mx.array([[1], [0]]),
    }
    out = conv(x_dict, edge_index_dict)
    assert mx.allclose(out["paper"], mx.array([[30.0], [18.0]]))


@pytest.mark.parametrize("aggr", ["sum", "mean", "max"])
def test_matches_running_each_layer_by_hand(aggr):
    x_dict, edge_index_dict = _inputs()
    conv = HeteroConv(_convs(), aggr=aggr)
    out = conv(x_dict, edge_index_dict)

    from_writes = conv.convs["author__writes__paper"](
        edge_index_dict[WRITES], (x_dict["author"], x_dict["paper"])
    )
    from_cites = conv.convs["paper__cites__paper"](
        edge_index_dict[CITES], x_dict["paper"]
    )
    stacked = mx.stack([from_writes, from_cites])
    expected = {"sum": stacked.sum(0), "mean": stacked.mean(0), "max": stacked.max(0)}
    assert mx.allclose(out["paper"], expected[aggr], atol=1e-5)

    # authors only receive from one edge type: no combining
    from_rev = conv.convs["paper__rev_writes__author"](
        edge_index_dict[REV_WRITES], (x_dict["paper"], x_dict["author"])
    )
    assert mx.allclose(out["author"], from_rev, atol=1e-5)


def test_per_edge_type_kwargs():
    x_dict, edge_index_dict = _inputs()
    conv = HeteroConv(_convs())
    weights = mx.array([2.0, 0.5, 1.0])

    out = conv(x_dict, edge_index_dict, edge_weights_dict={CITES: weights})

    from_writes = conv.convs["author__writes__paper"](
        edge_index_dict[WRITES], (x_dict["author"], x_dict["paper"])
    )
    from_cites = conv.convs["paper__cites__paper"](
        edge_index_dict[CITES], x_dict["paper"], edge_weights=weights
    )
    assert mx.allclose(out["paper"], from_writes + from_cites, atol=1e-5)


def test_empty_edge_type_in_batch():
    x_dict, edge_index_dict = _inputs()
    edge_index_dict[CITES] = mx.zeros((2, 0), dtype=mx.int64)
    out = HeteroConv(_convs())(x_dict, edge_index_dict)
    assert out["paper"].shape == (5, 8)
    assert not mx.isnan(out["paper"]).any().item()


# ---------------------------------------------------------------------------
# Training: parameters are registered and receive gradients
# ---------------------------------------------------------------------------


def test_gradients_reach_every_layer():
    x_dict, edge_index_dict = _inputs()
    model = HeteroConv(_convs())

    def loss_fn(model):
        out = model(x_dict, edge_index_dict)
        return sum((v**2).mean() for v in out.values())

    loss, grads = nn.value_and_grad(model, loss_fn)(model)
    assert set(grads["convs"]) == {
        "author__writes__paper",
        "paper__cites__paper",
        "paper__rev_writes__author",
    }
    for layer_grads in grads["convs"].values():
        assert mx.abs(layer_grads["neigh_proj"]["weight"]).sum().item() > 0
        assert mx.abs(layer_grads["self_proj"]["weight"]).sum().item() > 0


def test_two_layer_model_on_loader_batch():
    writes = mx.array([[0, 0, 1, 2, 3, 3], [0, 1, 1, 2, 3, 4]], dtype=mx.int64)
    data = HeteroGraphData(
        edge_index_dict={
            WRITES: writes,
            REV_WRITES: writes[::-1],
            CITES: mx.array([[1, 2, 3, 4], [0, 0, 1, 2]], dtype=mx.int64),
        },
        node_features_dict={
            "author": mx.random.normal((4, 6)),
            "paper": mx.random.normal((5, 3)),
        },
        node_labels_dict={"author": mx.array([0, 1, 0, 1])},
    )

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = HeteroConv(_convs(16))
            self.conv2 = HeteroConv(
                {
                    WRITES: SAGEConv(16, 16),
                    CITES: SAGEConv(16, 16),
                    REV_WRITES: SAGEConv(16, 16),
                }
            )
            self.head = nn.Linear(16, 2)

        def __call__(self, x_dict, edge_index_dict):
            h = self.conv1(x_dict, edge_index_dict)
            h = {k: nn.relu(v) for k, v in h.items()}
            h = self.conv2(h, edge_index_dict)
            return self.head(h["author"])

    model = Model()
    loader = NeighborLoader(data, [-1, -1], input_nodes="author", batch_size=2)
    for batch in loader:
        logits = model(batch.node_features_dict, batch.edge_index_dict)
        n = batch.batch_size
        assert logits.shape == (batch.n_id_dict["author"].shape[0], 2)
        loss = nn.losses.cross_entropy(
            logits[:n], batch.node_labels_dict["author"][:n], reduction="mean"
        )
        assert not mx.isnan(loss).item()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_invalid_aggr():
    with pytest.raises(ValueError, match="Invalid aggregation"):
        HeteroConv(_convs(), aggr="min")


def test_empty_convs():
    with pytest.raises(ValueError, match="at least one"):
        HeteroConv({})


def test_bad_edge_type_key():
    with pytest.raises(ValueError, match="tuples"):
        HeteroConv({"author_writes_paper": SAGEConv(3, 3)})
    with pytest.raises(ValueError, match="__"):
        HeteroConv({("pa__per", "x", "paper"): SAGEConv(3, 3)})


def test_bad_kwarg_name():
    x_dict, edge_index_dict = _inputs()
    with pytest.raises(ValueError, match="_dict"):
        HeteroConv(_convs())(x_dict, edge_index_dict, edge_weights={CITES: None})
