import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from mlx_graphs.data import HeteroGraphData
from mlx_graphs.loaders import NeighborLoader
from mlx_graphs.nn import HeteroConv, HeteroDictLinear, SAGEConv

WRITES = ("author", "writes", "paper")
REV_WRITES = ("paper", "rev_writes", "author")


def test_output_shapes():
    proj = HeteroDictLinear({"author": 3, "paper": 100}, 8)
    out = proj({"author": mx.ones((4, 3)), "paper": mx.ones((5, 100))})
    assert {k: v.shape for k, v in out.items()} == {"author": (4, 8), "paper": (5, 8)}


def test_matches_one_linear_per_node_type():
    proj = HeteroDictLinear({"author": 3, "paper": 7}, 4)
    x_dict = {"author": mx.random.normal((4, 3)), "paper": mx.random.normal((5, 7))}
    out = proj(x_dict)
    for node_type, x in x_dict.items():
        lin = proj.lins[node_type]
        assert mx.allclose(out[node_type], x @ lin.weight.T + lin.bias, atol=1e-5)


def test_missing_node_types_are_skipped():
    proj = HeteroDictLinear({"author": 3, "paper": 7}, 4)
    # paper missing from x_dict, venue has no layer
    out = proj({"author": mx.ones((2, 3)), "venue": mx.ones((2, 9))})
    assert set(out) == {"author"}


def test_no_bias():
    proj = HeteroDictLinear({"author": 3}, 4, bias=False)
    assert "bias" not in proj.lins["author"]
    assert mx.array_equal(
        proj({"author": mx.zeros((2, 3))})["author"], mx.zeros((2, 4))
    )


def test_accepts_num_node_features():
    data = HeteroGraphData(
        edge_index_dict={WRITES: mx.array([[0], [0]])},
        node_features_dict={"author": mx.ones((2, 3)), "paper": mx.ones((1, 5))},
    )
    proj = HeteroDictLinear(data.num_node_features, 8)
    out = proj(data.node_features_dict)
    assert out["author"].shape == (2, 8) and out["paper"].shape == (1, 8)


def test_empty_input_dims():
    with pytest.raises(ValueError, match="at least one"):
        HeteroDictLinear({}, 8)


def test_gradients_reach_every_node_type():
    proj = HeteroDictLinear({"author": 3, "paper": 7}, 4)
    x_dict = {"author": mx.random.normal((4, 3)), "paper": mx.random.normal((5, 7))}

    def loss_fn(model):
        return sum((v**2).mean() for v in model(x_dict).values())

    _, grads = nn.value_and_grad(proj, loss_fn)(proj)
    for node_type in ("author", "paper"):
        assert mx.abs(grads["lins"][node_type]["weight"]).sum().item() > 0


def _num_weights(module):
    """Number of entries in weight matrices (biases excluded)."""
    return sum(
        v.size for k, v in tree_flatten(module.parameters()) if k.endswith("weight")
    )


def test_projecting_first_uses_fewer_weights():
    """Authors have 3 features, papers 100, everything becomes 8.

    No projection: 3*8 + 100*8 (author->paper) + 100*8 + 3*8 (paper->author)
                   = 1648 weights
    Project first: 3*8 + 100*8 (projection) + 2 * (8*8 + 8*8) (convs)
                   = 1080 weights
    """
    no_projection = HeteroConv(
        {WRITES: SAGEConv((3, 100), 8), REV_WRITES: SAGEConv((100, 3), 8)}
    )
    projection = HeteroDictLinear({"author": 3, "paper": 100}, 8)
    convs = HeteroConv({WRITES: SAGEConv(8, 8), REV_WRITES: SAGEConv(8, 8)})

    assert _num_weights(no_projection) == 1648
    assert _num_weights(projection) + _num_weights(convs) == 1080


def test_project_first_model_on_loader_batch():
    writes = mx.array([[0, 0, 1, 2, 3, 3], [0, 1, 1, 2, 3, 4]], dtype=mx.int64)
    data = HeteroGraphData(
        edge_index_dict={WRITES: writes, REV_WRITES: writes[::-1]},
        node_features_dict={
            "author": mx.random.normal((4, 3)),
            "paper": mx.random.normal((5, 100)),
        },
        node_labels_dict={"author": mx.array([0, 1, 0, 1])},
    )

    class Model(nn.Module):
        def __init__(self, hidden=8):
            super().__init__()
            self.proj = HeteroDictLinear(data.num_node_features, hidden)
            self.conv = HeteroConv(
                {WRITES: SAGEConv(hidden, hidden), REV_WRITES: SAGEConv(hidden, hidden)}
            )
            self.head = nn.Linear(hidden, 2)

        def __call__(self, x_dict, edge_index_dict):
            h = {k: nn.relu(v) for k, v in self.proj(x_dict).items()}
            h = self.conv(h, edge_index_dict)
            return self.head(h["author"])

    model = Model()
    loader = NeighborLoader(data, [-1], input_nodes="author", batch_size=2)

    def loss_fn(model, batch):
        n = batch.batch_size
        logits = model(batch.node_features_dict, batch.edge_index_dict)
        return nn.losses.cross_entropy(
            logits[:n], batch.node_labels_dict["author"][:n], reduction="mean"
        )

    for batch in loader:
        loss, grads = nn.value_and_grad(model, loss_fn)(model, batch)
        assert not mx.isnan(loss).item()
        assert mx.abs(grads["proj"]["lins"]["paper"]["weight"]).sum().item() > 0
