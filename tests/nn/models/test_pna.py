import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pytest

from mlx_graphs.nn import PNA, PNAConv

NUM_NODES, NUM_EDGES, DIM, EDGE_DIM = 30, 120, 12, 3
PNA_KWARGS = dict(
    aggregators=["mean", "min", "max", "std"],
    scalers=["identity", "amplification", "attenuation"],
)


@pytest.fixture
def graph():
    rng = np.random.default_rng(0)
    edge_index = mx.array(rng.integers(0, NUM_NODES, (2, NUM_EDGES)))
    x = mx.array(rng.standard_normal((NUM_NODES, DIM)).astype(np.float32))
    edge_features = mx.array(
        rng.standard_normal((NUM_EDGES, EDGE_DIM)).astype(np.float32)
    )
    deg = PNAConv.get_degree_histogram([(edge_index, NUM_NODES)])
    return edge_index, x, edge_features, deg


@pytest.mark.parametrize("num_layers", [1, 3])
@pytest.mark.parametrize("out_dim", [None, 4])  # divisible by towers
@pytest.mark.parametrize("jk", [None, "cat"])
def test_output_shape(graph, num_layers, out_dim, jk):
    edge_index, x, _, deg = graph
    model = PNA(
        DIM,
        16,
        num_layers=num_layers,
        out_features_dim=out_dim,
        jk=jk,
        deg=deg,
        towers=4,
        **PNA_KWARGS,
    )
    out = model(edge_index, x)
    assert out.shape == (NUM_NODES, out_dim if out_dim is not None else 16)
    assert len(model.convs) == num_layers
    assert all(isinstance(conv, PNAConv) for conv in model.convs)
    assert bool(mx.all(mx.isfinite(out)).item())


def test_edge_features_and_training(graph):
    edge_index, x, edge_features, deg = graph
    labels = mx.array(np.random.default_rng(1).integers(0, 3, NUM_NODES))
    mx.random.seed(0)
    model = PNA(
        DIM,
        16,
        num_layers=2,
        out_features_dim=3,
        norm="batch",
        deg=deg,
        edge_features_dim=EDGE_DIM,
        **PNA_KWARGS,
    )
    with pytest.raises(ValueError):
        model(edge_index, x, edge_weights=mx.ones((NUM_EDGES,)))

    def loss_fn(model):
        logits = model(edge_index, x, edge_features=edge_features)
        return nn.losses.cross_entropy(logits, labels, reduction="mean")

    optimizer = optim.Adam(learning_rate=0.01)
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    first = None
    for _ in range(30):
        loss, grads = loss_and_grad(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        first = loss.item() if first is None else first
    assert loss.item() < first
