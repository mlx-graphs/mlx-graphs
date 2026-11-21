import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch
import torch_geometric.nn as pyg_nn
from mlx.utils import tree_flatten

from mlx_graphs.nn import GINEConv, GPSConv
from tests.pyg_copy import load_torch, torch_state

C = 8
# MLX GPU matmuls are only accurate to ~1e-3 relative.
TOL = dict(rtol=1e-2, atol=2e-3)
GPS_RENAMES = [
    (r"^mlp\.(\d)\.", r"mlp.layers.\1."),
    (r"^conv\.nn\.(\d)\.", r"conv.mlp.layers.\1."),
]


def make_graph(seed=0, sizes=(5, 3, 7)):
    rng = np.random.default_rng(seed)
    batch = np.concatenate([np.full(n, i) for i, n in enumerate(sizes)])
    offsets = np.cumsum((0,) + sizes[:-1])
    edges = []
    for n, off in zip(sizes, offsets):
        src = rng.integers(0, n, 2 * n)
        dst = rng.integers(0, n, 2 * n)
        edges.append(np.stack([src, dst]) + off)
    edge_index = np.concatenate(edges, axis=1)
    x = rng.standard_normal((batch.size, C)).astype(np.float32)
    edge_attr = rng.standard_normal((edge_index.shape[1], C)).astype(np.float32)
    return x, edge_index.astype(np.int64), edge_attr, batch.astype(np.int64)


def build(norm="batch_norm", attn_type="multihead", heads=2, with_conv=True, **kw):
    torch.manual_seed(0)
    attn_kwargs = {"head_channels": 4} if attn_type == "performer" else {}

    def torch_mlp():
        return torch.nn.Sequential(
            torch.nn.Linear(C, C), torch.nn.ReLU(), torch.nn.Linear(C, C)
        )

    pyg = pyg_nn.GPSConv(
        C,
        pyg_nn.GINEConv(torch_mlp()) if with_conv else None,
        heads=heads,
        norm=norm,
        attn_type=attn_type,
        attn_kwargs=attn_kwargs,
        **kw,
    )
    mlx_conv = None
    if with_conv:
        mlx_conv = GINEConv(nn.Sequential(nn.Linear(C, C), nn.ReLU(), nn.Linear(C, C)))
    layer = GPSConv(
        C,
        mlx_conv,
        heads=heads,
        norm=norm,
        attn_type=attn_type,
        attn_kwargs=attn_kwargs,
        **kw,
    )
    # non-trivial running statistics for the batch norms
    for name, buf in pyg.named_buffers():
        if name.endswith("running_mean"):
            buf.uniform_(-0.5, 0.5)
        elif name.endswith("running_var"):
            buf.uniform_(0.5, 2.0)
    renames = GPS_RENAMES + [(r"^attn\.fast_attn\.", "attn.")]
    load_torch(layer, torch_state(pyg, renames=renames, drop=[r"^conv\.eps$"]))
    return layer, pyg


def run(layer, pyg, graph):
    x, edge_index, edge_attr, batch = graph
    layer.eval()
    pyg.eval()
    kwargs = {"edge_features": mx.array(edge_attr)} if layer.conv is not None else {}
    out = layer(mx.array(edge_index), mx.array(x), mx.array(batch), **kwargs)
    with torch.no_grad():
        pyg_kwargs = (
            {"edge_attr": torch.tensor(edge_attr)} if pyg.conv is not None else {}
        )
        expected = pyg(
            torch.tensor(x), torch.tensor(edge_index), torch.tensor(batch), **pyg_kwargs
        )
    return np.array(out), expected.numpy()


@pytest.mark.parametrize("norm", ["batch_norm", "graph_norm", "layer_norm", None])
@pytest.mark.parametrize("heads", [1, 2])
def test_gps_conv_matches_pyg(norm, heads):
    layer, pyg = build(norm=norm, heads=heads)
    out, expected = run(layer, pyg, make_graph())
    assert out.shape == (15, C)
    np.testing.assert_allclose(out, expected, **TOL)


def test_gps_conv_without_local_conv_and_act():
    layer, pyg = build(
        with_conv=False, act="leaky_relu", act_kwargs={"negative_slope": 0.3}
    )
    out, expected = run(layer, pyg, make_graph(1))
    np.testing.assert_allclose(out, expected, **TOL)


def test_gps_conv_performer():
    layer, pyg = build(attn_type="performer")
    assert "attn.projection_matrix" not in dict(
        tree_flatten(layer.trainable_parameters())
    )
    out, expected = run(layer, pyg, make_graph(2))
    # the performer normalizer amplifies the matmul rounding errors
    np.testing.assert_allclose(out, expected, rtol=1e-2, atol=1e-2)
    before = np.array(layer.attn.projection_matrix)
    layer.redraw_projection_matrix()
    assert not np.allclose(before, np.array(layer.attn.projection_matrix))


def test_gps_conv_single_graph_and_unsorted_batch():
    layer, pyg = build()
    x, edge_index, edge_attr, batch = make_graph(3)
    # single graph: batch_indices=None
    single = (x[:5], edge_index[:, :10], edge_attr[:10], np.zeros(5, np.int64))
    layer.eval()
    out = layer(
        mx.array(single[1]), mx.array(single[0]), edge_features=mx.array(single[2])
    )
    _, expected = run(layer, pyg, single)
    np.testing.assert_allclose(np.array(out), expected, **TOL)

    # the nodes of a batch can be in any order: the output is permuted the same
    perm = np.random.default_rng(0).permutation(x.shape[0])
    inv = np.argsort(perm)
    ref = layer(
        mx.array(edge_index),
        mx.array(x),
        mx.array(batch),
        edge_features=mx.array(edge_attr),
    )
    shuffled = layer(
        mx.array(inv[edge_index]),
        mx.array(x[perm]),
        mx.array(batch[perm]),
        edge_features=mx.array(edge_attr),
    )
    np.testing.assert_allclose(np.array(shuffled), np.array(ref)[perm], **TOL)


def test_gps_conv_precomputed_layout_and_compile():
    """A layout built on the device (no host sync) gives the default output,
    also with a looser bound on the graph sizes, and the layer compiles."""
    from mlx_graphs.nn.aggr import DenseLayout

    layer, _ = build()
    layer.eval()
    x, edge_index, edge_attr, batch = (mx.array(a) for a in make_graph(5))
    ref = layer(edge_index, x, batch, edge_features=edge_attr)
    for max_num_nodes in (7, 10):  # the largest graph has 7 nodes
        layout = DenseLayout.from_sizes(batch, x.shape[0], 3, max_num_nodes)
        out = layer(edge_index, x, batch, layout=layout, edge_features=edge_attr)
        np.testing.assert_allclose(np.array(out), np.array(ref), **TOL)

    def forward(edge_index, x, batch, edge_attr):
        layout = DenseLayout.from_sizes(batch, x.shape[0], 3, 7)
        return layer(edge_index, x, batch, layout=layout, edge_features=edge_attr)

    out = mx.compile(forward)(edge_index, x, batch, edge_attr)
    np.testing.assert_allclose(np.array(out), np.array(ref), **TOL)
    with pytest.raises(ValueError, match="layout"):
        layer(
            edge_index,
            x,
            batch,
            layout=DenseLayout(None, 3),
            edge_features=edge_attr,
        )


def test_gps_conv_training_and_gradients():
    layer, _ = build(dropout=0.2)
    layer.attn.dropout = 0.3
    x, edge_index, edge_attr, batch = (mx.array(a) for a in make_graph(4))
    layer.train()

    def loss(params):
        layer.update(params)
        return layer(edge_index, x, batch, edge_features=edge_attr).sum()

    grads = dict(tree_flatten(mx.grad(loss)(layer.trainable_parameters())))
    assert all(np.isfinite(np.array(g)).all() for g in grads.values())
    for key in [
        "attn.query_proj.weight",
        "conv.mlp.layers.0.weight",
        "mlp.layers.3.weight",
    ]:
        assert float(mx.abs(grads[key]).sum()) > 0


def test_gps_conv_errors():
    with pytest.raises(ValueError):
        GPSConv(C, None, attn_type="bigbird")
    with pytest.raises(ValueError):
        GPSConv(C, None, norm="unknown_norm")
