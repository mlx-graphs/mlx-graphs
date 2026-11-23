"""Tests of TGNMemory, its building blocks and LastNeighborLoader, with parity
tests against torch_geometric.nn.models.tgn."""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pytest

from mlx_graphs.data import TemporalData
from mlx_graphs.loaders import TemporalDataLoader
from mlx_graphs.nn import TGNMemory, TransformerConv
from mlx_graphs.nn.models.tgn import (
    GRUCell,
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader,
    MeanAggregator,
    TimeEncoder,
)

NUM_NODES, RAW_DIM, MEM_DIM, TIME_DIM = 12, 3, 4, 5
TOL = 1e-4


@pytest.fixture(autouse=True)
def cpu_device():
    """Runs on the CPU: GPU matmuls use reduced precision, and the recurrent
    memory updates accumulate the differences."""
    device = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(device)


def _events(num_events=60, seed=0, ties=True):
    rng = np.random.default_rng(seed)
    src = rng.integers(0, NUM_NODES // 2, num_events)
    dst = rng.integers(NUM_NODES // 2, NUM_NODES, num_events)
    if ties:
        t = np.sort(rng.integers(1, num_events // 2, num_events))
    else:
        t = np.arange(1, num_events + 1) * 3
    msg = rng.standard_normal((num_events, RAW_DIM)).astype(np.float32)
    return src, dst, t, msg


def _to_mx(tensor):
    return mx.array(tensor.detach().numpy())


@pytest.fixture
def pyg_tgn(monkeypatch):
    """PyG's tgn module, with ``scatter_argmax`` giving out-of-range indices to
    empty groups (as with torch-scatter). PyG's fallback without
    torch-scatter returns ``dim_size - 1`` for them, which makes
    ``LastAggregator`` copy an unrelated message to nodes without messages."""
    torch = pytest.importorskip("torch")
    tgn = pytest.importorskip("torch_geometric.nn.models.tgn")

    def scatter_argmax(src, index, dim=0, dim_size=None):
        res = src.new_empty(dim_size)
        res.scatter_reduce_(0, index, src, reduce="amax", include_self=False)
        out = index.new_full((dim_size,), fill_value=src.size(0))
        nonzero = (src == res[index]).nonzero().view(-1)
        out[index[nonzero]] = nonzero
        return out

    monkeypatch.setattr(tgn, "scatter_argmax", scatter_argmax)

    # PyG's LastNeighborLoader.insert sorts nodes with `Tensor.sort()`, whose
    # order of ties is unspecified: x86 builds of torch use an unstable SIMD
    # sort, so the neighbors kept when a node gets more than `size` events
    # depend on the CPU. mlx-graphs follows the stable order.
    stable_sort = torch.Tensor.sort

    def sort(self, *args, **kwargs):
        kwargs.setdefault("stable", True)
        return stable_sort(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "sort", sort)
    torch.manual_seed(0)
    return torch, tgn


def _copy_memory(torch, memory, pyg_memory):
    with torch.no_grad():
        pyg_memory.time_enc.lin.weight.copy_(
            torch.tensor(np.array(memory.time_enc.lin.weight))
        )
        pyg_memory.time_enc.lin.bias.copy_(
            torch.tensor(np.array(memory.time_enc.lin.bias))
        )
        for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
            getattr(pyg_memory.gru, name).copy_(
                torch.tensor(np.array(getattr(memory.gru, name)))
            )


def _memories(torch, tgn, aggr):
    memory = TGNMemory(
        NUM_NODES,
        RAW_DIM,
        MEM_DIM,
        TIME_DIM,
        IdentityMessage(RAW_DIM, MEM_DIM, TIME_DIM),
        LastAggregator() if aggr == "last" else MeanAggregator(),
    )
    pyg_memory = tgn.TGNMemory(
        NUM_NODES,
        RAW_DIM,
        MEM_DIM,
        TIME_DIM,
        tgn.IdentityMessage(RAW_DIM, MEM_DIM, TIME_DIM),
        tgn.LastAggregator() if aggr == "last" else tgn.MeanAggregator(),
    )
    _copy_memory(torch, memory, pyg_memory)
    return memory, pyg_memory


# Building blocks


def test_gru_cell_matches_torch(pyg_tgn):
    torch, _ = pyg_tgn
    cell = GRUCell(6, 4)
    ref = torch.nn.GRUCell(6, 4)
    with torch.no_grad():
        for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
            getattr(ref, name).copy_(torch.tensor(np.array(getattr(cell, name))))
    x = np.random.default_rng(0).standard_normal((5, 6)).astype(np.float32)
    h = np.random.default_rng(1).standard_normal((5, 4)).astype(np.float32)
    out = cell(mx.array(x), mx.array(h))
    expected = ref(torch.tensor(x), torch.tensor(h))
    assert np.allclose(np.array(out), expected.detach().numpy(), atol=TOL)


def test_time_encoder_and_identity_message():
    enc = TimeEncoder(8)
    assert enc.out_channels == enc.out_features_dim == 8
    out = enc(mx.array([0.0, 1.0, 10.0]))
    assert out.shape == (3, 8)
    expected = np.cos(
        np.array([[0.0], [1.0], [10.0]]) @ np.array(enc.lin.weight).T
        + np.array(enc.lin.bias)
    )
    assert np.allclose(np.array(out), expected, atol=1e-5)

    msg = IdentityMessage(3, 4, 5)
    assert msg.out_channels == 3 + 2 * 4 + 5
    out = msg(mx.ones((2, 4)), mx.zeros((2, 4)), mx.ones((2, 3)), mx.ones((2, 5)))
    assert out.shape == (2, 16)


def test_last_and_mean_aggregator():
    msg = mx.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    index = mx.array([0, 0, 2, 2, 2])
    t = mx.array([5, 3, 1, 7, 7])
    out = LastAggregator()(msg, index, t, 4)
    # node 0: latest t=5 -> 1; node 2: tie at t=7, the last stored wins -> 5
    assert np.array(out).reshape(-1).tolist() == [1.0, 0.0, 5.0, 0.0]
    out = MeanAggregator()(msg, index, t, 4)
    assert np.allclose(np.array(out).reshape(-1), [1.5, 0.0, 4.0, 0.0])


def test_state_is_not_a_parameter():
    memory = TGNMemory(
        NUM_NODES, RAW_DIM, MEM_DIM, TIME_DIM,
        IdentityMessage(RAW_DIM, MEM_DIM, TIME_DIM), LastAggregator(),
    )  # fmt: skip
    keys = set(memory.parameters().keys())
    assert keys == {"msg_s_module", "msg_d_module", "aggr_module", "time_enc", "gru"}
    assert memory.memory.shape == (NUM_NODES, MEM_DIM)
    assert memory.last_update.tolist() == [0] * NUM_NODES


def _memory():
    return TGNMemory(
        NUM_NODES, RAW_DIM, MEM_DIM, TIME_DIM,
        IdentityMessage(RAW_DIM, MEM_DIM, TIME_DIM), LastAggregator(),
    )  # fmt: skip


def test_non_integer_timestamps_raise():
    """Timestamps are int64 as in PyG: non-integer values are not truncated
    silently, integer-valued floats are accepted."""
    src, dst, msg = mx.array([0, 1]), mx.array([6, 7]), mx.ones((2, RAW_DIM))
    memory = _memory()
    with pytest.raises(ValueError, match="integer timestamps"):
        memory.update_state(src, dst, mx.array([1.5, 2.0]), msg)
    memory.update_state(src, dst, mx.array([3.0, 4.0]), msg)
    memory.eval()
    assert memory.last_update[[0, 6]].tolist() == [3, 3]


def test_memory_state_roundtrip(tmp_path):
    """The memory is not in save_weights; memory_state() checkpoints it."""
    memory = _memory()
    memory.eval()
    memory.update_state(
        mx.array([0, 1]), mx.array([6, 7]), mx.array([5, 9]), mx.ones((2, RAW_DIM))
    )
    path = str(tmp_path / "memory.npz")
    mx.savez(path, **memory.memory_state())
    other = _memory()
    other.load_memory_state(mx.load(path))
    assert mx.allclose(other.memory, memory.memory)
    assert other.last_update.tolist() == memory.last_update.tolist()
    assert other.last_update[[0, 1, 2]].tolist() == [5, 9, 0]
    with pytest.raises(ValueError, match="shape"):
        other.load_memory_state({"memory": mx.zeros((2, 2)), "last_update": [0]})


# Memory parity


@pytest.mark.parametrize("aggr", ["last", "mean"])
@pytest.mark.parametrize("ties", [True, False])
def test_tgn_memory_matches_pyg(pyg_tgn, aggr, ties):
    torch, tgn = pyg_tgn
    memory, pyg_memory = _memories(torch, tgn, aggr)
    src, dst, t, msg = _events(ties=ties)
    rng = np.random.default_rng(1)

    def check(n_id):
        z, last_update = memory(mx.array(n_id))
        ref_z, ref_last_update = pyg_memory(torch.tensor(n_id))
        assert np.allclose(np.array(z), ref_z.detach().numpy(), atol=TOL)
        assert np.array(last_update).tolist() == ref_last_update.tolist()

    batches = [slice(i, i + 6) for i in range(0, len(src), 6)]
    for step, sl in enumerate(batches):
        if step == 6:  # flush into the memory when switching to eval mode
            memory.eval()
            pyg_memory.eval()
            assert np.allclose(
                np.array(memory.memory), pyg_memory.memory.detach().numpy(), atol=TOL
            )
            assert memory.last_update.tolist() == pyg_memory.last_update.tolist()
        extra = rng.integers(0, NUM_NODES, 3)
        n_id = np.unique(np.concatenate([src[sl], dst[sl], extra]))
        check(n_id)
        memory.update_state(
            mx.array(src[sl]), mx.array(dst[sl]), mx.array(t[sl]), mx.array(msg[sl])
        )
        pyg_memory.update_state(
            torch.tensor(src[sl]),
            torch.tensor(dst[sl]),
            torch.tensor(t[sl]),
            torch.tensor(msg[sl]),
        )
        if memory.training:
            memory.detach()
            pyg_memory.detach()

    assert np.allclose(
        np.array(memory.memory), pyg_memory.memory.detach().numpy(), atol=TOL
    )
    # back to training mode, then a full reset
    memory.train()
    pyg_memory.train()
    check(np.arange(NUM_NODES))
    memory.reset_state()
    pyg_memory.reset_state()
    check(np.arange(NUM_NODES))


def test_eval_flush_through_parent_module(pyg_tgn):
    """`.eval()` on a model containing the memory flushes it too."""
    torch, tgn = pyg_tgn
    memory, pyg_memory = _memories(torch, tgn, "last")

    class Wrapper(nn.Module):
        def __init__(self, memory):
            super().__init__()
            self.memory = memory

    model = Wrapper(memory)
    src, dst, t, msg = _events(12)
    memory.update_state(mx.array(src), mx.array(dst), mx.array(t), mx.array(msg))
    pyg_memory.update_state(
        torch.tensor(src), torch.tensor(dst), torch.tensor(t), torch.tensor(msg)
    )
    before = np.array(memory.memory)
    model.eval()
    torch.nn.Sequential(pyg_memory).eval()
    assert not np.allclose(before, np.array(memory.memory))
    assert np.allclose(
        np.array(memory.memory), pyg_memory.memory.detach().numpy(), atol=TOL
    )


def test_tgn_memory_gradients_match_pyg(pyg_tgn):
    torch, tgn = pyg_tgn
    memory, pyg_memory = _memories(torch, tgn, "last")
    src, dst, t, msg = _events(30, ties=False)
    for sl in (slice(0, 10), slice(10, 20)):
        memory.update_state(
            mx.array(src[sl]), mx.array(dst[sl]), mx.array(t[sl]), mx.array(msg[sl])
        )
        pyg_memory.update_state(
            torch.tensor(src[sl]),
            torch.tensor(dst[sl]),
            torch.tensor(t[sl]),
            torch.tensor(msg[sl]),
        )
        memory.detach()
        pyg_memory.detach()

    n_id = np.arange(NUM_NODES)
    weight = np.random.default_rng(0).standard_normal((NUM_NODES, MEM_DIM))
    weight = weight.astype(np.float32)

    def loss_fn():
        z, _ = memory(mx.array(n_id))
        return (z * mx.array(weight)).sum()

    loss, grads = nn.value_and_grad(memory, loss_fn)()
    z, _ = pyg_memory(torch.tensor(n_id))
    ref_loss = (z * torch.tensor(weight)).sum()
    ref_loss.backward()

    assert np.isclose(loss.item(), ref_loss.item(), atol=TOL)
    for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
        ref = getattr(pyg_memory.gru, name).grad.numpy()
        assert np.allclose(np.array(grads["gru"][name]), ref, atol=TOL)
    ref = pyg_memory.time_enc.lin.weight.grad.numpy()
    assert np.allclose(np.array(grads["time_enc"]["lin"]["weight"]), ref, atol=TOL)
    # the state is untouched by value_and_grad and by the optimizer
    state = np.array(memory.memory)
    optim.SGD(learning_rate=0.1).update(memory, grads)
    assert np.array_equal(state, np.array(memory.memory))


def test_update_state_inside_loss_function():
    """Calling update_state inside the differentiated function (PyG's order)
    gives the same state and gradients as calling it afterwards."""
    src, dst, t, msg = _events(24, ties=False)

    def run(inside):
        mx.random.seed(3)
        memory = TGNMemory(
            NUM_NODES, RAW_DIM, MEM_DIM, TIME_DIM,
            IdentityMessage(RAW_DIM, MEM_DIM, TIME_DIM), LastAggregator(),
        )  # fmt: skip
        grads_list = []
        for sl in (slice(0, 8), slice(8, 16), slice(16, 24)):
            args = [mx.array(a[sl]) for a in (src, dst, t, msg)]

            def loss_fn():
                z, _ = memory(mx.arange(NUM_NODES))
                if inside:
                    memory.update_state(*args)
                return (z**2).sum()

            _, grads = nn.value_and_grad(memory, loss_fn)()
            if not inside:
                memory.update_state(*args)
            memory.detach()
            grads_list.append(np.array(grads["gru"]["weight_ih"]))
        return np.array(memory.memory), grads_list

    state_in, grads_in = run(True)
    state_out, grads_out = run(False)
    assert np.allclose(state_in, state_out, atol=1e-6)
    for a, b in zip(grads_in, grads_out):
        assert np.allclose(a, b, atol=1e-6)


# LastNeighborLoader


def test_last_neighbor_loader_matches_pyg(pyg_tgn):
    torch, tgn = pyg_tgn
    loader = LastNeighborLoader(NUM_NODES, size=4)
    ref = tgn.LastNeighborLoader(NUM_NODES, size=4)
    rng = np.random.default_rng(0)
    for step in range(6):
        # some nodes get more than `size` new neighbors in one insertion
        n = 10 if step % 2 else 3
        src = rng.integers(0, 4, n)
        dst = rng.integers(4, NUM_NODES, n)
        loader.insert(mx.array(src), mx.array(dst))
        ref.insert(torch.tensor(src), torch.tensor(dst))
        n_id = np.unique(rng.integers(0, NUM_NODES, 5))
        out = loader(mx.array(n_id))
        expected = ref(torch.tensor(n_id))
        for a, b in zip(out, expected):
            assert np.array(a).tolist() == b.tolist()
    loader.reset_state()
    n_id, edge_index, e_id = loader(mx.array([0, 1]))
    assert n_id.tolist() == [0, 1] and edge_index.shape == (2, 0)


# Full TGN pipeline (PyG's examples/tgn.py) on a small stream


class _GraphAttentionEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim, msg_dim, time_dim):
        super().__init__()
        self.conv = TransformerConv(
            in_dim, out_dim // 2, heads=2, edge_features_dim=msg_dim + time_dim
        )

    def __call__(self, x, last_update, edge_index, t, msg, time_enc):
        rel_t = (last_update[edge_index[0]] - t).astype(x.dtype)
        edge_attr = mx.concatenate([time_enc(rel_t), msg], axis=-1)
        return self.conv(edge_index, x, edge_attr)


class _LinkPredictor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin_src = nn.Linear(dim, dim)
        self.lin_dst = nn.Linear(dim, dim)
        self.lin_final = nn.Linear(dim, 1)

    def __call__(self, z_src, z_dst):
        return self.lin_final(nn.relu(self.lin_src(z_src) + self.lin_dst(z_dst)))


class _TGN(nn.Module):
    def __init__(self, num_nodes, msg_dim, dim):
        super().__init__()
        self.memory = TGNMemory(
            num_nodes, msg_dim, dim, dim,
            IdentityMessage(msg_dim, dim, dim), LastAggregator(),
        )  # fmt: skip
        self.gnn = _GraphAttentionEmbedding(dim, dim, msg_dim, dim)
        self.link_pred = _LinkPredictor(dim)


def _copy_linear(torch, dst, src):
    with torch.no_grad():
        dst.weight.copy_(torch.tensor(np.array(src.weight)))
        if dst.bias is not None:
            dst.bias.copy_(torch.tensor(np.array(src.bias)))


def test_tgn_training_matches_pyg(pyg_tgn):
    torch, tgn = pyg_tgn
    from torch_geometric.nn import TransformerConv as PyGTransformerConv

    dim = 8
    src, dst, t, msg = _events(80, ties=False)
    data = TemporalData(
        src=mx.array(src), dst=mx.array(dst), t=mx.array(t), msg=mx.array(msg)
    )
    model = _TGN(NUM_NODES, RAW_DIM, dim)

    # PyG modules with the same weights
    pyg_memory = tgn.TGNMemory(
        NUM_NODES, RAW_DIM, dim, dim,
        tgn.IdentityMessage(RAW_DIM, dim, dim), tgn.LastAggregator(),
    )  # fmt: skip
    _copy_memory(torch, model.memory, pyg_memory)
    pyg_conv = PyGTransformerConv(
        dim, dim // 2, heads=2, dropout=0.0, edge_dim=RAW_DIM + dim
    )
    for name in ("lin_key", "lin_query", "lin_value", "lin_edge", "lin_skip"):
        _copy_linear(torch, getattr(pyg_conv, name), getattr(model.gnn.conv, name))
    pyg_lins = {
        name: torch.nn.Linear(dim, 1 if name == "lin_final" else dim)
        for name in ("lin_src", "lin_dst", "lin_final")
    }
    for name, lin in pyg_lins.items():
        _copy_linear(torch, lin, getattr(model.link_pred, name))
    pyg_params = (
        list(pyg_memory.parameters())
        + list(pyg_conv.parameters())
        + [p for lin in pyg_lins.values() for p in lin.parameters()]
    )
    pyg_opt = torch.optim.Adam(pyg_params, lr=0.01)
    opt = optim.Adam(learning_rate=0.01, bias_correction=True)

    neighbor_loader = LastNeighborLoader(NUM_NODES, size=3)
    pyg_neighbor_loader = tgn.LastNeighborLoader(NUM_NODES, size=3)
    assoc = np.zeros(NUM_NODES, dtype=np.int64)
    t_all, msg_all = torch.tensor(t), torch.tensor(msg)
    bce = torch.nn.BCEWithLogitsLoss()

    def pyg_step(batch, neg_dst):
        b_src, b_dst = (
            torch.tensor(np.array(batch.src)),
            torch.tensor(np.array(batch.dst)),
        )
        b_neg = torch.tensor(neg_dst)
        n_id = torch.cat([b_src, b_dst, b_neg]).unique()
        n_id, edge_index, e_id = pyg_neighbor_loader(n_id)
        local = torch.empty(NUM_NODES, dtype=torch.long)
        local[n_id] = torch.arange(n_id.size(0))
        z, last_update = pyg_memory(n_id)
        rel_t = (last_update[edge_index[0]] - t_all[e_id]).to(z.dtype)
        edge_attr = torch.cat([pyg_memory.time_enc(rel_t), msg_all[e_id]], dim=-1)
        z = pyg_conv(z, edge_index, edge_attr)

        def pred(a, b):
            h = pyg_lins["lin_src"](a) + pyg_lins["lin_dst"](b)
            return pyg_lins["lin_final"](h.relu())

        pos = pred(z[local[b_src]], z[local[b_dst]])
        neg = pred(z[local[b_src]], z[local[b_neg]])
        loss = bce(pos, torch.ones_like(pos)) + bce(neg, torch.zeros_like(neg))
        pyg_memory.update_state(
            b_src,
            b_dst,
            torch.tensor(np.array(batch.t)),
            torch.tensor(np.array(batch.msg)),
        )
        pyg_neighbor_loader.insert(b_src, b_dst)
        pyg_opt.zero_grad()
        loss.backward()
        pyg_opt.step()
        pyg_memory.detach()
        return loss.item()

    def loss_fn(batch, neg_dst, n_id, edge_index, e_id):
        z, last_update = model.memory(n_id)
        z = model.gnn(
            z, last_update, edge_index, data.t[e_id], data.msg[e_id],
            model.memory.time_enc,
        )  # fmt: skip
        idx = lambda nodes: mx.array(assoc[np.array(nodes)])  # noqa: E731
        pos = model.link_pred(z[idx(batch.src)], z[idx(batch.dst)])
        neg = model.link_pred(z[idx(batch.src)], z[idx(neg_dst)])
        return nn.losses.binary_cross_entropy(
            pos, mx.ones_like(pos), reduction="mean"
        ) + nn.losses.binary_cross_entropy(neg, mx.zeros_like(neg), reduction="mean")

    step_fn = nn.value_and_grad(model, loss_fn)
    loader = TemporalDataLoader(data, batch_size=10, neg_sampling_ratio=1.0)
    mx.random.seed(0)
    for batch in loader:
        neg_dst = np.array(batch.neg_dst)
        n_id, edge_index, e_id = neighbor_loader(batch.n_id)
        assoc[np.array(n_id)] = np.arange(n_id.size)
        loss, grads = step_fn(batch, batch.neg_dst, n_id, edge_index, e_id)
        model.memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
        neighbor_loader.insert(batch.src, batch.dst)
        opt.update(model, grads)
        model.memory.detach()
        mx.eval(model.parameters(), opt.state)

        ref_loss = pyg_step(batch, neg_dst)
        assert np.isclose(loss.item(), ref_loss, atol=1e-3), (loss.item(), ref_loss)

    # evaluation mode: flush and compare the memories
    model.eval()
    pyg_memory.eval()
    assert np.allclose(
        np.array(model.memory.memory), pyg_memory.memory.detach().numpy(), atol=1e-3
    )
