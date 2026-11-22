"""Temporal Graph Network (TGN) link prediction on a JODIE dataset, mirroring
PyG's ``examples/tgn.py``.

    python examples/tgn_jodie.py --dataset wikipedia --epochs 50

The events are split by time into training, validation and test events. For
every mini-batch of 200 events, the model reads the memory of the involved
nodes (and of their 10 most recent neighbors), embeds them with a graph
transformer over the most recent interactions, and scores the true
destinations against random negative destinations. The events are then
written to the memory and to the neighbor loader, so the model never sees a
mini-batch before predicting it.

How the TGN memory works with ``nn.value_and_grad``: the memory state is not
an MLX parameter (see :mod:`mlx_graphs.nn.models.tgn`). ``memory(n_id)`` is
called inside the loss function and is differentiable w.r.t. the memory
modules, while ``memory.update_state`` and ``memory.detach`` are called after
computing the gradients and before the optimizer step, as PyG does with
``loss.backward()`` and ``optimizer.step()``.
"""

import json
import time
from argparse import ArgumentParser

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from mlx_graphs.datasets import JODIEDataset
from mlx_graphs.loaders import TemporalDataLoader
from mlx_graphs.metrics import average_precision_score, roc_auc_score
from mlx_graphs.nn import TGNMemory, TransformerConv
from mlx_graphs.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader,
    TimeEncoder,
)


class GraphAttentionEmbedding(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, msg_dim: int, time_dim: int):
        super().__init__()
        self.conv = TransformerConv(
            in_dim, out_dim // 2, heads=2, dropout=0.1,
            edge_features_dim=msg_dim + time_dim,
        )  # fmt: skip

    def __call__(self, x, last_update, edge_index, t, msg, time_enc: TimeEncoder):
        # The time encoder of the memory is passed at call time instead of
        # being stored here, so that it appears only once in the model's
        # parameters (MLX parameter trees do not deduplicate shared modules).
        rel_t = (last_update[edge_index[0]] - t).astype(x.dtype)
        edge_attr = mx.concatenate([time_enc(rel_t), msg], axis=-1)
        return self.conv(edge_index, x, edge_attr)


class LinkPredictor(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin_src = nn.Linear(dim, dim)
        self.lin_dst = nn.Linear(dim, dim)
        self.lin_final = nn.Linear(dim, 1)

    def __call__(self, z_src, z_dst):
        h = nn.relu(self.lin_src(z_src) + self.lin_dst(z_dst))
        return self.lin_final(h)


class TGN(nn.Module):
    def __init__(self, num_nodes: int, msg_dim: int, dim: int = 100):
        super().__init__()
        memory_dim = time_dim = embedding_dim = dim
        self.memory = TGNMemory(
            num_nodes, msg_dim, memory_dim, time_dim,
            message_module=IdentityMessage(msg_dim, memory_dim, time_dim),
            aggregator_module=LastAggregator(),
        )  # fmt: skip
        self.gnn = GraphAttentionEmbedding(memory_dim, embedding_dim, msg_dim, time_dim)
        self.link_pred = LinkPredictor(embedding_dim)

    def __call__(self, data, n_id, edge_index, e_id, src, pos_dst, neg_dst):
        """Logits of the positive and negative events (node indices local to
        ``n_id``)."""
        z, last_update = self.memory(n_id)
        z = self.gnn(
            z, last_update, edge_index, data.t[e_id], data.msg[e_id],
            self.memory.time_enc,
        )  # fmt: skip
        pos_out = self.link_pred(z[src], z[pos_dst])
        neg_out = self.link_pred(z[src], z[neg_dst])
        return pos_out, neg_out


def loss_fn(model, *args):
    pos_out, neg_out = model(*args)
    return nn.losses.binary_cross_entropy(
        pos_out, mx.ones_like(pos_out), reduction="mean"
    ) + nn.losses.binary_cross_entropy(
        neg_out, mx.zeros_like(neg_out), reduction="mean"
    )


def prepare(batch, neighbor_loader, assoc):
    """Neighborhood of the batch and local indices of its nodes."""
    n_id, edge_index, e_id = neighbor_loader(batch.n_id)
    assoc[np.array(n_id)] = np.arange(n_id.size)

    def local(nodes):
        return mx.array(assoc[np.array(nodes)].astype(np.int32))

    return (
        n_id, edge_index, e_id,
        local(batch.src), local(batch.dst), local(batch.neg_dst),
    )  # fmt: skip


def train(model, optimizer, data, loader, neighbor_loader, assoc):
    model.train()
    model.memory.reset_state()  # start with a fresh memory
    neighbor_loader.reset_state()  # start with an empty graph
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    total_loss = 0.0
    for batch in loader:
        args = prepare(batch, neighbor_loader, assoc)
        loss, grads = loss_and_grad(model, data, *args)

        # Update the memory and the neighbor loader with the ground-truth
        # events, before the optimizer step (as PyG).
        model.memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
        neighbor_loader.insert(batch.src, batch.dst)

        optimizer.update(model, grads)
        model.memory.detach()
        mx.eval(model.parameters(), optimizer.state, loss)
        total_loss += loss.item() * batch.num_events
    return total_loss / loader.data.num_events


def test(model, data, loader, neighbor_loader, assoc):
    model.eval()  # also flushes the pending messages into the memory
    mx.random.seed(12345)  # reproducible negative destinations

    aps, aucs = [], []
    for batch in loader:
        args = prepare(batch, neighbor_loader, assoc)
        pos_out, neg_out = model(data, *args)
        y_pred = np.array(mx.sigmoid(mx.concatenate([pos_out, neg_out]))).reshape(-1)
        y_true = np.concatenate([np.ones(pos_out.shape[0]), np.zeros(neg_out.shape[0])])
        aps.append(average_precision_score(y_true, y_pred))
        aucs.append(roc_auc_score(y_true, y_pred))

        model.memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
        neighbor_loader.insert(batch.src, batch.dst)
    return float(np.mean(aps)), float(np.mean(aucs))


def main(args):
    mx.random.seed(args.seed)
    dataset = JODIEDataset(args.dataset, base_dir=args.base_dir)
    data = dataset[0]

    train_data, val_data, test_data = data.train_val_test_split(
        val_ratio=0.15, test_ratio=0.15
    )
    train_loader = TemporalDataLoader(
        train_data, batch_size=args.batch_size, neg_sampling_ratio=1.0
    )
    val_loader = TemporalDataLoader(
        val_data, batch_size=args.batch_size, neg_sampling_ratio=1.0
    )
    test_loader = TemporalDataLoader(
        test_data, batch_size=args.batch_size, neg_sampling_ratio=1.0
    )
    neighbor_loader = LastNeighborLoader(data.num_nodes, size=args.num_neighbors)

    model = TGN(data.num_nodes, data.msg.shape[-1], dim=args.dim)
    # bias_correction=True gives the update rule of torch.optim.Adam
    optimizer = optim.Adam(learning_rate=args.lr, bias_correction=True)
    # Helper array to map global node indices to local ones.
    assoc = np.zeros(data.num_nodes, dtype=np.int64)

    history = []
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        loss = train(model, optimizer, data, train_loader, neighbor_loader, assoc)
        val_ap, val_auc = test(model, data, val_loader, neighbor_loader, assoc)
        test_ap, test_auc = test(model, data, test_loader, neighbor_loader, assoc)
        seconds = time.perf_counter() - start
        print(
            f"Epoch: {epoch:02d}, Loss: {loss:.4f}, "
            f"Val AP: {val_ap:.4f}, Val AUC: {val_auc:.4f}, "
            f"Test AP: {test_ap:.4f}, Test AUC: {test_auc:.4f} ({seconds:.1f} s)"
        )
        history.append(
            dict(
                epoch=epoch, loss=loss, val_ap=val_ap, val_auc=val_auc,
                test_ap=test_ap, test_auc=test_auc, seconds=seconds,
            )
        )  # fmt: skip
    if args.json:
        print(json.dumps(history))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="wikipedia", choices=JODIEDataset.names)
    parser.add_argument("--base_dir", default=None, help="dataset directory")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument("--num_neighbors", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true", help="print a JSON history")
    main(parser.parse_args())
