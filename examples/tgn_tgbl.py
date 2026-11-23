"""Temporal Graph Network (TGN) on a TGB dynamic link property prediction
dataset, evaluated with the official TGB protocol, mirroring the TGN example
of py-tgb (``examples/linkproppred/tgbl-wiki/tgn.py``).

    python examples/tgn_tgbl.py --dataset tgbl-wiki --epochs 50

The model is trained as in ``tgn_jodie.py``, with one random negative
destination per event. Every validation and test event is then ranked against
its fixed negative destinations of TGB (999 for ``tgbl-wiki``), and the
metric is the mean reciprocal rank (MRR) of the true destinations.

As in py-tgb, the decoder ends with a sigmoid and the loss is
``BCEWithLogits`` of these probabilities, and the test MRR is measured with
the parameters and memory of the epoch with the best validation MRR (with
the same early stopping).

The TGB example scores the positive edges one at a time; here the events of a
mini-batch are scored together, which gives the same scores: in evaluation
mode the memory and the neighbors are only updated after each mini-batch.
"""

import json
import time
from argparse import ArgumentParser

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_map

from mlx_graphs.datasets import TGBLinkPropPredDataset
from mlx_graphs.loaders import TemporalDataLoader
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
        rel_t = (last_update[edge_index[0]] - t).astype(x.dtype)
        edge_attr = mx.concatenate([time_enc(rel_t), msg], axis=-1)
        return self.conv(edge_index, x, edge_attr)


class LinkPredictor(nn.Module):
    """py-tgb's ``modules.decoder.LinkPredictor`` (with a final sigmoid)."""

    def __init__(self, dim: int):
        super().__init__()
        self.lin_src = nn.Linear(dim, dim)
        self.lin_dst = nn.Linear(dim, dim)
        self.lin_final = nn.Linear(dim, 1)

    def __call__(self, z_src, z_dst):
        h = nn.relu(self.lin_src(z_src) + self.lin_dst(z_dst))
        return mx.sigmoid(self.lin_final(h))


class TGN(nn.Module):
    def __init__(self, num_nodes: int, msg_dim: int, mem_dim: int, time_dim: int,
                 emb_dim: int):  # fmt: skip
        super().__init__()
        self.memory = TGNMemory(
            num_nodes, msg_dim, mem_dim, time_dim,
            message_module=IdentityMessage(msg_dim, mem_dim, time_dim),
            aggregator_module=LastAggregator(),
        )  # fmt: skip
        self.gnn = GraphAttentionEmbedding(mem_dim, emb_dim, msg_dim, time_dim)
        self.link_pred = LinkPredictor(emb_dim)

    def embed(self, data, n_id, edge_index, e_id):
        z, last_update = self.memory(n_id)
        return self.gnn(
            z, last_update, edge_index, data.t[e_id], data.msg[e_id],
            self.memory.time_enc,
        )  # fmt: skip


def loss_fn(model, data, n_id, edge_index, e_id, src, pos_dst, neg_dst):
    z = model.embed(data, n_id, edge_index, e_id)
    pos_out = model.link_pred(z[src], z[pos_dst])
    neg_out = model.link_pred(z[src], z[neg_dst])
    # BCEWithLogits of the sigmoid outputs, as py-tgb
    return nn.losses.binary_cross_entropy(
        pos_out, mx.ones_like(pos_out), reduction="mean"
    ) + nn.losses.binary_cross_entropy(
        neg_out, mx.zeros_like(neg_out), reduction="mean"
    )


def local_index(assoc, nodes):
    return mx.array(assoc[np.array(nodes)].astype(np.int32))


def train(model, optimizer, data, loader, neighbor_loader, assoc, min_dst, max_dst):
    model.train()
    model.memory.reset_state()  # start with a fresh memory
    neighbor_loader.reset_state()  # start with an empty graph
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    total_loss = 0.0
    for batch in loader:
        src, pos_dst = batch.src, batch.dst
        neg_dst = mx.random.randint(min_dst, max_dst + 1, src.shape, dtype=src.dtype)
        n_id = np.unique(np.concatenate([np.array(src), np.array(pos_dst),
                                         np.array(neg_dst)]))  # fmt: skip
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[np.array(n_id)] = np.arange(n_id.size)

        loss, grads = loss_and_grad(
            model, data, n_id, edge_index, e_id, local_index(assoc, src),
            local_index(assoc, pos_dst), local_index(assoc, neg_dst),
        )  # fmt: skip
        # Update the memory and the neighbor loader with the ground-truth
        # events, before the optimizer step (as PyG).
        model.memory.update_state(src, pos_dst, batch.t, batch.msg)
        neighbor_loader.insert(src, pos_dst)

        optimizer.update(model, grads)
        model.memory.detach()
        mx.eval(model.parameters(), optimizer.state, loss)
        total_loss += loss.item() * batch.num_events
    return total_loss / loader.data.num_events


def test(model, data, loader, neighbor_loader, assoc, sampler, evaluator, split):
    """MRR of the events of ``loader``, each ranked against its TGB negative
    destinations."""
    model.eval()  # also flushes the pending messages into the memory
    pos_scores, neg_scores = [], []
    for batch in loader:
        src, pos_dst = np.array(batch.src), np.array(batch.dst)
        neg_dst, valid = sampler.query_batch_array(
            batch.src, batch.dst, batch.t, split_mode=split
        )
        neg_dst, valid = np.array(neg_dst), np.array(valid)
        # padding entries score the positive (and are then masked)
        neg_dst = np.where(valid, neg_dst, pos_dst[:, None])
        dst = np.concatenate([pos_dst[:, None], neg_dst], axis=1)  # [B, 1 + K]

        n_id = np.unique(np.concatenate([src, dst.reshape(-1)]))
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[np.array(n_id)] = np.arange(n_id.size)
        z = model.embed(data, n_id, edge_index, e_id)

        z_src = z[local_index(assoc, src)][:, None, :]
        z_dst = z[local_index(assoc, dst)]
        scores = model.link_pred(mx.broadcast_to(z_src, z_dst.shape), z_dst).squeeze(-1)
        scores = np.array(scores)
        pos_scores.append(scores[:, 0])
        neg = np.where(valid, scores[:, 1:], -np.inf)
        neg_scores.append(neg)

        model.memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
        neighbor_loader.insert(batch.src, batch.dst)

    width = max(n.shape[1] for n in neg_scores)
    neg_scores = [
        np.pad(n, ((0, 0), (0, width - n.shape[1])), constant_values=-np.inf)
        for n in neg_scores
    ]
    return evaluator.eval(
        {
            "y_pred_pos": np.concatenate(pos_scores),
            "y_pred_neg": np.concatenate(neg_scores),
            "eval_metric": ["mrr"],
        }
    )["mrr"]


class EarlyStopMonitor:
    """py-tgb's early stopping: keeps the model of the best validation MRR
    (its parameters and memory, as the ``state_dict`` saved by py-tgb) and
    stops after ``patience`` epochs without a relative improvement larger
    than ``tolerance``."""

    def __init__(self, tolerance: float, patience: int):
        self.tolerance, self.patience = tolerance, patience
        self.best, self.best_epoch, self.counter, self.epoch = None, 0, 0, 1
        self.state = None

    def step_check(self, metric: float, model: TGN) -> bool:
        if self.best is None or (metric - self.best) / abs(self.best) > self.tolerance:
            self.best, self.best_epoch, self.counter = metric, self.epoch, 0
            self.state = (
                tree_map(lambda x: x, model.parameters()),
                model.memory.memory_state(),
            )
        else:
            self.counter += 1
        self.epoch += 1
        return self.counter >= self.patience

    def load_checkpoint(self, model: TGN):
        params, memory = self.state
        model.update(params)
        model.memory.load_memory_state(memory)


def main(args):
    mx.random.seed(args.seed)
    np.random.seed(args.seed)
    dataset = TGBLinkPropPredDataset(args.dataset, base_dir=args.base_dir)
    data = dataset[0]
    train_data = data[dataset.train_mask]
    val_data = data[dataset.val_mask]
    test_data = data[dataset.test_mask]
    train_loader = TemporalDataLoader(train_data, batch_size=args.batch_size)
    val_loader = TemporalDataLoader(val_data, batch_size=args.batch_size)
    test_loader = TemporalDataLoader(test_data, batch_size=args.batch_size)
    neighbor_loader = LastNeighborLoader(data.num_nodes, size=args.num_neighbors)
    min_dst, max_dst = dataset.min_dst_idx, dataset.max_dst_idx

    evaluator = dataset.evaluator
    sampler = dataset.negative_sampler
    dataset.load_val_ns()

    model = TGN(data.num_nodes, data.msg.shape[-1], args.mem_dim, args.time_dim,
                args.emb_dim)  # fmt: skip
    # bias_correction=True gives the update rule of torch.optim.Adam
    optimizer = optim.Adam(learning_rate=args.lr, bias_correction=True)
    assoc = np.zeros(data.num_nodes, dtype=np.int64)

    early_stopper = EarlyStopMonitor(args.tolerance, args.patience)
    history = []
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        loss = train(model, optimizer, data, train_loader, neighbor_loader, assoc,
                     min_dst, max_dst)  # fmt: skip
        train_seconds = time.perf_counter() - start
        start = time.perf_counter()
        val_mrr = test(model, data, val_loader, neighbor_loader, assoc, sampler,
                       evaluator, "val")  # fmt: skip
        val_seconds = time.perf_counter() - start
        print(
            f"Epoch: {epoch:02d}, Loss: {loss:.4f}, Val MRR: {val_mrr:.4f} "
            f"(train {train_seconds:.1f} s, val {val_seconds:.1f} s)",
            flush=True,
        )
        history.append(dict(epoch=epoch, loss=loss, val_mrr=val_mrr,
                            train_seconds=train_seconds,
                            val_seconds=val_seconds))  # fmt: skip
        if early_stopper.step_check(val_mrr, model):
            break

    early_stopper.load_checkpoint(model)
    best_epoch = early_stopper.best_epoch
    dataset.load_test_ns()
    start = time.perf_counter()
    test_mrr = test(model, data, test_loader, neighbor_loader, assoc, sampler,
                    evaluator, "test")  # fmt: skip
    test_seconds = time.perf_counter() - start
    print(f"Best epoch: {best_epoch:02d}, Test MRR: {test_mrr:.4f} "
          f"({test_seconds:.1f} s)")  # fmt: skip
    if args.json:
        summary = dict(history=history, best_epoch=best_epoch, test_mrr=test_mrr,
                       test_seconds=test_seconds)  # fmt: skip
        print(json.dumps(summary))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="tgbl-wiki",
                        choices=list(TGBLinkPropPredDataset.names))  # fmt: skip
    parser.add_argument("--base_dir", default=None, help="dataset directory")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--mem_dim", type=int, default=100)
    parser.add_argument("--time_dim", type=int, default=100)
    parser.add_argument("--emb_dim", type=int, default=100)
    parser.add_argument("--num_neighbors", type=int, default=10)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--json", action="store_true", help="print a JSON summary")
    main(parser.parse_args())
