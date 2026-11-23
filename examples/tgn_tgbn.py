"""Temporal Graph Network (TGN) on a TGB dynamic node property prediction
dataset, mirroring the TGN example of py-tgb
(``examples/nodeproppred/tgbn-trade/tgn.py``).

    python examples/tgn_tgbn.py --dataset tgbn-trade --epochs 50

The events are read in mini-batches of 200 and written to the TGN memory and
to the neighbor loader. Whenever a mini-batch reaches the next label
timestamp, the events before that timestamp are processed first, then the
model predicts, for every labeled node, its distribution of interactions over
the classes (e.g. the share of each trading partner of a nation the next
year), trained with a soft-label cross-entropy and evaluated with NDCG@10.
As in py-tgb, the reported test NDCG is the one of the epoch with the best
validation NDCG.
"""

import json
import time
from argparse import ArgumentParser

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from mlx_graphs.datasets import TGBNodePropPredDataset
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


class NodePredictor(nn.Module):
    """py-tgb's ``modules.decoder.NodePredictor``."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin_node = nn.Linear(in_dim, in_dim)
        self.out = nn.Linear(in_dim, out_dim)

    def __call__(self, x):
        return self.out(nn.relu(self.lin_node(x)))


class TGN(nn.Module):
    def __init__(self, num_nodes: int, msg_dim: int, num_classes: int, dim: int):
        super().__init__()
        self.memory = TGNMemory(
            num_nodes, msg_dim, dim, dim,
            message_module=IdentityMessage(msg_dim, dim, dim),
            aggregator_module=LastAggregator(),
        )  # fmt: skip
        self.gnn = GraphAttentionEmbedding(dim, dim, msg_dim, dim)
        self.node_pred = NodePredictor(dim, num_classes)


class Runner:
    def __init__(self, model, data, dataset, neighbor_loader):
        self.model, self.data, self.dataset = model, data, dataset
        self.neighbor_loader = neighbor_loader
        self.assoc = np.zeros(data.num_nodes, dtype=np.int64)
        self.evaluator = dataset.evaluator

    def process_edges(self, events):
        if events.num_events > 0:
            self.model.memory.update_state(events.src, events.dst, events.t,
                                           events.msg)  # fmt: skip
            self.neighbor_loader.insert(events.src, events.dst)

    def predict(self, model, label_srcs):
        n_id, edge_index, e_id = self.neighbor_loader(label_srcs)
        self.assoc[np.array(n_id)] = np.arange(n_id.size)
        z, last_update = model.memory(n_id)
        z = model.gnn(
            z, last_update, edge_index, self.data.t[e_id], self.data.msg[e_id],
            model.memory.time_enc,
        )  # fmt: skip
        z = z[mx.array(self.assoc[np.array(label_srcs)].astype(np.int32))]
        return model.node_pred(z)

    def loss_fn(self, model, prev_events, label_srcs, labels):
        # As py-tgb, the events before the label time are written to the
        # memory inside the computation of the loss (in training mode, the
        # memory updates they cause are differentiable).
        self.process_edges(prev_events)
        pred = self.predict(model, label_srcs)
        # cross-entropy with probability targets, as torch's CrossEntropyLoss
        loss = -(labels * nn.log_softmax(pred, axis=-1)).sum(-1).mean()
        return loss, pred

    def ndcg(self, labels, pred):
        return self.evaluator.eval(
            {"y_true": labels, "y_pred": pred, "eval_metric": ["ndcg"]}
        )["ndcg"]

    def run(self, loader, optimizer=None):
        """One pass over the events of ``loader``, training if an optimizer
        is given."""
        model, dataset = self.model, self.dataset
        training = optimizer is not None
        if training:
            model.train()
            model.memory.reset_state()  # start with a fresh memory
            self.neighbor_loader.reset_state()  # start with an empty graph
            loss_and_grad = nn.value_and_grad(model, self.loss_fn)
        else:
            model.eval()

        label_t = dataset.get_label_time()
        total_loss, total_score, num_label_ts = 0.0, 0.0, 0
        for batch in loader:
            query_t = batch.t[-1].item()
            if query_t > label_t:
                label_tuple = dataset.get_node_label(query_t)
                if label_tuple is None:
                    break
                _, label_srcs, labels = label_tuple
                label_t = dataset.get_label_time()
                previous_day = np.array(batch.t) < label_t
                prev_events, batch = batch[previous_day], batch[~previous_day]
                if training:
                    (loss, pred), grads = loss_and_grad(
                        model, prev_events, label_srcs, labels
                    )
                    optimizer.update(model, grads)
                    mx.eval(model.parameters(), optimizer.state, loss)
                    total_loss += loss.item()
                else:
                    self.process_edges(prev_events)
                    pred = self.predict(model, label_srcs)
                total_score += self.ndcg(labels, pred)
                num_label_ts += 1

            self.process_edges(batch)
            if training:
                model.memory.detach()
        out = {"ndcg": total_score / max(num_label_ts, 1)}
        if training:
            out["ce"] = total_loss / max(num_label_ts, 1)
        return out


def main(args):
    mx.random.seed(args.seed)
    np.random.seed(args.seed)
    dataset = TGBNodePropPredDataset(args.dataset, base_dir=args.base_dir)
    data = dataset[0]
    loaders = {
        split: TemporalDataLoader(
            data[getattr(dataset, f"{split}_mask")], batch_size=args.batch_size
        )
        for split in ("train", "val", "test")
    }
    neighbor_loader = LastNeighborLoader(data.num_nodes, size=args.num_neighbors)
    model = TGN(data.num_nodes, data.msg.shape[-1], dataset.num_classes, args.dim)
    # bias_correction=True gives the update rule of torch.optim.Adam
    optimizer = optim.Adam(learning_rate=args.lr, bias_correction=True)
    runner = Runner(model, data, dataset, neighbor_loader)

    history = []
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        train = runner.run(loaders["train"], optimizer)
        train_seconds = time.perf_counter() - start
        start = time.perf_counter()
        val = runner.run(loaders["val"])
        val_seconds = time.perf_counter() - start
        start = time.perf_counter()
        test = runner.run(loaders["test"])
        test_seconds = time.perf_counter() - start
        dataset.reset_label_time()
        print(
            f"Epoch: {epoch:02d}, Loss: {train['ce']:.4f}, "
            f"Train NDCG: {train['ndcg']:.4f}, Val NDCG: {val['ndcg']:.4f}, "
            f"Test NDCG: {test['ndcg']:.4f} (train {train_seconds:.1f} s, "
            f"val {val_seconds:.1f} s, test {test_seconds:.1f} s)",
            flush=True,
        )
        history.append(dict(
            epoch=epoch, loss=train["ce"], train_ndcg=train["ndcg"],
            val_ndcg=val["ndcg"], test_ndcg=test["ndcg"],
            train_seconds=train_seconds, val_seconds=val_seconds,
            test_seconds=test_seconds,
        ))  # fmt: skip

    best = max(history, key=lambda h: h["val_ndcg"])
    print(f"Best epoch: {best['epoch']:02d}, Val NDCG: {best['val_ndcg']:.4f}, "
          f"Test NDCG: {best['test_ndcg']:.4f}")  # fmt: skip
    if args.json:
        print(json.dumps(dict(history=history, best_epoch=best["epoch"])))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="tgbn-trade",
                        choices=list(TGBNodePropPredDataset.names))  # fmt: skip
    parser.add_argument("--base_dir", default=None, help="dataset directory")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument("--num_neighbors", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--json", action="store_true", help="print a JSON summary")
    main(parser.parse_args())
