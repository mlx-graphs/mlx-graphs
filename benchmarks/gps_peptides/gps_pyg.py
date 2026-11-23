"""PyG twin of examples/graph_gps_peptides.py (same data, same model and
hyperparameters), on CPU with 2 threads."""

import json
import time
from argparse import ArgumentParser

import numpy as np
import torch
from torch.nn import BatchNorm1d, Embedding, Linear, ModuleList, ReLU, Sequential
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, GPSConv, global_mean_pool

WALK_LENGTH = 20


def average_precision(labels, scores):
    # same as examples/graph_gps_peptides.py
    aps = []
    for task in range(labels.shape[1]):
        y, s = labels[:, task], scores[:, task]
        if y.min() == y.max():
            continue
        order = np.argsort(-s, kind="stable")
        y = y[order]
        precision = np.cumsum(y) / np.arange(1, y.size + 1)
        aps.append(float((precision * y).sum() / y.sum()))
    return float(np.mean(aps))


torch.set_num_threads(2)


class FeatureEncoder(torch.nn.Module):
    def __init__(self, vocab_sizes, dims):
        super().__init__()
        self.embeddings = ModuleList([Embedding(v, dims) for v in vocab_sizes])

    def forward(self, features):
        return sum(emb(features[:, i]) for i, emb in enumerate(self.embeddings))


class GPS(torch.nn.Module):
    def __init__(self, atom_vocab, bond_vocab, channels, pe_dim, num_layers, heads,
                 attn_dropout, dropout):  # fmt: skip
        super().__init__()
        self.node_emb = FeatureEncoder(atom_vocab, channels - pe_dim)
        self.pe_lin = Linear(WALK_LENGTH, pe_dim)
        self.pe_norm = BatchNorm1d(WALK_LENGTH)
        self.edge_emb = FeatureEncoder(bond_vocab, channels)
        self.convs = ModuleList()
        for _ in range(num_layers):
            nn = Sequential(
                Linear(channels, channels), ReLU(), Linear(channels, channels)
            )
            self.convs.append(
                GPSConv(channels, GINEConv(nn), heads=heads, dropout=dropout,
                        attn_type="multihead",
                        attn_kwargs={"dropout": attn_dropout})
            )  # fmt: skip
        self.mlp = Sequential(
            Linear(channels, channels // 2), ReLU(),
            Linear(channels // 2, channels // 4), ReLU(),
            Linear(channels // 4, 10),
        )  # fmt: skip

    def forward(self, x, pe, edge_index, edge_attr, batch):
        x = torch.cat((self.node_emb(x), self.pe_lin(self.pe_norm(pe))), 1)
        edge_attr = self.edge_emb(edge_attr)
        for conv in self.convs:
            x = conv(x, edge_index, batch, edge_attr=edge_attr)
        return self.mlp(global_mean_pool(x, batch))


def load_split(path, max_graphs=None):
    z = np.load(path)
    nn, ne = z["nn"], z["ne"]
    no, eo = np.concatenate([[0], np.cumsum(nn)]), np.concatenate([[0], np.cumsum(ne)])
    x = torch.tensor(z["x"].astype(np.int64))
    ei = torch.tensor(z["ei"].astype(np.int64))
    ea = torch.tensor(z["ea"].astype(np.int64))
    y, pe = torch.tensor(z["y"]), torch.tensor(z["pe"])
    out = []
    for i in range(len(nn) if max_graphs is None else max_graphs):
        out.append(Data(x=x[no[i]:no[i + 1]], edge_index=ei[eo[i]:eo[i + 1]].t(),
                        edge_attr=ea[eo[i]:eo[i + 1]], y=y[i:i + 1],
                        pe=pe[no[i]:no[i + 1]]))  # fmt: skip
    return out, z["x"].max(0) + 1, z["ea"].max(0) + 1


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    labels, scores = [], []
    for d in loader:
        scores.append(model(d.x, d.pe, d.edge_index, d.edge_attr, d.batch).numpy())
        labels.append(d.y.numpy())
    model.train()
    return average_precision(np.concatenate(labels), np.concatenate(scores))


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data, atom_vocab, bond_vocab = {}, 0, 0
    for s in ["train", "val", "test"]:
        mg = args.max_train_graphs if s == "train" else None
        data[s], av, bv = load_split(f"{args.data_dir}/{s}.npz", mg)
        atom_vocab, bond_vocab = np.maximum(atom_vocab, av), np.maximum(bond_vocab, bv)
    model = GPS(atom_vocab.tolist(), bond_vocab.tolist(), args.channels,
                args.pe_dim, args.num_layers, args.heads, args.attn_dropout,
                args.dropout)  # fmt: skip
    train_loader = DataLoader(data["train"], args.batch_size, shuffle=True)
    val_loader = DataLoader(data["val"], args.eval_batch_size)
    test_loader = DataLoader(data["test"], args.eval_batch_size)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=args.patience, min_lr=args.min_lr
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()
    best_val, best_test, history = -1.0, -1.0, []
    for epoch in range(1, args.epochs + 1):
        start, total_loss, total = time.time(), 0.0, 0
        model.train()
        for d in train_loader:
            optimizer.zero_grad()
            out = model(d.x, d.pe, d.edge_index, d.edge_attr, d.batch)
            loss = loss_fn(out, d.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * d.num_graphs
            total += d.num_graphs
        train_time = time.time() - start
        val_ap, test_ap = evaluate(model, val_loader), evaluate(model, test_loader)
        scheduler.step(val_ap)
        if val_ap > best_val:
            best_val, best_test = val_ap, test_ap
        history.append(dict(epoch=epoch, loss=total_loss / total, val_ap=val_ap,
                            test_ap=test_ap, time=train_time))  # fmt: skip
        print(
            f"Epoch {epoch:03d} | loss {total_loss / total:.4f} | val AP "
            f"{val_ap:.4f} | test AP {test_ap:.4f} | {train_time:.1f}s",
            flush=True,
        )
    print(json.dumps(dict(best_val_ap=best_val, test_ap=best_test, history=history)))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--pe_dim", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--attn_dropout", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--max_train_graphs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
