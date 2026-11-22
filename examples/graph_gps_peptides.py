"""GPS graph transformer on Peptides-func (LRGB), after PyG's
``examples/graph_gps.py``.

PyG's example trains GPS on ZINC; this one uses the multi-label Peptides-func
task of the Long Range Graph Benchmark, where global attention matters. The
model follows the PyG example: random-walk structural encodings (RWSE) are
normalized, projected and concatenated to the node embeddings, then GPS
layers combine a local GINE convolution with global multi-head attention,
and an MLP on the mean-pooled graph embeddings gives the logits of the 10
functions. The metric is the average precision (AP) over the 10 tasks.

Run with ``python examples/graph_gps_peptides.py`` (the dataset, ~30 MB, is
downloaded and preprocessed on the first run).

References:
    - Rampášek et al., "Recipe for a General, Powerful, Scalable Graph
      Transformer", NeurIPS 2022 (https://arxiv.org/abs/2205.12454)
    - Dwivedi et al., "Long Range Graph Benchmark", NeurIPS 2022
      (https://arxiv.org/abs/2206.08164)
"""

import json
import time
from argparse import ArgumentParser

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_map

from mlx_graphs.datasets import LRGBDataset
from mlx_graphs.loaders import Dataloader
from mlx_graphs.nn import GINEConv, GPSConv, RWSEEncoder
from mlx_graphs.nn.pooling import global_mean_pool
from mlx_graphs.transforms import AddRandomWalkPE

WALK_LENGTH = 20


class FeatureEncoder(nn.Module):
    """Sum of one embedding per categorical feature (as OGB's AtomEncoder /
    BondEncoder), initialized like ``torch.nn.Embedding`` (N(0, 1))."""

    def __init__(self, vocab_sizes, dims):
        super().__init__()
        self.embeddings = [nn.Embedding(size, dims) for size in vocab_sizes]
        for embedding in self.embeddings:
            embedding.weight = mx.random.normal(embedding.weight.shape)

    def __call__(self, features):
        return sum(emb(features[:, i]) for i, emb in enumerate(self.embeddings))


class GPS(nn.Module):
    def __init__(
        self,
        atom_vocab,
        bond_vocab,
        channels: int,
        pe_dim: int,
        num_layers: int,
        num_tasks: int,
        heads: int = 4,
        attn_dropout: float = 0.5,
        dropout: float = 0.0,
        attn_type: str = "multihead",
    ):
        super().__init__()
        self.node_emb = FeatureEncoder(atom_vocab, channels - pe_dim)
        self.pe_encoder = RWSEEncoder(WALK_LENGTH, pe_dim)  # pe_lin(pe_norm(pe))
        self.edge_emb = FeatureEncoder(bond_vocab, channels)
        attn_kwargs = {"dropout": attn_dropout}
        if attn_type == "performer":
            attn_kwargs = {"head_channels": channels // heads, "dropout": attn_dropout}
        self.convs = []
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(channels, channels), nn.ReLU(), nn.Linear(channels, channels)
            )
            self.convs.append(
                GPSConv(
                    channels,
                    GINEConv(mlp),
                    heads=heads,
                    dropout=dropout,
                    attn_type=attn_type,
                    attn_kwargs=attn_kwargs,
                )
            )
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.ReLU(),
            nn.Linear(channels // 2, channels // 4),
            nn.ReLU(),
            nn.Linear(channels // 4, num_tasks),
        )

    def __call__(self, batch):
        x = mx.concatenate(
            [
                self.node_emb(batch.node_features),
                self.pe_encoder(batch.random_walk_pe),
            ],
            axis=1,
        )
        edge_features = self.edge_emb(batch.edge_features)
        for conv in self.convs:
            x = conv(
                batch.edge_index,
                x,
                batch.batch_indices,
                edge_features=edge_features,
            )
        x = global_mean_pool(x, batch.batch_indices, batch.num_graphs)
        return self.mlp(x)


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Mean over tasks of the average precision (as OGB / LRGB's ``eval_ap``),
    skipping tasks without both positive and negative labels."""
    aps = []
    for task in range(labels.shape[1]):
        y, s = labels[:, task], scores[:, task]
        if y.min() == y.max():
            continue
        order = np.argsort(-s, kind="stable")
        y = y[order]
        hits = np.cumsum(y)
        precision = hits / np.arange(1, y.size + 1)
        aps.append(float((precision * y).sum() / y.sum()))
    return float(np.mean(aps))


def loss_fn(model, batch):
    logits = model(batch)
    return nn.losses.binary_cross_entropy(
        logits, batch.graph_labels, with_logits=True, reduction="mean"
    )


def evaluate(model, loader):
    model.eval()
    labels, scores = [], []
    for batch in loader:
        logits = model(batch)
        scores.append(np.array(logits))
        labels.append(np.array(batch.graph_labels))
    model.train()
    return average_precision(np.concatenate(labels), np.concatenate(scores))


def main(args):
    # Batches have different shapes, so freed buffers are rarely reused: bound
    # MLX's buffer cache to keep the memory footprint small.
    mx.set_cache_limit(int(args.cache_limit_gb * 1024**3))
    mx.random.seed(args.seed)
    np.random.seed(args.seed)
    transform = AddRandomWalkPE(WALK_LENGTH, attr_name="random_walk_pe")
    datasets = {
        split: LRGBDataset(
            "peptides-func", split, base_dir=args.data_dir, pre_transform=transform
        )
        for split in ["train", "val", "test"]
    }
    if args.max_train_graphs:
        datasets["train"] = datasets["train"][: args.max_train_graphs]

    def vocab(attr):
        return (
            np.max(
                [
                    np.array(getattr(g, attr)).max(0)
                    for ds in datasets.values()
                    for g in ds.graphs
                ],
                axis=0,
            )
            + 1
        ).tolist()

    model = GPS(
        vocab("node_features"),
        vocab("edge_features"),
        channels=args.channels,
        pe_dim=args.pe_dim,
        num_layers=args.num_layers,
        num_tasks=10,
        heads=args.heads,
        attn_dropout=args.attn_dropout,
        dropout=args.dropout,
        attn_type=args.attn_type,
    )
    mx.eval(model.parameters())
    # bias correction, as torch.optim.Adam (MLX's default is False)
    optimizer = optim.Adam(learning_rate=args.lr, bias_correction=True)
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    train_loader = Dataloader(datasets["train"], args.batch_size, shuffle=True)
    val_loader = Dataloader(datasets["val"], args.eval_batch_size)
    test_loader = Dataloader(datasets["test"], args.eval_batch_size)

    # ReduceLROnPlateau on the validation AP, as in PyG's GPS example.
    best_val, best_test, bad_epochs = -1.0, -1.0, 0
    history = []
    model.train()
    for epoch in range(1, args.epochs + 1):
        start, total_loss, total_graphs = time.time(), 0.0, 0
        for batch in train_loader:
            loss, grads = loss_and_grad(model, batch)
            if args.weight_decay > 0:  # L2 penalty, as torch.optim.Adam
                grads = tree_map(
                    lambda g, p: g + args.weight_decay * p, grads, model.parameters()
                )
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)
            total_loss += loss.item() * batch.num_graphs
            total_graphs += batch.num_graphs
        train_time = time.time() - start
        val_ap, test_ap = evaluate(model, val_loader), evaluate(model, test_loader)
        if val_ap > best_val:
            best_val, best_test, bad_epochs = val_ap, test_ap, 0
        else:
            bad_epochs += 1
            if bad_epochs > args.patience:
                optimizer.learning_rate = max(
                    optimizer.learning_rate.item() * 0.5, args.min_lr
                )
                bad_epochs = 0
        history.append(
            dict(
                epoch=epoch,
                loss=total_loss / total_graphs,
                val_ap=val_ap,
                test_ap=test_ap,
                time=train_time,
            )
        )
        print(
            f"Epoch {epoch:03d} | loss {total_loss / total_graphs:.4f} | "
            f"val AP {val_ap:.4f} | test AP {test_ap:.4f} | {train_time:.1f}s",
            flush=True,
        )
    print(json.dumps(dict(best_val_ap=best_val, test_ap=best_test, history=history)))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--pe_dim", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--attn_dropout", type=float, default=0.5)
    parser.add_argument(
        "--attn_type", default="multihead", choices=["multihead", "performer"]
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--max_train_graphs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache_limit_gb", type=float, default=1.0)
    main(parser.parse_args())
