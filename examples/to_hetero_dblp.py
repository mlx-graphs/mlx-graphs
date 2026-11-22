"""Heterogeneous GNN on DBLP with ``to_hetero``.

Classifies DBLP authors into 4 research areas. DBLP has four node types
(author, paper, term, conference) with edges between authors and papers,
papers and terms, and papers and conferences, in both directions.

Instead of writing a heterogeneous model by hand (see
``hetero_sage_dblp.py``), this example builds an ordinary homogeneous model,
either a built-in ``GraphSAGE``/``GAT`` or a small model with skip connections
written in the PyG style, and converts it with ``to_hetero``: every message
passing layer gets one copy per edge type, every other layer one copy per node
type, and the outputs of edge types pointing to the same node type are summed.

MLX layers have no lazy input sizes (PyG's ``-1``), so the number of features
of every node type is passed as ``input_dims``: the first layers are rebuilt
per node type for its own input size. Conferences have no features and get a
single constant feature ``1.0``, as in PyG's DBLP examples.

Training is full-batch by default, or mini-batch with ``--loader neighbor``
(``NeighborLoader``) or ``--loader hgt`` (``HGTLoader``, the node-budget
sampler of the Heterogeneous Graph Transformer paper).

Run::

    python examples/to_hetero_dblp.py
    python examples/to_hetero_dblp.py --model gat --heads 8
    python examples/to_hetero_dblp.py --model skip --loader hgt --epochs 30
"""

from argparse import ArgumentParser
from time import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from mlx_graphs.data import HeteroGraphData
from mlx_graphs.datasets import DBLP
from mlx_graphs.loaders import HGTLoader, NeighborLoader
from mlx_graphs.nn import GAT, GraphSAGE, Linear, SAGEConv, to_hetero

TARGET = "author"


class SkipSAGE(nn.Module):
    """A homogeneous model in the style of PyG's hetero examples: GraphSAGE
    layers with linear skip connections."""

    def __init__(self, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.conv1 = SAGEConv(hidden_dim, hidden_dim)
        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)
        self.lin2 = Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, edge_index: mx.array, x: mx.array) -> mx.array:
        x = nn.relu(self.conv1(edge_index, x) + self.lin1(x))
        x = self.dropout(x)
        return self.conv2(edge_index, x) + self.lin2(x)


def build_model(args, data: HeteroGraphData, num_classes: int) -> nn.Module:
    # The input size given to the homogeneous model does not matter (the first
    # layers are rebuilt per node type with `input_dims`); a (src, dst) tuple
    # gives every layer separate source and destination weights, as in PyG's
    # GraphSAGE((-1, -1), ...).
    in_dim = (args.hidden_dim, args.hidden_dim)
    if args.model == "sage":
        model = GraphSAGE(
            in_dim,
            args.hidden_dim,
            num_layers=args.num_layers,
            out_features_dim=num_classes,
            dropout=args.dropout,
        )
    elif args.model == "gat":
        model = GAT(
            in_dim,
            args.hidden_dim,
            num_layers=args.num_layers,
            out_features_dim=num_classes,
            heads=args.heads,
            dropout=args.dropout,
            # no self-loops on bipartite edge types (as in PyG's example)
            add_self_loops=False,
        )
    else:
        model = SkipSAGE(args.hidden_dim, num_classes, args.dropout)
    return to_hetero(
        model, data.metadata(), aggr="sum", input_dims=data.num_node_features
    )


def add_constant_features(data: HeteroGraphData) -> HeteroGraphData:
    """Gives every node type without features a single constant feature 1.0."""
    for node_type, num_nodes in data.num_nodes.items():
        if node_type not in data.node_features_dict:
            data.node_features_dict[node_type] = mx.ones((num_nodes, 1))
    return data


def loss_fn(model, batch, n):
    logits = model(batch.edge_index_dict, batch.node_features_dict)[TARGET][:n]
    labels = batch.node_labels_dict[TARGET][:n]
    return nn.losses.cross_entropy(logits, labels, reduction="mean")


def masked_loss_fn(model, data, mask):
    logits = model(data.edge_index_dict, data.node_features_dict)[TARGET]
    losses = nn.losses.cross_entropy(logits, data.node_labels_dict[TARGET])
    return (losses * mask).sum() / mask.sum()


def full_batch_accuracy(model, data, masks):
    model.eval()
    pred = model(data.edge_index_dict, data.node_features_dict)[TARGET].argmax(-1)
    correct = pred == data.node_labels_dict[TARGET]
    model.train()
    return [((correct * m).sum() / m.sum()).item() for m in masks]


def loader_accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    for batch in loader:
        n = batch.batch_size
        out = model(batch.edge_index_dict, batch.node_features_dict)[TARGET][:n]
        correct += (out.argmax(-1) == batch.node_labels_dict[TARGET][:n]).sum().item()
        total += n
    model.train()
    return correct / total


def make_loader(args, data, mask, shuffle, train):
    if args.loader == "neighbor":
        fanout = [int(k) for k in args.num_neighbors.split(",")]
        return NeighborLoader(
            data,
            num_neighbors=fanout if train else [-1] * len(fanout),
            input_nodes=(TARGET, mask),
            batch_size=args.batch_size if train else 1024,
            shuffle=shuffle,
        )
    samples = [int(k) for k in args.num_samples.split(",")]
    return HGTLoader(
        data,
        num_samples=samples,
        input_nodes=(TARGET, mask),
        batch_size=args.batch_size if train else 1024,
        shuffle=shuffle,
    )


def main(args, data: HeteroGraphData = None) -> dict:
    mx.random.seed(args.seed)
    np.random.seed(args.seed)
    if data is None:
        data = DBLP(base_dir=args.data_dir)[0]
    data = add_constant_features(data)
    num_classes = data.num_node_classes[TARGET]

    model = build_model(args, data, num_classes)
    mx.eval(model.parameters())
    num_params = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    print(f"to_hetero({args.model}): {num_params} parameters")

    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=args.weight_decay)
    masks = [
        getattr(data, f"{TARGET}_{split}_mask").astype(mx.float32)
        for split in ("train", "val", "test")
    ]

    if args.loader == "full":
        step = nn.value_and_grad(model, masked_loss_fn)
    else:
        step = nn.value_and_grad(model, loss_fn)
        train_loader = make_loader(args, data, data.author_train_mask, True, True)
        val_loader = make_loader(args, data, data.author_val_mask, False, False)
        test_loader = make_loader(args, data, data.author_test_mask, False, False)

    best_val, best_state, times = -1.0, None, []
    for epoch in range(1, args.epochs + 1):
        start = time()
        if args.loader == "full":
            loss, grads = step(model, data, masks[0])
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)
            loss = loss.item()
        else:
            losses = []
            for batch in train_loader:
                batch_loss, grads = step(model, batch, batch.batch_size)
                optimizer.update(model, grads)
                mx.eval(model.parameters(), optimizer.state)
                losses.append(batch_loss.item())
            loss = sum(losses) / len(losses)
        times.append(time() - start)

        if args.loader == "full":
            train_acc, val_acc, _ = full_batch_accuracy(model, data, masks)
        else:
            train_acc, val_acc = float("nan"), loader_accuracy(model, val_loader)
        if val_acc > best_val:
            best_val = val_acc
            best_state = [(k, mx.array(v)) for k, v in tree_flatten(model.parameters())]
        if epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:3d} | loss {loss:.4f} | train {train_acc:.4f} | "
                f"val {val_acc:.4f} | {times[-1] * 1000:.0f} ms"
            )

    # Test with the parameters of the best validation epoch
    model.load_weights(best_state)
    if args.loader == "full":
        test_acc = full_batch_accuracy(model, data, masks)[2]
    else:
        test_acc = loader_accuracy(model, test_loader)
    epoch_time = float(np.median(times[1:] if len(times) > 1 else times))
    print(f"Best val acc: {best_val:.4f} | test acc: {test_acc:.4f}")
    print(f"Median epoch time: {epoch_time * 1000:.0f} ms")
    return {"val_acc": best_val, "test_acc": test_acc, "epoch_time": epoch_time}


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=["sage", "gat", "skip"], default="sage")
    parser.add_argument("--loader", choices=["full", "neighbor", "hgt"], default="full")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_neighbors", type=str, default="10,10")
    parser.add_argument(
        "--num_samples",
        type=str,
        default="256,256",
        help="HGTLoader: nodes sampled per node type at each hop",
    )
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    return parser


if __name__ == "__main__":
    main(get_parser().parse_args())
