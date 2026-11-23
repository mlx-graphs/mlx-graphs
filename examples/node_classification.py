"""Full-batch node classification with the ready-made models of mlx-graphs.

    python examples/node_classification.py --dataset cora --model gcn
    python examples/node_classification.py --dataset amazon-computers --model gat
    python examples/node_classification.py --dataset coauthor-cs --model sage

Everything runs on the Mac's GPU with MLX; no PyTorch needed. Datasets
without a public split (Amazon, Coauthor) get a random one with 20 training
nodes per class, as in Shchur et al., "Pitfalls of Graph Neural Network
Evaluation".
"""

from argparse import ArgumentParser
from time import perf_counter

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from mlx_graphs.datasets import AmazonDataset, CoauthorDataset, PlanetoidDataset
from mlx_graphs.nn import GAT, GCN, GraphSAGE
from mlx_graphs.transforms import RandomNodeSplit

DATASETS = {
    "cora": lambda: PlanetoidDataset("cora"),
    "citeseer": lambda: PlanetoidDataset("citeseer"),
    "pubmed": lambda: PlanetoidDataset("pubmed"),
    "amazon-computers": lambda: AmazonDataset("computers"),
    "amazon-photo": lambda: AmazonDataset("photo"),
    "coauthor-cs": lambda: CoauthorDataset("cs"),
    "coauthor-physics": lambda: CoauthorDataset("physics"),
}


def mask_to_index(mask: mx.array) -> mx.array:
    """Indices of the ``True`` entries of a boolean mask."""
    return mx.array(np.nonzero(np.array(mask))[0])


def build_model(name: str, in_dim: int, hidden: int, num_classes: int, dropout: float):
    if name == "gcn":
        return GCN(in_dim, hidden, 2, out_features_dim=num_classes, dropout=dropout)
    if name == "sage":
        return GraphSAGE(
            in_dim, hidden, 2, out_features_dim=num_classes, dropout=dropout
        )
    if name == "gat":
        return GAT(
            in_dim, hidden, 2, out_features_dim=num_classes, heads=8, dropout=dropout
        )
    raise ValueError(f"Unknown model {name}")


def main(args):
    mx.random.seed(args.seed)
    graph = DATASETS[args.dataset]()[0]
    if getattr(graph, "train_mask", None) is None:
        graph = RandomNodeSplit(
            split="random", num_train_per_class=20, num_val=500, num_test=1000
        )(graph)

    x, edge_index = graph.node_features, graph.edge_index
    labels = graph.node_labels.reshape(-1)
    # MLX indexes with integer arrays, not boolean masks
    train, val, test = (
        mask_to_index(getattr(graph, f"{split}_mask"))
        for split in ("train", "val", "test")
    )
    num_classes = int(labels.max().item()) + 1

    model = build_model(args.model, x.shape[1], args.hidden, num_classes, args.dropout)
    mx.eval(model.parameters())
    # bias_correction=True gives the same update rule as torch.optim.Adam; MLX's
    # default (no correction) takes much larger first steps.
    optimizer = optim.AdamW(
        learning_rate=args.lr, weight_decay=args.weight_decay, bias_correction=True
    )

    def loss_fn(model):
        logits = model(edge_index, x)
        return nn.losses.cross_entropy(logits[train], labels[train], reduction="mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    def accuracy(index):
        pred = mx.argmax(model(edge_index, x)[index], axis=1)
        return (pred == labels[index]).astype(mx.float32).mean().item()

    best_val, test_at_best, times = 0.0, 0.0, []
    for epoch in range(1, args.epochs + 1):
        start = perf_counter()
        model.train()
        loss, grads = loss_and_grad(model)
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters(), optimizer.state)
        times.append(perf_counter() - start)

        if epoch % 10 == 0 or epoch == args.epochs:
            model.eval()
            val_acc, test_acc = accuracy(val), accuracy(test)
            if val_acc > best_val:
                best_val, test_at_best = val_acc, test_acc
            print(
                f"epoch {epoch:4d} | loss {loss.item():.4f} | "
                f"val {val_acc:.3f} | test {test_acc:.3f}"
            )

    steady = sorted(times[5:]) or times  # skip warm-up epochs
    median_ms = 1e3 * steady[len(steady) // 2]
    print(
        f"\n{args.model} on {args.dataset}: test accuracy {test_at_best:.3f} "
        f"(best val {best_val:.3f}), median epoch {median_ms:.2f} ms"
    )


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=list(DATASETS), default="cora")
    parser.add_argument("--model", choices=["gcn", "sage", "gat"], default="gcn")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
