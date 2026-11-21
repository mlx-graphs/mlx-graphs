"""Link prediction with a GCN encoder and a dot-product decoder.

    python examples/link_prediction.py --dataset cora
    python examples/link_prediction.py --dataset amazon-photo --epochs 300

The graph's edges are split into training, validation and test edges with
:class:`~mlx_graphs.transforms.RandomLinkSplit`. The model only passes
messages along training edges; it learns to score an edge ``(u, v)`` as the
dot product of the two node embeddings, against negative edges drawn fresh
at every epoch with :func:`~mlx_graphs.utils.negative_sampling`.
"""

from argparse import ArgumentParser

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from mlx_graphs.datasets import AmazonDataset, CoauthorDataset, PlanetoidDataset
from mlx_graphs.nn import GCN
from mlx_graphs.transforms import RandomLinkSplit
from mlx_graphs.utils import negative_sampling

DATASETS = {
    "cora": lambda: PlanetoidDataset("cora"),
    "citeseer": lambda: PlanetoidDataset("citeseer"),
    "pubmed": lambda: PlanetoidDataset("pubmed"),
    "amazon-photo": lambda: AmazonDataset("photo"),
    "coauthor-cs": lambda: CoauthorDataset("cs"),
}


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve (probability that a positive edge is scored
    above a negative one), from the ranks of the scores."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    num_pos = labels.sum()
    num_neg = len(labels) - num_pos
    return float(
        (ranks[labels == 1].sum() - num_pos * (num_pos + 1) / 2) / (num_pos * num_neg)
    )


class LinkPredictor(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.encoder = GCN(in_dim, hidden, 2, out_features_dim=out_dim, dropout=0.5)

    def __call__(self, edge_index, x, edge_label_index):
        z = self.encoder(edge_index, x)
        src, dst = edge_label_index
        return (z[src] * z[dst]).sum(axis=-1)  # one score (logit) per edge


def main(args):
    mx.random.seed(args.seed)
    graph = DATASETS[args.dataset]()[0]
    num_nodes = graph.num_nodes
    train, val, test = RandomLinkSplit(
        num_val=0.05, num_test=0.1, is_undirected=True, add_negative_train_samples=False
    )(graph)
    print(
        f"{args.dataset}: {num_nodes} nodes | message-passing edges: "
        f"{train.edge_index.shape[1]} | supervision edges: "
        f"train {train.edge_label_index.shape[1]}, "
        f"val {val.edge_label_index.shape[1]}, test {test.edge_label_index.shape[1]}"
    )

    model = LinkPredictor(graph.num_node_features, args.hidden, args.out)
    mx.eval(model.parameters())
    optimizer = optim.Adam(learning_rate=args.lr, bias_correction=True)

    def loss_fn(model, label_index, labels):
        logits = model(train.edge_index, train.node_features, label_index)
        return nn.losses.binary_cross_entropy(
            logits, labels, with_logits=True, reduction="mean"
        )

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    def evaluate(split):
        model.eval()
        logits = model(train.edge_index, train.node_features, split.edge_label_index)
        return roc_auc(np.array(logits), np.array(split.edge_label))

    positives = train.edge_label_index
    best_val, test_at_best = 0.0, 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        # new negative edges at every epoch, one per positive edge
        negatives = negative_sampling(
            train.edge_index, num_nodes=num_nodes, num_neg_samples=positives.shape[1]
        )
        label_index = mx.concatenate([positives, negatives.astype(positives.dtype)], 1)
        labels = mx.concatenate(
            [mx.ones(positives.shape[1]), mx.zeros(negatives.shape[1])]
        )
        loss, grads = loss_and_grad(model, label_index, labels)
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters(), optimizer.state)

        if epoch % 10 == 0:
            val_auc, test_auc = evaluate(val), evaluate(test)
            if val_auc > best_val:
                best_val, test_at_best = val_auc, test_auc
            print(
                f"epoch {epoch:4d} | loss {loss.item():.4f} | "
                f"val AUC {val_auc:.4f} | test AUC {test_auc:.4f}"
            )

    print(f"\ntest ROC-AUC {test_at_best:.4f} (best validation {best_val:.4f})")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=list(DATASETS), default="cora")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--out", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
