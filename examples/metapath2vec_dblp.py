"""MetaPath2Vec on DBLP: unsupervised author embeddings.

DBLP has four node types (author, paper, term, conference). MetaPath2Vec
learns embeddings from random walks that follow the meta-path

    author -> paper -> conference -> paper -> author    (APCPA)

so two authors end up close in embedding space when they publish at the same
conferences, even if they never wrote a paper together. The walks are
sampled by ``mlx_cluster.metapath_random_walk`` (CPU or Metal).

Training uses no labels. To check what the embeddings capture, a logistic
regression is then fit on the (frozen) author embeddings of the training
split to predict each author's research area, and evaluated on the
validation and test splits.

Run::

    python examples/metapath2vec_dblp.py
    python examples/metapath2vec_dblp.py --epochs 10 --embedding_dim 64 --cpu
"""

from argparse import ArgumentParser
from time import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from mlx_graphs.algorithms import MetaPath2Vec
from mlx_graphs.data import HeteroGraphData
from mlx_graphs.datasets import DBLP

METAPATH = [
    ("author", "to", "paper"),
    ("paper", "to", "conference"),
    ("conference", "to", "paper"),
    ("paper", "to", "author"),
]


def mask_to_index(mask: mx.array) -> mx.array:
    return mx.array(np.flatnonzero(np.asarray(mask)), dtype=mx.int64)


def train_embeddings(model: MetaPath2Vec, args) -> None:
    optimizer = optim.Adam(learning_rate=args.lr)
    loss_and_grad_fn = nn.value_and_grad(model, model.loss)

    for epoch in range(args.epochs):
        start = time()
        losses = []
        for step, (pos, neg) in enumerate(model.dataloader(args.batch_size)):
            loss, grads = loss_and_grad_fn(pos, neg)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            losses.append(loss.item())
            if (step + 1) % args.log_every == 0:
                print(
                    f"Epoch: {epoch:2d} | Step: {step + 1:4d} | "
                    f"Loss: {sum(losses[-args.log_every :]) / args.log_every:.4f}"
                )
        print(
            f"Epoch: {epoch:2d} | Mean loss: {sum(losses) / len(losses):.4f} | "
            f"Time: {time() - start:.2f}s"
        )


def linear_probe(
    emb: mx.array,
    labels: mx.array,
    train_idx: mx.array,
    eval_idx: dict[str, mx.array],
    epochs: int = 300,
) -> dict[str, float]:
    """Logistic regression on frozen embeddings; returns accuracy per split."""
    # Standardize with training statistics; this makes the probe insensitive
    # to the overall scale of the embeddings.
    mean = emb[train_idx].mean(axis=0)
    std = emb[train_idx].std(axis=0) + 1e-6
    x = (emb - mean) / std

    num_classes = int(labels.max().item()) + 1
    clf = nn.Linear(x.shape[1], num_classes)
    optimizer = optim.Adam(learning_rate=0.01)

    def loss_fn(clf):
        logits = clf(x[train_idx])
        ce = nn.losses.cross_entropy(logits, labels[train_idx], reduction="mean")
        return ce + 1e-3 * (clf.weight**2).sum()

    loss_and_grad_fn = nn.value_and_grad(clf, loss_fn)
    for _ in range(epochs):
        _, grads = loss_and_grad_fn(clf)
        optimizer.update(clf, grads)
        mx.eval(clf.parameters(), optimizer.state)

    return {
        name: (clf(x[idx]).argmax(axis=-1) == labels[idx]).mean().item()
        for name, idx in eval_idx.items()
    }


def main(args, data: HeteroGraphData = None) -> float:
    mx.random.seed(args.seed)
    np.random.seed(args.seed)

    if data is None:
        data = DBLP(base_dir=args.data_dir)[0]
    print(f"Node counts: {data.num_nodes}")
    print(f"Metapath: {' -> '.join([METAPATH[0][0]] + [et[2] for et in METAPATH])}\n")

    model = MetaPath2Vec(
        data.edge_index_dict,
        embedding_dim=args.embedding_dim,
        metapath=METAPATH,
        walk_length=args.walk_length,
        context_size=args.context_size,
        walks_per_node=args.walks_per_node,
        num_negative_samples=args.num_negative_samples,
        num_nodes_dict=data.num_nodes,
        use_gpu=not args.cpu,
    )
    mx.eval(model.parameters())

    train_embeddings(model, args)

    emb = mx.stop_gradient(model("author"))
    labels = data.node_labels_dict["author"].astype(mx.int64)
    acc = linear_probe(
        emb,
        labels,
        train_idx=mask_to_index(data.author_train_mask),
        eval_idx={
            "train": mask_to_index(data.author_train_mask),
            "val": mask_to_index(data.author_val_mask),
            "test": mask_to_index(data.author_test_mask),
        },
    )
    print(
        f"\nLinear probe accuracy | train: {acc['train']:.3f} | "
        f"val: {acc['val']:.3f} | test: {acc['test']:.3f}"
    )
    return acc["test"]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--walk_length", type=int, default=50)
    parser.add_argument("--context_size", type=int, default=7)
    parser.add_argument("--walks_per_node", type=int, default=5)
    parser.add_argument("--num_negative_samples", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--cpu", action="store_true", help="Sample walks on CPU")
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
