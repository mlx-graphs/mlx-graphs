"""Runs ONE training configuration and prints its result as a JSON line.

Called by ``graph_quality.py``, one subprocess per configuration (under a
memory watchdog):

    python graph_worker.py --task tu --model gine --dataset MUTAG \
        --framework mlx --seed 0

Frameworks: ``mlx`` (mlx-graphs on the GPU) and ``pyg`` (PyTorch Geometric
on the CPU). Both train the same architecture with the same hyperparameters
on the same arrays (``graph_data.py``); MLX's Adam uses
``bias_correction=True`` (the update rule of ``torch.optim.Adam``), and an
L2 ``weight_decay`` is added to the MLX gradients as ``torch.optim.Adam``
does.
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import graph_data  # noqa: E402

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class Selector:
    """Keeps the test metrics of the epoch with the best validation metric
    (ties broken by the lower validation loss, then the earlier epoch)."""

    def __init__(self):
        self.best = None
        self.curve = []

    def add(self, epoch, val, test, val_loss=0.0, **extra):
        self.curve.append({"epoch": epoch, "val": val, "test": test, **extra})
        key = (val, -val_loss)
        if self.best is None or key > self.best[0]:
            self.best = (key, epoch, val, test, extra)

    def summary(self):
        _, epoch, val, test, extra = self.best
        return {
            "best_epoch": epoch,
            "val": val,
            "test": test,
            **{f"test_{k}": v for k, v in extra.items()},
            "curve": self.curve,
        }


def mlx_adam(lr, weight_decay=0.0):
    """MLX Adam with torch's update rule; returns (optimizer, grad_fn) where
    grad_fn adds torch-style L2 weight decay to the gradients."""
    import mlx.optimizers as optim
    from mlx.utils import tree_map

    optimizer = optim.Adam(learning_rate=lr, bias_correction=True)

    def decay(grads, params):
        if not weight_decay:
            return grads
        return tree_map(lambda g, p: g + weight_decay * p, grads, params)

    return optimizer, decay


def seed_all(framework, seed):
    np.random.seed(seed)
    if framework == "mlx":
        import mlx.core as mx

        mx.random.seed(seed)
    else:
        import torch

        torch.manual_seed(seed)
        torch.set_num_threads(2)


# ---------------------------------------------------------------------------
# Graph classification on TU datasets
# ---------------------------------------------------------------------------


def run_tu(args):
    import graph_tu

    data = graph_data.tu_dataset(args.dataset)
    graphs = graph_data.tu_graphs(data)
    # same split for both frameworks (depends on the seed only)
    splits = graph_data.split_indices(args.seed, len(graphs))
    in_dim = data["x"].shape[1]
    edge_dim = data["edge_attr"].shape[1] if "edge_attr" in data else None
    num_classes = int(data["y"].max()) + 1
    deg = graph_data.degree_histogram([graphs[i] for i in splits[0]])
    k = graph_tu.sort_k(graphs, splits[0])
    hp = {**graph_tu.DEFAULT_TRAIN_ARGS, **graph_tu.TRAIN_ARGS.get(args.model, {})}
    model_args = (args.model, in_dim, edge_dim, num_classes, deg, k)
    follow = args.model == "graphnorm"  # exercises follow_batch / drop_last

    if args.framework == "mlx":
        import mlx.core as mx
        import mlx.nn as nn

        from mlx_graphs.data import GraphData
        from mlx_graphs.loaders import Dataloader

        def to_graph(g):
            return GraphData(
                edge_index=mx.array(g["edge_index"].astype(np.int32)),
                node_features=mx.array(g["x"]),
                edge_features=mx.array(g["edge_attr"]) if "edge_attr" in g else None,
                graph_labels=mx.array(g["y"].astype(np.int32)),
            )

        mgraphs = [to_graph(g) for g in graphs]
        sets = [[mgraphs[i] for i in s] for s in splits]
        extra = dict(follow_batch=["node_features"]) if follow else {}
        train_loader = Dataloader(
            sets[0],
            batch_size=hp["batch_size"],
            shuffle=True,
            drop_last=follow,
            **extra,
        )
        eval_loaders = [Dataloader(s, batch_size=128, **extra) for s in sets[1:]]
        model = graph_tu.mlx_model(*model_args)
        optimizer, _ = mlx_adam(hp["lr"])

        def loss_fn(model, b):
            return nn.losses.cross_entropy(
                model(b), b.graph_labels.reshape(-1), reduction="mean"
            )

        loss_and_grad = nn.value_and_grad(model, loss_fn)

        def train_epoch():
            model.train()
            total, count = 0.0, 0
            for b in train_loader:
                loss, grads = loss_and_grad(model, b)
                optimizer.update(model, grads)
                mx.eval(loss, model.parameters(), optimizer.state)
                total += loss.item() * b.num_graphs
                count += b.num_graphs
            return total / count

        def evaluate(loader):
            model.eval()
            correct, loss, total = 0, 0.0, 0
            for b in loader:
                logits = model(b)
                y = b.graph_labels.reshape(-1)
                correct += (mx.argmax(logits, axis=1) == y).sum().item()
                loss += nn.losses.cross_entropy(logits, y, reduction="sum").item()
                total += b.num_graphs
            return correct / total, loss / total

    else:
        import torch
        import torch.nn.functional as F
        from torch_geometric.data import Data
        from torch_geometric.loader import DataLoader

        def to_data(g):
            d = Data(
                x=torch.from_numpy(g["x"]),
                edge_index=torch.from_numpy(g["edge_index"]),
                y=torch.from_numpy(g["y"]),
            )
            if "edge_attr" in g:
                d.edge_attr = torch.from_numpy(g["edge_attr"])
            return d

        pgraphs = [to_data(g) for g in graphs]
        sets = [[pgraphs[i] for i in s] for s in splits]
        extra = dict(follow_batch=["x"]) if follow else {}
        train_loader = DataLoader(
            sets[0],
            batch_size=hp["batch_size"],
            shuffle=True,
            drop_last=follow,
            **extra,
        )
        eval_loaders = [DataLoader(s, batch_size=128, **extra) for s in sets[1:]]
        model = graph_tu.pyg_model(*model_args)
        optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"])

        def train_epoch():
            model.train()
            total, count = 0.0, 0
            for b in train_loader:
                optimizer.zero_grad()
                loss = F.cross_entropy(model(b), b.y)
                loss.backward()
                optimizer.step()
                total += loss.item() * b.num_graphs
                count += b.num_graphs
            return total / count

        @torch.no_grad()
        def evaluate(loader):
            model.eval()
            correct, loss, total = 0, 0.0, 0
            for b in loader:
                logits = model(b)
                correct += (logits.argmax(1) == b.y).sum().item()
                loss += F.cross_entropy(logits, b.y, reduction="sum").item()
                total += b.num_graphs
            return correct / total, loss / total

    selector, losses = Selector(), []
    for epoch in range(1, args.epochs + 1):
        losses.append(train_epoch())
        val_acc, val_loss = evaluate(eval_loaders[0])
        test_acc, _ = evaluate(eval_loaders[1])
        selector.add(epoch, val_acc, test_acc, val_loss)
    return {"metric": "accuracy", "loss_curve": losses, **selector.summary()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--framework", choices=["mlx", "pyg"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    seed_all(args.framework, args.seed)
    if args.framework == "mlx":
        import mlx.core as mx

        mx.set_cache_limit(1024**3)
    if args.task == "tu":
        run = run_tu
    else:
        import graph_tasks

        run = getattr(graph_tasks, f"run_{args.task}")
    start = time.perf_counter()
    result = run(args)
    result["seconds"] = time.perf_counter() - start
    print(json.dumps({**vars(args), **result}))


if __name__ == "__main__":
    main()
