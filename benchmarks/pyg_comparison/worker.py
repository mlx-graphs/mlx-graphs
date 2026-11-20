"""Runs ONE benchmark configuration and prints its result as a JSON line.

Called by ``run.py``, one subprocess per configuration, so that every
measurement starts from a fresh process (clean caches, exact peak memory) and
a configuration that runs out of memory cannot take the others down.

    python worker.py layer --framework mlx-graphs --layer gcn \
        --num-nodes 100000 --num-edges 1000000
    python worker.py node --framework pyg-mps --dataset cora --model gat
    python worker.py graph --framework mlx-graphs --dataset PROTEINS --model gin

Frameworks: ``mlx-graphs`` (GPU), ``pyg-mps`` (PyTorch Geometric on the MPS
GPU backend) and ``pyg-cpu``.
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import peak_footprint_gb, random_graph, time_fn  # noqa: E402

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".mlx_graphs_data"
)
LAYERS = ("gcn", "sage", "gin", "gat", "gatv2")
HEADS = 4


# ---------------------------------------------------------------------------
# Models. The same architectures are built in both frameworks.
# GAT / GATv2 run on a graph whose self-loops were added once beforehand, for
# both frameworks (PyG adds them inside the layer by default, mlx-graphs does
# not). GCN adds its self-loops in the layer in both.
# ---------------------------------------------------------------------------


def mlx_conv(name: str, in_dim: int, out_dim: int, heads: int = HEADS):
    import mlx.nn as nn

    import mlx_graphs.nn as gnn

    if name == "gcn":
        return gnn.GCNConv(in_dim, out_dim, add_self_loops=True)
    if name == "sage":
        return gnn.SAGEConv(in_dim, out_dim)
    if name == "gin":
        return gnn.GINConv(
            nn.Sequential(
                nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)
            )
        )
    if name == "gat":
        return gnn.GATConv(in_dim, out_dim // heads, heads=heads)
    if name == "gatv2":
        return gnn.GATv2Conv(in_dim, out_dim // heads, heads=heads)
    raise ValueError(name)


def pyg_conv(
    name: str, in_dim: int, out_dim: int, heads: int = HEADS, fixed_graph=False
):
    import torch
    import torch_geometric.nn as pnn

    if name == "gcn":
        # On a graph that never changes, cached=True is PyG's own option to
        # compute the normalization once, which mlx-graphs does automatically.
        return pnn.GCNConv(in_dim, out_dim, add_self_loops=True, cached=fixed_graph)
    if name == "sage":
        return pnn.SAGEConv(in_dim, out_dim)
    if name == "gin":
        return pnn.GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(in_dim, out_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(out_dim, out_dim),
            )
        )
    if name == "gat":
        return pnn.GATConv(in_dim, out_dim // heads, heads=heads, add_self_loops=False)
    if name == "gatv2":
        return pnn.GATv2Conv(
            in_dim, out_dim // heads, heads=heads, add_self_loops=False
        )
    raise ValueError(name)


def build_mlx_model(name, in_dim, hidden, out_dim, num_layers, dropout, pool=False):
    import mlx.core as mx
    import mlx.nn as nn

    import mlx_graphs.nn as gnn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            dims = [in_dim] + [hidden] * (num_layers - 1)
            self.convs = [mlx_conv(name, a, b) for a, b in zip(dims[:-1], dims[1:])]
            if pool:  # graph classification: conv stack -> pooling -> linear
                self.convs.append(mlx_conv(name, dims[-1], hidden))
                self.lin = gnn.Linear(hidden, out_dim)
            else:  # node classification: the last conv maps to the classes
                self.convs.append(mlx_conv(name, dims[-1], out_dim, heads=1))
            self.dropout = nn.Dropout(dropout)

        def __call__(self, edge_index, x, batch_indices=None):
            for conv in self.convs[:-1]:
                x = self.dropout(nn.relu(conv(edge_index, x)))
            x = self.convs[-1](edge_index, x)
            if pool:
                x = gnn.global_mean_pool(nn.relu(x), batch_indices)
                x = self.lin(self.dropout(x))
            return x

    model = Model()
    mx.eval(model.parameters())
    return model


def build_pyg_model(name, in_dim, hidden, out_dim, num_layers, dropout, pool=False):
    import torch
    import torch.nn.functional as F
    import torch_geometric.nn as pnn

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            dims = [in_dim] + [hidden] * (num_layers - 1)
            fixed = not pool  # full-batch node classification: one graph
            convs = [
                pyg_conv(name, a, b, fixed_graph=fixed)
                for a, b in zip(dims[:-1], dims[1:])
            ]
            if pool:
                convs.append(pyg_conv(name, dims[-1], hidden))
                self.lin = torch.nn.Linear(hidden, out_dim)
            else:
                convs.append(pyg_conv(name, dims[-1], out_dim, 1, fixed_graph=True))
            self.convs = torch.nn.ModuleList(convs)
            self.p = dropout

        def forward(self, x, edge_index, batch=None):
            for conv in self.convs[:-1]:
                x = F.dropout(F.relu(conv(x, edge_index)), self.p, self.training)
            x = self.convs[-1](x, edge_index)
            if pool:
                x = pnn.global_mean_pool(F.relu(x), batch)
                x = self.lin(F.dropout(x, self.p, self.training))
            return x

    return Model()


def torch_device(framework: str):
    import torch

    device = torch.device("mps" if framework == "pyg-mps" else "cpu")
    sync = torch.mps.synchronize if framework == "pyg-mps" else (lambda: None)
    return device, sync


# ---------------------------------------------------------------------------
# Task 1: one layer, forward and forward + backward, on a random graph
# ---------------------------------------------------------------------------


def run_layer(args) -> dict:
    edge_index = random_graph(args.num_nodes, args.num_edges)
    x = np.random.default_rng(1).standard_normal(
        (args.num_nodes, args.dim), dtype=np.float32
    )

    if args.framework == "mlx-graphs":
        import mlx.core as mx
        import mlx.nn as nn

        ei, feats = mx.array(edge_index.astype(np.int32)), mx.array(x)
        mx.eval(ei, feats)
        layer = mlx_conv(args.layer, args.dim, args.dim)
        mx.eval(layer.parameters())
        loss_and_grad = nn.value_and_grad(layer, lambda m: (m(ei, feats) ** 2).mean())

        def forward():
            mx.eval(layer(ei, feats))

        def forward_backward():
            mx.eval(loss_and_grad(layer))

    else:
        import torch

        device, sync = torch_device(args.framework)
        ei = torch.from_numpy(edge_index).to(device)
        feats = torch.from_numpy(x).to(device)
        layer = pyg_conv(args.layer, args.dim, args.dim).to(device)

        def forward():
            with torch.no_grad():
                layer(feats, ei)
            sync()

        def forward_backward():
            layer.zero_grad()
            (layer(feats, ei) ** 2).mean().backward()
            sync()

    fwd = time_fn(forward, args.repeats, args.warmup)
    both = time_fn(forward_backward, args.repeats, args.warmup)
    return {
        "forward_ms": fwd["ms"],
        "forward_backward_ms": both["ms"],
        "forward_backward_ms_q1": both["ms_q1"],
        "forward_backward_ms_q3": both["ms_q3"],
    }


# ---------------------------------------------------------------------------
# Task 2: full-batch node classification
# ---------------------------------------------------------------------------


def load_node_dataset(name: str, seed: int) -> dict:
    """Loads a node classification graph as numpy arrays, through mlx-graphs'
    datasets, so that both frameworks train on exactly the same data."""
    import contextlib
    import io

    import mlx_graphs.datasets as datasets
    from mlx_graphs.utils import to_undirected

    with contextlib.redirect_stdout(io.StringIO()):
        if name in ("cora", "citeseer", "pubmed"):
            graph = datasets.PlanetoidDataset(name, base_dir=DATA_DIR)[0]
        elif name.startswith("amazon-"):
            graph = datasets.AmazonDataset(name.split("-")[1], base_dir=DATA_DIR)[0]
        elif name.startswith("coauthor-"):
            graph = datasets.CoauthorDataset(name.split("-")[1], base_dir=DATA_DIR)[0]
        elif name == "ogbn-arxiv":
            graph = datasets.OGBDataset(name, base_dir=DATA_DIR)[0]
            graph.edge_index = to_undirected(
                graph.edge_index, coalesce=True, num_nodes=graph.num_nodes
            )
        else:
            raise ValueError(f"unknown dataset {name}")

    labels = np.array(graph.node_labels).reshape(-1).astype(np.int64)
    num_nodes = labels.shape[0]

    if getattr(graph, "train_mask", None) is not None:
        train, val, test = (
            np.nonzero(np.array(getattr(graph, f"{split}_mask")))[0]
            for split in ("train", "val", "test")
        )
    else:
        # Amazon / Coauthor have no public split: 20 train and 30 validation
        # nodes per class, the rest for testing (Shchur et al., 2018).
        rng = np.random.default_rng(seed)
        train, val = [], []
        for c in np.unique(labels):
            nodes = rng.permutation(np.nonzero(labels == c)[0])
            train.append(nodes[:20])
            val.append(nodes[20:50])
        train, val = np.concatenate(train), np.concatenate(val)
        test = np.setdiff1d(np.arange(num_nodes), np.concatenate([train, val]))

    return {
        "x": np.array(graph.node_features).astype(np.float32),
        "edge_index": np.array(graph.edge_index).astype(np.int64),
        "y": labels,
        "train": train,
        "val": val,
        "test": test,
        "num_classes": int(labels.max()) + 1,
    }


def with_self_loops(edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
    edge_index = edge_index[:, edge_index[0] != edge_index[1]]
    loops = np.arange(num_nodes, dtype=edge_index.dtype)
    return np.concatenate([edge_index, np.stack([loops, loops])], axis=1)


def run_node(args) -> dict:
    data = load_node_dataset(args.dataset, args.seed)
    num_nodes = data["x"].shape[0]
    edge_index = data["edge_index"]
    if args.model in ("gat", "gatv2"):
        edge_index = with_self_loops(edge_index, num_nodes)
    model_args = (
        args.model,
        data["x"].shape[1],
        args.hidden,
        data["num_classes"],
        args.num_layers,
        args.dropout,
    )
    step_times, best_val, test_at_best = [], -1.0, 0.0
    start = time.perf_counter()

    if args.framework == "mlx-graphs":
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim

        mx.random.seed(args.seed)
        x, y = mx.array(data["x"]), mx.array(data["y"])
        ei = mx.array(edge_index.astype(np.int32))
        idx = {k: mx.array(data[k]) for k in ("train", "val", "test")}
        mx.eval(x, ei, y, idx)
        model = build_mlx_model(*model_args)
        # bias_correction=True: same update rule as torch.optim.Adam
        optimizer = optim.Adam(learning_rate=args.lr, bias_correction=True)

        def loss_fn(model):
            logits = model(ei, x)
            return nn.losses.cross_entropy(
                logits[idx["train"]], y[idx["train"]], reduction="mean"
            )

        loss_and_grad = nn.value_and_grad(model, loss_fn)

        def train_step():
            model.train()
            loss, grads = loss_and_grad(model)
            optimizer.update(model, grads)
            mx.eval(loss, model.parameters(), optimizer.state)

        def evaluate():
            model.eval()
            pred = mx.argmax(model(ei, x), axis=1)
            return [
                (pred[idx[k]] == y[idx[k]]).astype(mx.float32).mean().item()
                for k in ("val", "test")
            ]

    else:
        import torch
        import torch.nn.functional as F

        torch.manual_seed(args.seed)
        device, sync = torch_device(args.framework)
        x = torch.from_numpy(data["x"]).to(device)
        ei = torch.from_numpy(edge_index).to(device)
        y = torch.from_numpy(data["y"]).to(device)
        idx = {
            k: torch.from_numpy(data[k]).to(device) for k in ("train", "val", "test")
        }
        model = build_pyg_model(*model_args).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        def train_step():
            model.train()
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x, ei)[idx["train"]], y[idx["train"]])
            loss.backward()
            optimizer.step()
            sync()

        @torch.no_grad()
        def evaluate():
            model.eval()
            pred = model(x, ei).argmax(dim=1)
            return [
                (pred[idx[k]] == y[idx[k]]).float().mean().item()
                for k in ("val", "test")
            ]

    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        train_step()
        step_times.append((time.perf_counter() - t0) * 1e3)
        if epoch % args.eval_every == args.eval_every - 1 or epoch == args.epochs - 1:
            val_acc, test_acc = evaluate()
            if val_acc > best_val:
                best_val, test_at_best = val_acc, test_acc

    timed = step_times[args.warmup :]
    q1, median, q3 = np.percentile(timed, [25, 50, 75])
    return {
        "num_nodes": int(num_nodes),
        "num_edges": int(edge_index.shape[1]),
        "epoch_ms": float(median),
        "epoch_ms_q1": float(q1),
        "epoch_ms_q3": float(q3),
        "total_s": time.perf_counter() - start,
        "val_acc": float(best_val),
        "test_acc": float(test_at_best),
    }


# ---------------------------------------------------------------------------
# Task 3: graph classification with mini-batches of small graphs.
# The epoch time includes loading and batching the graphs.
# ---------------------------------------------------------------------------


def run_graph(args) -> dict:
    import contextlib
    import io

    rng = np.random.default_rng(args.seed)

    if args.framework == "mlx-graphs":
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim

        from mlx_graphs.datasets import TUDataset
        from mlx_graphs.loaders import Dataloader

        mx.random.seed(args.seed)
        with contextlib.redirect_stdout(io.StringIO()):
            dataset = TUDataset(args.dataset, base_dir=DATA_DIR)
        perm = rng.permutation(len(dataset)).tolist()
        split = int(0.8 * len(dataset))
        train_set, test_set = dataset[perm[:split]], dataset[perm[split:]]
        train_loader = Dataloader(train_set, batch_size=args.batch_size, shuffle=True)
        test_loader = Dataloader(test_set, batch_size=args.batch_size)

        model = build_mlx_model(
            args.model,
            dataset.num_node_features,
            args.hidden,
            dataset.num_graph_classes,
            args.num_layers,
            args.dropout,
            pool=True,
        )
        # bias_correction=True: same update rule as torch.optim.Adam
        optimizer = optim.Adam(learning_rate=args.lr, bias_correction=True)

        def loss_fn(model, batch):
            logits = model(batch.edge_index, batch.node_features, batch.batch_indices)
            return nn.losses.cross_entropy(
                logits, batch.graph_labels.reshape(-1), reduction="mean"
            )

        loss_and_grad = nn.value_and_grad(model, loss_fn)

        def train_epoch():
            model.train()
            for batch in train_loader:
                loss, grads = loss_and_grad(model, batch)
                optimizer.update(model, grads)
                mx.eval(loss, model.parameters(), optimizer.state)

        def evaluate():
            model.eval()
            correct = 0
            for batch in test_loader:
                logits = model(
                    batch.edge_index, batch.node_features, batch.batch_indices
                )
                labels = batch.graph_labels.reshape(-1)
                correct += (mx.argmax(logits, axis=1) == labels).sum().item()
            return correct / len(test_set)

        num_graphs = len(dataset)

    else:
        import torch
        import torch.nn.functional as F
        from torch_geometric.datasets import TUDataset
        from torch_geometric.loader import DataLoader

        torch.manual_seed(args.seed)
        device, sync = torch_device(args.framework)
        with contextlib.redirect_stderr(io.StringIO()):
            dataset = TUDataset(os.path.join(DATA_DIR, "pyg"), args.dataset)
        perm = rng.permutation(len(dataset)).tolist()
        split = int(0.8 * len(dataset))
        train_set, test_set = dataset[perm[:split]], dataset[perm[split:]]
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=args.batch_size)

        model = build_pyg_model(
            args.model,
            dataset.num_node_features,
            args.hidden,
            dataset.num_classes,
            args.num_layers,
            args.dropout,
            pool=True,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        def train_epoch():
            model.train()
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                logits = model(batch.x, batch.edge_index, batch.batch)
                F.cross_entropy(logits, batch.y).backward()
                optimizer.step()
            sync()

        @torch.no_grad()
        def evaluate():
            model.eval()
            correct = 0
            for batch in test_loader:
                batch = batch.to(device)
                logits = model(batch.x, batch.edge_index, batch.batch)
                correct += (logits.argmax(dim=1) == batch.y).sum().item()
            return correct / len(test_set)

        num_graphs = len(dataset)

    start = time.perf_counter()
    epoch_times = []
    for _ in range(args.epochs):
        t0 = time.perf_counter()
        train_epoch()
        epoch_times.append((time.perf_counter() - t0) * 1e3)
    total = time.perf_counter() - start
    timed = epoch_times[args.warmup :]
    q1, median, q3 = np.percentile(timed, [25, 50, 75])
    return {
        "num_graphs": int(num_graphs),
        "epoch_ms": float(median),
        "epoch_ms_q1": float(q1),
        "epoch_ms_q3": float(q3),
        "total_s": total,
        "test_acc": float(evaluate()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="task", required=True)

    def common_args(p):
        p.add_argument("--framework", required=True)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--warmup", type=int, default=5)

    p = sub.add_parser("layer")
    common_args(p)
    p.add_argument("--layer", choices=LAYERS, required=True)
    p.add_argument("--num-nodes", type=int, required=True)
    p.add_argument("--num-edges", type=int, required=True)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--repeats", type=int, default=20)

    p = sub.add_parser("node")
    common_args(p)
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", choices=LAYERS, required=True)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--eval-every", type=int, default=10)

    p = sub.add_parser("graph")
    common_args(p)
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", choices=LAYERS, required=True)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)

    args = parser.parse_args()
    if args.framework == "mlx-graphs":
        import mlx.core as mx

        # keep MLX from holding on to freed buffers without bound
        mx.set_cache_limit(4 * 1024**3)

    result = {"layer": run_layer, "node": run_node, "graph": run_graph}[args.task](args)
    result["peak_memory_gb"] = peak_footprint_gb()
    print(json.dumps({**vars(args), **result}))


if __name__ == "__main__":
    main()
