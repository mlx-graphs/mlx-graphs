"""Shared layer, node classification, and graph classification workloads."""

import os
import time

import numpy as np

from benchmarks.common import random_graph, time_fn
from benchmarks.versions import REPO_ROOT

# Datasets are shared by every checkout that sets MLX_GRAPHS_BENCH_DATA (e.g.
# worktrees of other versions), and downloaded only once.
DATA_DIR = os.environ.get("MLX_GRAPHS_BENCH_DATA") or os.path.join(
    REPO_ROOT, ".mlx_graphs_data"
)
LAYERS = ("gcn", "sage", "gin", "gat", "gatv2", "transformer")
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
        return gnn.GATConv(in_dim, out_dim // heads, heads=heads, add_self_loops=False)
    if name == "gatv2":
        return gnn.GATv2Conv(
            in_dim, out_dim // heads, heads=heads, add_self_loops=False
        )
    if name == "transformer":
        return gnn.TransformerConv(in_dim, out_dim // heads, heads=heads)
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
    if name == "transformer":
        return pnn.TransformerConv(in_dim, out_dim // heads, heads=heads)
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
    dataset = getattr(args, "dataset", None)
    if dataset is None:
        edge_index = random_graph(args.num_nodes, args.num_edges)
        num_nodes = args.num_nodes
    else:
        # A real graph's structure (hubs, locality) with random features of
        # size dim, so the layer cost does not depend on the dataset's features.
        data = load_node_dataset(dataset, seed=0)
        edge_index, num_nodes = data["edge_index"], data["y"].shape[0]
    x = np.random.default_rng(1).standard_normal(
        (num_nodes, args.dim), dtype=np.float32
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
        "num_nodes": num_nodes,
        "num_edges": int(edge_index.shape[1]),
        "forward_ms": fwd["ms"],
        "forward_backward_ms": both["ms"],
        "forward_backward_ms_q1": both["ms_q1"],
        "forward_backward_ms_q3": both["ms_q3"],
    }


# ---------------------------------------------------------------------------
# Task 2: full-batch node classification
# ---------------------------------------------------------------------------


def load_graph_dataset(name: str):
    """Loads a TU dataset through PyG and converts it to mlx-graphs graphs.

    Both frameworks train on exactly the same arrays. (mlx-graphs'
    ``TUDataset`` also keeps continuous node attributes, e.g. on PROTEINS,
    which PyG's ``TUDataset`` leaves out by default.) The timed epochs still
    include mlx-graphs' own batching of the graphs.

    Returns:
        The list of ``GraphData``, the number of node features and of classes.
    """
    import contextlib
    import io

    import mlx.core as mx
    from torch_geometric.datasets import TUDataset

    from mlx_graphs.data import GraphData

    with contextlib.redirect_stdout(io.StringIO()):
        with contextlib.redirect_stderr(io.StringIO()):
            dataset = TUDataset(os.path.join(DATA_DIR, "pyg"), name)
    graphs = [
        GraphData(
            edge_index=mx.array(data.edge_index.numpy().astype(np.int32)),
            node_features=mx.array(data.x.numpy()),
            graph_labels=mx.array(data.y.numpy().astype(np.int32)),
        )
        for data in dataset
    ]
    return graphs, dataset.num_node_features, dataset.num_classes


def load_node_dataset(name: str, seed: int) -> dict:
    """Loads a node classification graph as numpy arrays through PyG and OGB.

    Every framework, and every version of mlx-graphs, trains on exactly the
    same arrays, and the harness does not depend on which datasets the
    measured version of mlx-graphs has. (mlx-graphs' own loaders return the
    same arrays; its dataset tests check this against PyG.)
    """
    import contextlib
    import io

    root = os.path.join(DATA_DIR, "pyg")
    quiet = contextlib.ExitStack()
    quiet.enter_context(contextlib.redirect_stdout(io.StringIO()))
    quiet.enter_context(contextlib.redirect_stderr(io.StringIO()))
    masks = None
    with quiet:
        if name in ("cora", "citeseer", "pubmed"):
            from torch_geometric.datasets import Planetoid

            pyg_name = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}
            data = Planetoid(root, pyg_name[name])[0]
            masks = [
                data[f"{split}_mask"].numpy() for split in ("train", "val", "test")
            ]
        elif name.startswith(("amazon-", "coauthor-")):
            from torch_geometric.datasets import Amazon, Coauthor

            group, which = name.split("-")
            pyg_name = {"computers": "Computers", "photo": "Photo", "cs": "CS",
                        "physics": "Physics"}[which]  # fmt: skip
            data = (Amazon if group == "amazon" else Coauthor)(root, pyg_name)[0]
        elif name == "ogbn-arxiv":
            import torch
            from ogb.nodeproppred import NodePropPredDataset

            # OGB caches the processed graph (numpy arrays) with torch.save
            # and reads it back with torch.load, which refuses non-tensor
            # objects since PyTorch 2.6 (weights_only=True by default). The
            # file is OGB's own cache of the official dataset.
            load = torch.load
            torch.load = lambda *a, **k: load(*a, **{**k, "weights_only": False})
            try:
                dataset = NodePropPredDataset(
                    name, root=os.path.join(DATA_DIR, "ogbn-arxiv", "raw")
                )
            finally:
                torch.load = load
            graph, label = dataset[0]
            n = graph["num_nodes"]
            split = dataset.get_idx_split()
            masks = [
                np.isin(np.arange(n), split[s]) for s in ("train", "valid", "test")
            ]
            # undirected, without duplicate edges, sorted by (source, target)
            edges = graph["edge_index"].astype(np.int64)
            edges = np.concatenate([edges, edges[::-1]], axis=1)
            _, first = np.unique(edges[0] * n + edges[1], return_index=True)
            data = {
                "x": graph["node_feat"],
                "edge_index": edges[:, first],
                "y": label.reshape(-1),
            }
        else:
            raise ValueError(f"unknown dataset {name}")

    if not isinstance(data, dict):
        data = {
            "x": data.x.numpy(),
            "edge_index": data.edge_index.numpy(),
            "y": data.y.numpy(),
        }
    labels = np.asarray(data["y"]).reshape(-1).astype(np.int64)
    num_nodes = labels.shape[0]

    if masks is not None:
        train, val, test = (np.nonzero(mask)[0] for mask in masks)
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
        "x": np.asarray(data["x"]).astype(np.float32),
        "edge_index": np.asarray(data["edge_index"]).astype(np.int64),
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


class Curves:
    """Accuracy measured during training, and the model selected on it."""

    def __init__(self):
        self.epochs, self.train, self.val, self.test = [], [], [], []

    def add(self, epoch, train_acc, val_acc, test_acc):
        self.epochs.append(epoch)
        self.train.append(float(train_acc))
        self.val.append(float(val_acc))
        self.test.append(float(test_acc))

    def summary(self, losses):
        """Final metrics, taken at the evaluation with the best validation
        accuracy (the earliest one on ties), plus the full curves."""
        best = int(np.argmax(self.val))
        return {
            "best_epoch": self.epochs[best],
            "train_acc": self.train[best],
            "val_acc": self.val[best],
            "test_acc": self.test[best],
            "final_train_loss": float(np.mean(losses[-10:])),
            "loss_curve": [round(v, 5) for v in losses],
            "eval_epochs": self.epochs,
            "train_acc_curve": self.train,
            "val_acc_curve": self.val,
            "test_acc_curve": self.test,
        }


def split_indices(rng, size, train=0.8, val=0.1):
    """Random train / validation / test split of ``range(size)``."""
    perm = rng.permutation(size)
    n_train, n_val = int(train * size), int(val * size)
    return (
        perm[:n_train].tolist(),
        perm[n_train : n_train + n_val].tolist(),
        perm[n_train + n_val :].tolist(),
    )


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
    step_times, losses = [], []
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
            return loss

        def evaluate():
            model.eval()
            pred = mx.argmax(model(ei, x), axis=1)
            return [
                (pred[idx[k]] == y[idx[k]]).astype(mx.float32).mean().item()
                for k in ("train", "val", "test")
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
            return loss.detach()

        @torch.no_grad()
        def evaluate():
            model.eval()
            pred = model(x, ei).argmax(dim=1)
            return [
                (pred[idx[k]] == y[idx[k]]).float().mean().item()
                for k in ("train", "val", "test")
            ]

    curves = Curves()
    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        loss = train_step()
        step_times.append((time.perf_counter() - t0) * 1e3)
        # read outside the timed section: the step already synchronized
        losses.append(float(loss.item()))
        if epoch % args.eval_every == args.eval_every - 1 or epoch == args.epochs - 1:
            curves.add(epoch + 1, *evaluate())

    timed = step_times[args.warmup :]
    q1, median, q3 = np.percentile(timed, [25, 50, 75])
    return {
        "num_nodes": int(num_nodes),
        "num_edges": int(edge_index.shape[1]),
        "epoch_ms": float(median),
        "epoch_ms_q1": float(q1),
        "epoch_ms_q3": float(q3),
        "total_s": time.perf_counter() - start,
        **curves.summary(losses),
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

        from mlx_graphs.loaders import Dataloader

        mx.random.seed(args.seed)
        graphs, num_features, num_classes = load_graph_dataset(args.dataset)
        splits = split_indices(rng, len(graphs))
        train_set, val_set, test_set = ([graphs[i] for i in s] for s in splits)
        train_loader = Dataloader(train_set, batch_size=args.batch_size, shuffle=True)
        eval_loaders = [
            Dataloader(subset, batch_size=args.batch_size)
            for subset in (train_set, val_set, test_set)
        ]

        model = build_mlx_model(
            args.model,
            num_features,
            args.hidden,
            num_classes,
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
            batch_losses = []
            for batch in train_loader:
                loss, grads = loss_and_grad(model, batch)
                optimizer.update(model, grads)
                mx.eval(loss, model.parameters(), optimizer.state)
                batch_losses.append((loss, batch.num_graphs))
            return batch_losses

        def accuracy(loader):
            correct, total = 0, 0
            for batch in loader:
                logits = model(
                    batch.edge_index, batch.node_features, batch.batch_indices
                )
                labels = batch.graph_labels.reshape(-1)
                correct += (mx.argmax(logits, axis=1) == labels).sum().item()
                total += batch.num_graphs
            return correct / total

        def evaluate():
            model.eval()
            return [accuracy(loader) for loader in eval_loaders]

        num_graphs = len(graphs)

    else:
        import torch
        import torch.nn.functional as F
        from torch_geometric.datasets import TUDataset
        from torch_geometric.loader import DataLoader

        torch.manual_seed(args.seed)
        device, sync = torch_device(args.framework)
        with contextlib.redirect_stderr(io.StringIO()):
            dataset = TUDataset(os.path.join(DATA_DIR, "pyg"), args.dataset)
        splits = split_indices(rng, len(dataset))
        train_set, val_set, test_set = (dataset[split] for split in splits)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        eval_loaders = [
            DataLoader(subset, batch_size=args.batch_size)
            for subset in (train_set, val_set, test_set)
        ]

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
            batch_losses = []
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                logits = model(batch.x, batch.edge_index, batch.batch)
                loss = F.cross_entropy(logits, batch.y)
                loss.backward()
                optimizer.step()
                # kept on the device: reading it here would add a sync per batch
                batch_losses.append((loss.detach(), batch.num_graphs))
            sync()
            return batch_losses

        def accuracy(loader):
            correct, total = 0, 0
            for batch in loader:
                batch = batch.to(device)
                logits = model(batch.x, batch.edge_index, batch.batch)
                correct += (logits.argmax(dim=1) == batch.y).sum().item()
                total += batch.num_graphs
            return correct / total

        @torch.no_grad()
        def evaluate():
            model.eval()
            return [accuracy(loader) for loader in eval_loaders]

        num_graphs = len(dataset)

    start = time.perf_counter()
    epoch_times, losses, curves = [], [], Curves()
    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        batch_losses = train_epoch()
        epoch_times.append((time.perf_counter() - t0) * 1e3)
        # mean training loss over the epoch, read outside the timed section
        losses.append(
            sum(float(loss.item()) * n for loss, n in batch_losses)
            / sum(n for _, n in batch_losses)
        )
        if epoch % args.eval_every == args.eval_every - 1 or epoch == args.epochs - 1:
            curves.add(epoch + 1, *evaluate())
    total = time.perf_counter() - start
    timed = epoch_times[args.warmup :]
    q1, median, q3 = np.percentile(timed, [25, 50, 75])
    return {
        "num_graphs": int(num_graphs),
        "epoch_ms": float(median),
        "epoch_ms_q1": float(q1),
        "epoch_ms_q3": float(q3),
        "total_s": total,
        **curves.summary(losses),
    }
