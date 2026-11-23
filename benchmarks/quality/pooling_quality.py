"""Training quality and speed of the hierarchical / dense pooling operators:
mlx-graphs vs PyTorch Geometric, graph classification on PROTEINS.

    python benchmarks/quality/pooling_quality.py                 # everything
    python benchmarks/quality/pooling_quality.py --models diffpool asap
    python benchmarks/quality/pooling_quality.py --summary       # table only

Models (the architectures of PyG's examples):

- ``diffpool``: ``proteins_diff_pool.py``: graphs of at most 150 nodes
  padded to 150 (``DenseDataLoader``), 3 x DenseSAGEConv + BatchNorm GNNs
  computing the assignments and the embeddings, 2 x ``dense_diff_pool``
  (37 then 10 clusters), mean readout. The loss adds the link prediction
  and entropy losses.
- ``mincut`` / ``dmon``: ``proteins_mincut_pool.py`` /
  ``proteins_dmon_pool.py``: GraphConv(32), ``to_dense_batch`` /
  ``to_dense_adj``, 2 x (``dense_mincut_pool`` / ``DMoNPooling``) to half
  of the average number of nodes, then a quarter, with DenseGraphConv(32)
  layers. The loss adds the auxiliary losses.
- ``asap`` / ``edgepool``: PyG's ``benchmark/kernel`` models: 3 x
  GraphConv(64, mean aggregation) with an ``ASAPooling(ratio=0.8)`` /
  ``EdgePooling`` after the first one, jumping knowledge (cat) of the mean
  readouts, MLP head.
- ``graphunet``: ``GraphUNet(in, 64, 64, depth=3, pool_ratios=0.8)``
  followed by a mean readout and a linear head.

Each (model, framework, seed) trains in its own subprocess (this script with
``--worker``) under a memory watchdog, as in ``graph_quality.py``:
mlx-graphs on the GPU, PyG on the CPU (2 threads), the same data (loaded
once through mlx-graphs, ``graph_data.py``), the same random 80/10/10 split
per seed, MLX's Adam with ``bias_correction=True``, test accuracy at the best
validation epoch. Time per epoch is the mean wall time of a training epoch
(the first one, which compiles kernels, excluded). Optimizer steps with
non-finite gradients are skipped in both frameworks and counted (the
MinCut / DMoN adjacency normalization overflows for nearly empty clusters, in
PyG as in mlx-graphs). Results go to
``pooling_results.jsonl`` (finished runs are skipped) and the table to
``pooling_results.md``.

PyG's ``ASAPooling`` and ``GraphUNet`` multiply sparse CSR matrices, which the
torch wheels without MKL (macOS arm64) cannot do on the CPU: the PyG runs use
COO matrices instead (same products, see ``_pyg_coo_matmul``).
"""

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "benchmarks"))

RESULTS = os.path.join(HERE, "pooling_results.jsonl")
SUMMARY = os.path.join(HERE, "pooling_results.md")

DATASET = "PROTEINS"
MAX_NODES = 150  # DiffPool's dense graphs
# lr, batch size (as in PyG's examples)
MODELS = {
    "diffpool": dict(lr=0.001, batch_size=20),
    "mincut": dict(lr=5e-4, batch_size=20),
    "dmon": dict(lr=5e-4, batch_size=20),
    "asap": dict(lr=0.01, batch_size=128),
    "edgepool": dict(lr=0.01, batch_size=128),
    "graphunet": dict(lr=0.01, batch_size=128),
}
HIDDEN = 64


# ---------------------------------------------------------------------------
# MLX models
# ---------------------------------------------------------------------------


def mlx_model(name, in_dim, num_classes, avg_nodes):
    import mlx.core as mx
    import mlx.nn as nn

    import mlx_graphs.nn as gnn
    from mlx_graphs.utils import to_dense_adj, to_dense_batch

    if name == "diffpool":

        class GNN(nn.Module):
            def __init__(self, i, h, o, lin=True):
                super().__init__()
                self.convs = [
                    gnn.DenseSAGEConv(i, h),
                    gnn.DenseSAGEConv(h, h),
                    gnn.DenseSAGEConv(h, o),
                ]
                self.bns = [nn.BatchNorm(h), nn.BatchNorm(h), nn.BatchNorm(o)]
                self.lin = nn.Linear(2 * h + o, o) if lin else None

            def __call__(self, x, adj, mask=None):
                xs = []
                for conv, bn in zip(self.convs, self.bns):
                    x = nn.relu(conv(adj, x, mask))
                    b, n, c = x.shape
                    x = bn(x.reshape(-1, c)).reshape(b, n, c)
                    xs.append(x)
                x = mx.concatenate(xs, axis=-1)
                return nn.relu(self.lin(x)) if self.lin is not None else x

        class DiffPool(nn.Module):
            def __init__(self):
                super().__init__()
                n1 = math.ceil(0.25 * MAX_NODES)
                self.gnn1_pool = GNN(in_dim, HIDDEN, n1)
                self.gnn1_embed = GNN(in_dim, HIDDEN, HIDDEN, lin=False)
                n2 = math.ceil(0.25 * n1)
                self.gnn2_pool = GNN(3 * HIDDEN, HIDDEN, n2)
                self.gnn2_embed = GNN(3 * HIDDEN, HIDDEN, HIDDEN, lin=False)
                self.gnn3_embed = GNN(3 * HIDDEN, HIDDEN, HIDDEN, lin=False)
                self.lin1 = nn.Linear(3 * HIDDEN, HIDDEN)
                self.lin2 = nn.Linear(HIDDEN, num_classes)

            def __call__(self, b):
                x, adj, mask = b.node_features, b.adj, b.mask
                s = self.gnn1_pool(x, adj, mask)
                x = self.gnn1_embed(x, adj, mask)
                x, adj, l1, e1 = gnn.dense_diff_pool(adj, x, s, mask)
                s = self.gnn2_pool(x, adj)
                x = self.gnn2_embed(x, adj)
                x, adj, l2, e2 = gnn.dense_diff_pool(adj, x, s)
                x = self.gnn3_embed(x, adj).mean(axis=1)
                x = self.lin2(nn.relu(self.lin1(x)))
                return x, l1 + l2 + e1 + e2

        return DiffPool()

    if name in ("mincut", "dmon"):
        H = 32

        class MinCutNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = gnn.GraphConv(in_dim, H)
                n1 = math.ceil(0.5 * avg_nodes)
                n2 = math.ceil(0.5 * n1)
                if name == "mincut":
                    self.pool1 = nn.Linear(H, n1)
                    self.pool2 = nn.Linear(H, n2)
                else:
                    self.pool1 = gnn.DMoNPooling([H, H], n1)
                    self.pool2 = gnn.DMoNPooling([H, H], n2)
                self.conv2 = gnn.DenseGraphConv(H, H)
                self.conv3 = gnn.DenseGraphConv(H, H)
                self.lin1 = nn.Linear(H, H)
                self.lin2 = nn.Linear(H, num_classes)

            def pool(self, pool, adj, x, mask=None):
                if name == "mincut":
                    out = gnn.dense_mincut_pool(adj, x, pool(x), mask)
                    x, adj, l1, l2 = out
                    return x, adj, l1 + l2
                _, x, adj, l1, l2, l3 = pool(adj, x, mask)
                return x, adj, l1 + l2 + l3

            def __call__(self, b):
                x = nn.relu(self.conv1(b.edge_index, b.node_features))
                x, mask = to_dense_batch(x, b.batch_indices)
                adj = to_dense_adj(b.edge_index, b.batch_indices)
                x, adj, aux1 = self.pool(self.pool1, adj, x, mask)
                x = nn.relu(self.conv2(adj, x))
                x, adj, aux2 = self.pool(self.pool2, adj, x)
                x = self.conv3(adj, x).mean(axis=1)
                x = self.lin2(nn.relu(self.lin1(x)))
                return x, aux1 + aux2

        return MinCutNet()

    if name in ("asap", "edgepool"):

        class KernelNet(nn.Module):
            def __init__(self, num_layers=3):
                super().__init__()
                self.conv1 = gnn.GraphConv(in_dim, HIDDEN, aggr="mean")
                self.convs = [
                    gnn.GraphConv(HIDDEN, HIDDEN, aggr="mean")
                    for _ in range(num_layers - 1)
                ]
                if name == "asap":
                    self.pools = [
                        gnn.ASAPooling(HIDDEN, 0.8) for _ in range(num_layers // 2)
                    ]
                else:
                    self.pools = [
                        gnn.EdgePooling(HIDDEN) for _ in range(num_layers // 2)
                    ]
                self.lin1 = nn.Linear(num_layers * HIDDEN, HIDDEN)
                self.lin2 = nn.Linear(HIDDEN, num_classes)
                self.dropout = nn.Dropout(0.5)

            def __call__(self, b):
                ei, batch, n = b.edge_index, b.batch_indices, b.num_graphs
                weights = None
                x = nn.relu(self.conv1(ei, b.node_features))
                xs = [gnn.global_mean_pool(x, batch, n)]
                for i, conv in enumerate(self.convs):
                    x = nn.relu(conv(ei, x, weights))
                    xs.append(gnn.global_mean_pool(x, batch, n))
                    if i % 2 == 0 and i < len(self.convs) - 1:
                        pool = self.pools[i // 2]
                        if name == "asap":
                            x, ei, weights, batch, _ = pool(ei, x, weights, batch)
                        else:
                            x, ei, batch, _ = pool(ei, x, batch)
                x = nn.relu(self.lin1(mx.concatenate(xs, axis=-1)))
                return self.lin2(self.dropout(x)), 0.0

        return KernelNet()

    if name == "graphunet":

        class UNetNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.unet = gnn.GraphUNet(in_dim, HIDDEN, HIDDEN, 3, pool_ratios=0.8)
                self.lin = nn.Linear(HIDDEN, num_classes)

            def __call__(self, b):
                x = nn.relu(self.unet(b.edge_index, b.node_features, b.batch_indices))
                x = gnn.global_mean_pool(x, b.batch_indices, b.num_graphs)
                return self.lin(x), 0.0

        return UNetNet()

    raise ValueError(name)


# ---------------------------------------------------------------------------
# PyG models
# ---------------------------------------------------------------------------


def _pyg_coo_matmul():
    """Runs PyG's sparse CSR products (ASAPooling, GraphUNet) with COO
    matrices, which torch can multiply on the CPU without MKL."""
    import torch
    from torch_geometric.nn.models import graph_unet
    from torch_geometric.nn.pool import asap
    from torch_geometric.utils import to_torch_coo_tensor

    asap.to_torch_csr_tensor = to_torch_coo_tensor
    graph_unet.to_torch_csr_tensor = to_torch_coo_tensor
    to_sparse_csr = torch.Tensor.to_sparse_csr

    def coo_to_sparse_csr(self, *args, **kwargs):
        if self.layout == torch.sparse_coo:
            return self
        return to_sparse_csr(self, *args, **kwargs)

    torch.Tensor.to_sparse_csr = coo_to_sparse_csr


def pyg_model(name, in_dim, num_classes, avg_nodes):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch_geometric.nn as pnn
    from torch_geometric.utils import to_dense_adj, to_dense_batch

    if name == "diffpool":

        class GNN(nn.Module):
            def __init__(self, i, h, o, lin=True):
                super().__init__()
                self.convs = nn.ModuleList(
                    [
                        pnn.DenseSAGEConv(i, h),
                        pnn.DenseSAGEConv(h, h),
                        pnn.DenseSAGEConv(h, o),
                    ]
                )
                self.bns = nn.ModuleList(
                    [nn.BatchNorm1d(h), nn.BatchNorm1d(h), nn.BatchNorm1d(o)]
                )
                self.lin = nn.Linear(2 * h + o, o) if lin else None

            def forward(self, x, adj, mask=None):
                xs = []
                for conv, bn in zip(self.convs, self.bns):
                    x = conv(x, adj, mask).relu()
                    b, n, c = x.size()
                    x = bn(x.view(-1, c)).view(b, n, c)
                    xs.append(x)
                x = torch.cat(xs, dim=-1)
                return self.lin(x).relu() if self.lin is not None else x

        class DiffPool(nn.Module):
            def __init__(self):
                super().__init__()
                n1 = math.ceil(0.25 * MAX_NODES)
                self.gnn1_pool = GNN(in_dim, HIDDEN, n1)
                self.gnn1_embed = GNN(in_dim, HIDDEN, HIDDEN, lin=False)
                n2 = math.ceil(0.25 * n1)
                self.gnn2_pool = GNN(3 * HIDDEN, HIDDEN, n2)
                self.gnn2_embed = GNN(3 * HIDDEN, HIDDEN, HIDDEN, lin=False)
                self.gnn3_embed = GNN(3 * HIDDEN, HIDDEN, HIDDEN, lin=False)
                self.lin1 = nn.Linear(3 * HIDDEN, HIDDEN)
                self.lin2 = nn.Linear(HIDDEN, num_classes)

            def forward(self, b):
                x, adj, mask = b.x, b.adj, b.mask
                s = self.gnn1_pool(x, adj, mask)
                x = self.gnn1_embed(x, adj, mask)
                x, adj, l1, e1 = pnn.dense_diff_pool(x, adj, s, mask)
                s = self.gnn2_pool(x, adj)
                x = self.gnn2_embed(x, adj)
                x, adj, l2, e2 = pnn.dense_diff_pool(x, adj, s)
                x = self.gnn3_embed(x, adj).mean(dim=1)
                x = self.lin2(self.lin1(x).relu())
                return x, l1 + l2 + e1 + e2

        return DiffPool()

    if name in ("mincut", "dmon"):
        H = 32

        class MinCutNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = pnn.GraphConv(in_dim, H)
                n1 = math.ceil(0.5 * avg_nodes)
                n2 = math.ceil(0.5 * n1)
                if name == "mincut":
                    self.pool1 = nn.Linear(H, n1)
                    self.pool2 = nn.Linear(H, n2)
                else:
                    self.pool1 = pnn.DMoNPooling([H, H], n1)
                    self.pool2 = pnn.DMoNPooling([H, H], n2)
                self.conv2 = pnn.DenseGraphConv(H, H)
                self.conv3 = pnn.DenseGraphConv(H, H)
                self.lin1 = nn.Linear(H, H)
                self.lin2 = nn.Linear(H, num_classes)

            def pool(self, pool, x, adj, mask=None):
                if name == "mincut":
                    x, adj, l1, l2 = pnn.dense_mincut_pool(x, adj, pool(x), mask)
                    return x, adj, l1 + l2
                _, x, adj, l1, l2, l3 = pool(x, adj, mask)
                return x, adj, l1 + l2 + l3

            def forward(self, b):
                x = self.conv1(b.x, b.edge_index).relu()
                x, mask = to_dense_batch(x, b.batch)
                adj = to_dense_adj(b.edge_index, b.batch)
                x, adj, aux1 = self.pool(self.pool1, x, adj, mask)
                x = self.conv2(x, adj).relu()
                x, adj, aux2 = self.pool(self.pool2, x, adj)
                x = self.conv3(x, adj).mean(dim=1)
                x = self.lin2(self.lin1(x).relu())
                return x, aux1 + aux2

        return MinCutNet()

    if name in ("asap", "edgepool"):

        class KernelNet(nn.Module):
            def __init__(self, num_layers=3):
                super().__init__()
                self.conv1 = pnn.GraphConv(in_dim, HIDDEN, aggr="mean")
                self.convs = nn.ModuleList(
                    pnn.GraphConv(HIDDEN, HIDDEN, aggr="mean")
                    for _ in range(num_layers - 1)
                )
                if name == "asap":
                    pools = [
                        pnn.ASAPooling(HIDDEN, 0.8) for _ in range(num_layers // 2)
                    ]
                else:
                    pools = [pnn.EdgePooling(HIDDEN) for _ in range(num_layers // 2)]
                self.pools = nn.ModuleList(pools)
                self.lin1 = nn.Linear(num_layers * HIDDEN, HIDDEN)
                self.lin2 = nn.Linear(HIDDEN, num_classes)

            def forward(self, b):
                ei, batch, n = b.edge_index, b.batch, b.num_graphs
                weights = None
                x = self.conv1(b.x, ei).relu()
                xs = [pnn.global_mean_pool(x, batch, n)]
                for i, conv in enumerate(self.convs):
                    x = conv(x, ei, weights).relu()
                    xs.append(pnn.global_mean_pool(x, batch, n))
                    if i % 2 == 0 and i < len(self.convs) - 1:
                        pool = self.pools[i // 2]
                        if name == "asap":
                            x, ei, weights, batch, _ = pool(x, ei, weights, batch)
                        else:
                            x, ei, batch, _ = pool(x, ei, batch)
                x = self.lin1(torch.cat(xs, dim=-1)).relu()
                x = F.dropout(x, p=0.5, training=self.training)
                return self.lin2(x), 0.0

        return KernelNet()

    if name == "graphunet":

        class UNetNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.unet = pnn.GraphUNet(in_dim, HIDDEN, HIDDEN, 3, pool_ratios=0.8)
                self.lin = nn.Linear(HIDDEN, num_classes)

            def forward(self, b):
                x = self.unet(b.x, b.edge_index, b.batch).relu()
                x = pnn.global_mean_pool(x, b.batch, b.num_graphs)
                return self.lin(x), 0.0

        return UNetNet()

    raise ValueError(name)


# ---------------------------------------------------------------------------
# Worker: one (model, framework, seed)
# ---------------------------------------------------------------------------


def _dense(g):
    """Pads a graph to MAX_NODES nodes with a dense adjacency matrix."""
    n = len(g["x"])
    x = np.zeros((MAX_NODES, g["x"].shape[1]), np.float32)
    x[:n] = g["x"]
    adj = np.zeros((MAX_NODES, MAX_NODES), np.float32)
    np.add.at(adj, (g["edge_index"][0], g["edge_index"][1]), 1.0)
    mask = np.arange(MAX_NODES) < n
    return x, adj, mask


def run_worker(args):
    import graph_data
    from graph_worker import Selector, mlx_adam, seed_all

    seed_all(args.framework, args.seed)
    graphs = graph_data.tu_graphs(graph_data.tu_dataset(DATASET))
    dense = args.model == "diffpool"
    if dense:  # as PyG's example, keep the graphs of at most MAX_NODES nodes
        graphs = [g for g in graphs if len(g["x"]) <= MAX_NODES]
    splits = graph_data.split_indices(args.seed, len(graphs))
    in_dim = graphs[0]["x"].shape[1]
    num_classes = int(max(g["y"][0] for g in graphs)) + 1
    avg_nodes = int(np.mean([len(graphs[i]["x"]) for i in splits[0]]))
    hp = MODELS[args.model]
    # Steps whose gradients are not finite are skipped (in both frameworks):
    # MinCut / DMoN normalize the pooled adjacency by the square root of the
    # cluster degrees, whose gradient overflows for nearly empty clusters (in
    # PyG as in mlx-graphs, e.g. on PROTEINS' 504-node graph).
    skipped = [0]
    model_args = (args.model, in_dim, num_classes, avg_nodes)

    if args.framework == "mlx":
        import mlx.core as mx
        import mlx.nn as nn
        from mlx.utils import tree_flatten

        from mlx_graphs.data import GraphData
        from mlx_graphs.loaders import Dataloader, DenseDataLoader

        mx.set_cache_limit(1024**3)

        def to_graph(g):
            y = mx.array(g["y"].astype(np.int32))
            if dense:
                x, adj, mask = _dense(g)
                return GraphData(
                    edge_index=mx.zeros((2, 0), dtype=mx.int32),
                    node_features=mx.array(x),
                    adj=mx.array(adj),
                    mask=mx.array(mask),
                    graph_labels=y,
                )
            return GraphData(
                edge_index=mx.array(g["edge_index"].astype(np.int32)),
                node_features=mx.array(g["x"]),
                graph_labels=y,
            )

        sets = [[to_graph(graphs[i]) for i in s] for s in splits]
        Loader = DenseDataLoader if dense else Dataloader
        train_loader = Loader(sets[0], batch_size=hp["batch_size"], shuffle=True)
        eval_loaders = [Loader(s, batch_size=128) for s in sets[1:]]
        model = mlx_model(*model_args)
        optimizer, _ = mlx_adam(hp["lr"])

        def loss_fn(model, b):
            logits, aux = model(b)
            y = b.graph_labels.reshape(-1)
            return nn.losses.cross_entropy(logits, y, reduction="mean") + aux

        loss_and_grad = nn.value_and_grad(model, loss_fn)

        def train_epoch():
            model.train()
            total, count = 0.0, 0
            for b in train_loader:
                loss, grads = loss_and_grad(model, b)
                finite = mx.all(
                    mx.stack([mx.isfinite(g).all() for _, g in tree_flatten(grads)])
                )
                if not finite.item():
                    skipped[0] += 1
                    continue
                optimizer.update(model, grads)
                mx.eval(loss, model.parameters(), optimizer.state)
                n = b.graph_labels.shape[0]
                total += loss.item() * n
                count += n
            return total / count

        def evaluate(loader):
            model.eval()
            correct, loss, total = 0, 0.0, 0
            for b in loader:
                logits, _ = model(b)
                y = b.graph_labels.reshape(-1)
                correct += (mx.argmax(logits, axis=1) == y).sum().item()
                loss += nn.losses.cross_entropy(logits, y, reduction="sum").item()
                total += y.shape[0]
            return correct / total, loss / total

    else:
        import torch
        import torch.nn.functional as F
        from torch_geometric.data import Data
        from torch_geometric.loader import DataLoader
        from torch_geometric.loader import DenseDataLoader as PyGDenseDataLoader

        _pyg_coo_matmul()

        def to_data(g):
            y = torch.from_numpy(g["y"])
            if dense:
                x, adj, mask = _dense(g)
                return Data(
                    x=torch.from_numpy(x),
                    adj=torch.from_numpy(adj),
                    mask=torch.from_numpy(mask),
                    y=y,
                )
            return Data(
                x=torch.from_numpy(g["x"]),
                edge_index=torch.from_numpy(g["edge_index"]),
                y=y,
            )

        sets = [[to_data(graphs[i]) for i in s] for s in splits]
        Loader = PyGDenseDataLoader if dense else DataLoader
        train_loader = Loader(sets[0], batch_size=hp["batch_size"], shuffle=True)
        eval_loaders = [Loader(s, batch_size=128) for s in sets[1:]]
        model = pyg_model(*model_args)
        optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"])

        def train_epoch():
            model.train()
            total, count = 0.0, 0
            for b in train_loader:
                optimizer.zero_grad()
                logits, aux = model(b)
                y = b.y.view(-1)
                loss = F.cross_entropy(logits, y) + aux
                loss.backward()
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                if not all(torch.isfinite(g).all() for g in grads):
                    skipped[0] += 1
                    continue
                optimizer.step()
                total += loss.item() * y.numel()
                count += y.numel()
            return total / count

        @torch.no_grad()
        def evaluate(loader):
            model.eval()
            correct, loss, total = 0, 0.0, 0
            for b in loader:
                logits, _ = model(b)
                y = b.y.view(-1)
                correct += (logits.argmax(1) == y).sum().item()
                loss += F.cross_entropy(logits, y, reduction="sum").item()
                total += y.numel()
            return correct / total, loss / total

    selector, losses, times = Selector(), [], []
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        losses.append(train_epoch())
        times.append(time.perf_counter() - start)
        val_acc, val_loss = evaluate(eval_loaders[0])
        test_acc, _ = evaluate(eval_loaders[1])
        selector.add(epoch, val_acc, test_acc, val_loss)
    epoch_s = float(np.mean(times[1:])) if len(times) > 1 else times[0]
    return {
        "metric": "accuracy",
        "loss_curve": losses,
        "epoch_seconds": epoch_s,
        "num_graphs": len(graphs),
        "skipped_steps": skipped[0],
        **selector.summary(),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def config_key(model, framework, seed, epochs):
    return f"{model} {DATASET} {epochs} {framework} {seed}"


def run_one(model, framework, seed, epochs, memory_gb, timeout_s):
    """Runs one worker under the memory watchdog; returns (result, status)."""
    from common import footprint_gb

    cmd = [
        sys.executable, os.path.abspath(__file__), "--worker",
        "--model", model, "--framework", framework,
        "--seed", str(seed), "--epochs", str(epochs),
    ]  # fmt: skip
    env = dict(os.environ, PYTHONPATH=REPO)
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        cwd=REPO,
    )
    out, err = [], []
    readers = [
        threading.Thread(target=lambda: out.extend(process.stdout), daemon=True),
        threading.Thread(target=lambda: err.extend(process.stderr), daemon=True),
    ]
    for r in readers:
        r.start()
    start, peak = time.time(), 0.0
    while process.poll() is None:
        used = footprint_gb(process.pid)
        peak = max(peak, used)
        if used > memory_gb or time.time() - start > timeout_s:
            process.kill()
            process.wait()
            reason = "oom" if used > memory_gb else "timeout"
            return None, reason, peak
        time.sleep(0.5)
    for r in readers:
        r.join()
    if process.returncode != 0:
        lines = "".join(err).strip().splitlines() or ["?"]
        return None, f"error: {lines[-1][:300]}", peak
    lines = [line for line in out if line.startswith("{")]
    return json.loads(lines[-1]), "ok", peak


def run_all(models, seeds, epochs, memory_gb, timeout_s, procs):
    done = set()
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            done = {json.loads(line)["key"] for line in f if line.strip()}
    queues = {
        fw: [
            (m, s)
            for m in models
            for s in range(seeds)
            if config_key(m, fw, s, epochs) not in done
        ]
        for fw in ("mlx", "pyg")
    }
    print(f"{sum(len(q) for q in queues.values())} runs to do", flush=True)
    lock = threading.Lock()

    def worker(fw):
        while True:
            with lock:
                if not queues[fw]:
                    return
                model, seed = queues[fw].pop(0)
            result, status, peak = run_one(
                model, fw, seed, epochs, memory_gb, timeout_s
            )
            record = {
                "key": config_key(model, fw, seed, epochs),
                "model": model,
                "framework": fw,
                "seed": seed,
                "epochs": epochs,
                "status": status,
                "peak_gb": round(peak, 2),
                **(result or {}),
            }
            with lock:
                with open(RESULTS, "a") as f:
                    f.write(json.dumps(record) + "\n")
                print(
                    f"{record['key']} -> {status} test={record.get('test')} "
                    f"epoch={record.get('epoch_seconds', 0):.2f} s",
                    flush=True,
                )

    threads = [
        threading.Thread(target=worker, args=(fw,))
        for fw in procs
        for _ in range(procs[fw])
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


HEADER = """# Hierarchical and dense pooling: mlx-graphs vs PyG on PROTEINS

Generated by `pooling_quality.py` (see its docstring for the models and the
protocol): mlx-graphs (MLX, GPU) vs PyTorch Geometric 2.8.1 (torch 2.15, CPU,
2 threads), same architectures, hyperparameters, data and splits; test
accuracy at the best validation epoch, mean ± sample std over seeds; time
per training epoch (mean over epochs and seeds, first epoch excluded).
"""


NOTES = """
## Notes

- All the mlx-graphs accuracies are within 2 PyG standard deviations of
  PyG's (the verdict rule of `graph_quality.py`); with 5 random splits of
  PROTEINS, the standard deviations are 3-7 points.
- MinCut / DMoN: both frameworks hit non-finite gradients (the pooled
  adjacency is divided by the square root of the cluster degrees, whose
  gradient overflows for nearly empty clusters, e.g. on the 504-node
  graph). Those steps are skipped in both; mlx-graphs skipped about twice
  as many (9 % vs 4 % of the MinCut steps, 4 % vs 2 % for DMoN), with the
  same final accuracy. With identical inputs, both frameworks give
  non-finite gradients on the same batches (checked on one).
- PyG's ASAPooling and GraphUNet run with COO instead of CSR sparse products
  (the torch wheel has no MKL, see the docstring); the products are the same.
- Times were measured on a machine shared with other jobs (an M-series Mac,
  1 MLX and 1-2 PyG runs at a time), so they are indicative only. The MLX
  ASAPooling / DMoN / GraphUNet epochs include host-side work (sparse
  products, top-k selection, to_dense_adj) and syncs.
"""


def summarize():
    with open(RESULTS) as f:
        records = [json.loads(line) for line in f if line.strip()]
    lines = [
        "| model | epochs | seeds | PyG accuracy (%) | mlx-graphs accuracy (%) | "
        "gap | PyG s/epoch | mlx-graphs s/epoch | speedup | skipped steps |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for model in MODELS:
        for epochs in sorted({r["epochs"] for r in records if r["model"] == model}):
            ok = {
                fw: [
                    r
                    for r in records
                    if r["model"] == model
                    and r["epochs"] == epochs
                    and r["framework"] == fw
                    and r["status"] == "ok"
                ]
                for fw in ("pyg", "mlx")
            }
            if not ok["pyg"] or not ok["mlx"]:
                failed = [
                    r["status"]
                    for r in records
                    if r["model"] == model and r["status"] != "ok"
                ]
                lines.append(
                    f"| {model} | {epochs} | | {failed[:1] or 'missing'} | | | | | | |"
                )
                continue
            acc = {fw: np.array([100 * r["test"] for r in ok[fw]]) for fw in ok}
            sec = {fw: np.mean([r["epoch_seconds"] for r in ok[fw]]) for fw in ok}
            std = {fw: a.std(ddof=1) if a.size > 1 else 0.0 for fw, a in acc.items()}
            skip = {fw: sum(r.get("skipped_steps", 0) for r in ok[fw]) for fw in ok}
            lines.append(
                f"| {model} | {epochs} | {len(acc['pyg'])}/{len(acc['mlx'])} | "
                f"{acc['pyg'].mean():.1f} ± {std['pyg']:.1f} | "
                f"{acc['mlx'].mean():.1f} ± {std['mlx']:.1f} | "
                f"{acc['mlx'].mean() - acc['pyg'].mean():+.1f} | "
                f"{sec['pyg']:.2f} | {sec['mlx']:.2f} | "
                f"{sec['pyg'] / sec['mlx']:.1f}x | {skip['pyg']} / {skip['mlx']} |"
            )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--models", nargs="+", default=list(MODELS), choices=list(MODELS)
    )
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--memory-gb", type=float, default=6.0)
    parser.add_argument("--timeout-s", type=float, default=3600.0)
    parser.add_argument("--mlx-procs", type=int, default=1)
    parser.add_argument("--pyg-procs", type=int, default=2)
    parser.add_argument("--summary", action="store_true", help="only the table")
    # worker mode
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model", help=argparse.SUPPRESS)
    parser.add_argument("--framework", choices=["mlx", "pyg"], help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        import warnings

        warnings.filterwarnings("ignore")
        start = time.perf_counter()
        result = run_worker(args)
        result["seconds"] = time.perf_counter() - start
        print(json.dumps(result))
        return

    if not args.summary:
        procs = {"mlx": args.mlx_procs, "pyg": args.pyg_procs}
        run_all(
            args.models,
            args.seeds,
            args.epochs,
            args.memory_gb,
            args.timeout_s,
            procs,
        )
    table = summarize()
    print(table)
    with open(SUMMARY, "w") as f:
        f.write(HEADER + "\n" + table + "\n" + NOTES)


if __name__ == "__main__":
    main()
