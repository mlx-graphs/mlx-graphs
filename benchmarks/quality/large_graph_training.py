"""GraphSAGE mini-batch training with NeighborLoader on large graphs, in
mlx-graphs and in PyG, with the same hyperparameters, fan-outs and seeds.

    python benchmarks/run_capped.py --gb 24 -- \\
        python benchmarks/quality/large_graph_training.py \\
        --data-dir .mlx_graphs_data --datasets flickr reddit --seeds 0 1 2

PyG's ``NeighborLoader`` needs pyg-lib (or torch_sparse): put it on the
``PYTHONPATH`` if it is not installed.

For every dataset and seed, a 2-layer GraphSAGE (mean aggregation, hidden
256, dropout 0.5, Adam lr 0.003) is trained for ``--epochs`` epochs on
mini-batches of 1024 training seeds with fan-outs ``[25, 10]``, the loss
being computed on the seeds only. Test (and validation) accuracy is computed
after every epoch with the same sampled neighborhoods (``NeighborLoader``
over the test nodes, batches of 4096), the reported test accuracy is the
one at the best validation accuracy. Time per epoch covers sampling,
feature gathering, forward, backward and the optimizer step, and is the
median over the epochs (the first epoch included).

Runs:

- ``mlx-graphs``: ``NeighborLoader(GraphData)`` (MLX on the GPU).
- ``mlx-graphs memmap``: ``NeighborLoader((MemmapFeatureStore,
  InMemoryGraphStore))``: the same model, with features read from ``.npy``
  files per mini-batch (``--memmap``).
- ``PyG``: ``torch_geometric.loader.NeighborLoader`` with pyg-lib, PyTorch on
  the CPU (``--torch-threads``).

Every run is a separate subprocess; results are printed as a markdown table.
"""

import argparse
import contextlib
import io
import json
import os
import shutil
import statistics
import subprocess
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

HIDDEN, DROPOUT, LR = 256, 0.5, 0.003
FANOUT, BATCH, EVAL_BATCH = [25, 10], 1024, 4096
CLASSES = {"flickr": "Flickr", "reddit": "Reddit", "reddit2": "Reddit2"}


def load_mlx(name, data_dir):
    import mlx_graphs.datasets as mxd

    with contextlib.redirect_stdout(io.StringIO()):
        return getattr(mxd, CLASSES[name])(base_dir=data_dir)[0]


def run_mlx(name, data_dir, seed, epochs, memmap_dir=None):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    from mlx_graphs.data import InMemoryGraphStore, MemmapFeatureStore
    from mlx_graphs.loaders import NeighborLoader
    from mlx_graphs.nn import SAGEConv

    # MLX keeps freed buffers in a cache that grows up to the memory limit by
    # default: cap it, as the machine is shared (does not change the speed).
    mx.set_cache_limit(2 << 30)
    graph = load_mlx(name, data_dir)
    num_classes = int(mx.max(graph.node_labels).item()) + 1
    num_features = graph.node_features.shape[1]
    train, val, test = (graph.train_mask, graph.val_mask, graph.test_mask)
    data = graph
    if memmap_dir is not None:
        fs = MemmapFeatureStore.from_data(
            graph, memmap_dir, attrs=["node_features", "node_labels"]
        )
        gs = InMemoryGraphStore.from_data(graph)
        data = (fs, gs)
        del graph

    class SAGE(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = SAGEConv(num_features, HIDDEN)
            self.conv2 = SAGEConv(HIDDEN, num_classes)
            self.dropout = nn.Dropout(DROPOUT)

        def __call__(self, edge_index, x):
            x = nn.relu(self.conv1(edge_index, x))
            x = self.dropout(x)
            return self.conv2(edge_index, x)

    mx.random.seed(seed)
    np.random.seed(seed)
    model = SAGE()
    mx.eval(model.parameters())
    opt = optim.Adam(learning_rate=LR, bias_correction=True)

    def loss_fn(model, edge_index, x, y, n):
        logits = model(edge_index, x)[:n]
        return nn.losses.cross_entropy(logits, y[:n], reduction="mean")

    step = nn.value_and_grad(model, loss_fn)
    train_loader = NeighborLoader(
        data, FANOUT, input_nodes=train, batch_size=BATCH, shuffle=True
    )

    def evaluate(mask):
        model.eval()
        correct = total = 0
        for b in NeighborLoader(data, FANOUT, input_nodes=mask, batch_size=EVAL_BATCH):
            n = b.batch_size
            pred = mx.argmax(model(b.edge_index, b.node_features)[:n], axis=-1)
            correct += int((pred == b.node_labels[:n]).sum().item())
            total += n
        model.train()
        return correct / total

    times, history = [], []
    for _ in range(epochs):
        t0 = time.perf_counter()
        losses = []
        for b in train_loader:
            loss, grads = step(
                model, b.edge_index, b.node_features, b.node_labels, b.batch_size
            )
            opt.update(model, grads)
            mx.eval(loss, model.parameters(), opt.state)
            losses.append(loss.item())
        times.append(time.perf_counter() - t0)
        history.append((float(np.mean(losses)), evaluate(val), evaluate(test)))
    return times, history


def run_pyg(name, data_dir, seed, epochs, threads):
    import torch
    import torch.nn.functional as F
    from torch_geometric.loader import NeighborLoader
    from torch_geometric.nn import SAGEConv

    torch.set_num_threads(threads)
    import torch_geometric.datasets as pygd

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        data = getattr(pygd, CLASSES[name])(
            os.path.join(data_dir, "pyg", CLASSES[name])
        )[0]
    num_classes = int(data.y.max()) + 1

    class SAGE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = SAGEConv(data.num_features, HIDDEN)
            self.conv2 = SAGEConv(HIDDEN, num_classes)

        def forward(self, x, edge_index):
            x = F.relu(self.conv1(x, edge_index))
            x = F.dropout(x, p=DROPOUT, training=self.training)
            return self.conv2(x, edge_index)

    torch.manual_seed(seed)
    model = SAGE()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    train_loader = NeighborLoader(
        data, FANOUT, input_nodes=data.train_mask, batch_size=BATCH, shuffle=True
    )

    @torch.no_grad()
    def evaluate(mask):
        model.eval()
        correct = total = 0
        for b in NeighborLoader(data, FANOUT, input_nodes=mask, batch_size=EVAL_BATCH):
            n = b.batch_size
            pred = model(b.x, b.edge_index)[:n].argmax(-1)
            correct += int((pred == b.y[:n]).sum())
            total += n
        model.train()
        return correct / total

    times, history = [], []
    for _ in range(epochs):
        t0 = time.perf_counter()
        losses = []
        for b in train_loader:
            opt.zero_grad()
            n = b.batch_size
            loss = F.cross_entropy(model(b.x, b.edge_index)[:n], b.y[:n])
            loss.backward()
            opt.step()
            losses.append(float(loss))
        times.append(time.perf_counter() - t0)
        history.append(
            (float(np.mean(losses)), evaluate(data.val_mask), evaluate(data.test_mask))
        )
    return times, history


def summarize(times, history):
    best = max(range(len(history)), key=lambda i: history[i][1])
    return {
        "epoch_s": statistics.median(times),
        "val": history[best][1],
        "test": history[best][2],
        "final_loss": history[-1][0],
        "times": times,
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", default=os.path.join(REPO_ROOT, ".mlx_graphs_data")
    )
    parser.add_argument("--datasets", nargs="+", default=["flickr"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--frameworks", nargs="+", default=["mlx", "memmap", "pyg"])
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--memmap-dir", default=None)
    parser.add_argument("--out", default=None, help="append JSON lines here")
    parser.add_argument("--worker", nargs=3, help=argparse.SUPPRESS)
    args = parser.parse_args()
    data_dir = os.path.abspath(args.data_dir)

    if args.worker:
        framework, name, seed = args.worker
        seed = int(seed)
        if framework == "pyg":
            times, history = run_pyg(
                name, data_dir, seed, args.epochs, args.torch_threads
            )
        else:
            memmap_dir = None
            if framework == "memmap":
                memmap_dir = args.memmap_dir or os.path.join(
                    data_dir, f"_memmap_{name}"
                )
            try:
                times, history = run_mlx(name, data_dir, seed, args.epochs, memmap_dir)
            finally:
                if memmap_dir is not None:
                    shutil.rmtree(memmap_dir, ignore_errors=True)
        print("RESULT " + json.dumps(summarize(times, history)), flush=True)
        return

    results = {}
    for name in args.datasets:
        for framework in args.frameworks:
            for seed in args.seeds:
                out = subprocess.run(
                    [
                        sys.executable,
                        __file__,
                        "--data-dir",
                        data_dir,
                        "--epochs",
                        str(args.epochs),
                        "--torch-threads",
                        str(args.torch_threads),
                        "--worker",
                        framework,
                        name,
                        str(seed),
                    ],
                    capture_output=True,
                    text=True,
                )
                lines = [x for x in out.stdout.splitlines() if x.startswith("RESULT ")]
                if out.returncode != 0 or not lines:
                    print(
                        f"{name} {framework} {seed}: failed\n{out.stderr[-3000:]}",
                        flush=True,
                    )
                    continue
                res = json.loads(lines[-1][7:])
                res.update(dataset=name, framework=framework, seed=seed)
                results.setdefault((name, framework), []).append(res)
                print(
                    f"{name} {framework} seed={seed}: test={res['test']:.4f} "
                    f"val={res['val']:.4f} epoch={res['epoch_s']:.2f}s",
                    flush=True,
                )
                if args.out:
                    with open(args.out, "a") as f:
                        f.write(json.dumps(res) + "\n")

    label = {
        "mlx": "mlx-graphs",
        "memmap": "mlx-graphs (MemmapFeatureStore)",
        "pyg": "PyG",
    }
    print("\n| dataset | framework | seeds | test acc | val acc | s / epoch |")
    print("|---|---|---|---|---|---|")
    for (name, framework), rs in results.items():
        test = [r["test"] for r in rs]
        val = [r["val"] for r in rs]
        ep = [r["epoch_s"] for r in rs]
        print(
            f"| {CLASSES[name]} | {label[framework]} | {len(rs)} | "
            f"{np.mean(test):.4f} ± {np.std(test):.4f} | {np.mean(val):.4f} | "
            f"{np.median(ep):.2f} |"
        )


if __name__ == "__main__":
    main()
