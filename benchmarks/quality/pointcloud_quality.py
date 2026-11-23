"""Training quality and speed of point cloud models: mlx-graphs vs PyTorch
Geometric on ModelNet10.

    python benchmarks/quality/pointcloud_quality.py                 # everything
    python benchmarks/quality/pointcloud_quality.py --models pointnet2 --epochs 5
    python benchmarks/quality/pointcloud_quality.py --summary       # table only

Models:

- ``pointnet2``: PyG's ``examples/pointnet2_classification.py`` (farthest
  point sampling, radius neighborhoods, ``PointNetConv``, global max pooling).
- ``dgcnn``: PyG's ``examples/dgcnn_classification.py`` (``DynamicEdgeConv``
  with k = 20).

Fairness rules: same architecture, hyperparameters (Adam, lr 1e-3, batch size
32; MLX's Adam with ``bias_correction=True``) and data. The meshes are loaded
once through mlx-graphs (``ModelNet`` + ``NormalizeScale``) and 1024 points
are sampled once per shape (``SamplePoints``); both frameworks train on these
same fixed point clouds (PyG's example resamples every epoch), with the same
shuffling. Each (model, framework, seed) runs in its own subprocess: mlx-graphs
on the GPU, PyG on the CPU (``--torch-threads`` threads).

PyG >= 2.7 runs ``fps`` and ``radius`` from ``pyg-lib``, which has no wheels for
this torch version; the PyG model then calls ``torch_cluster`` (the same
algorithms, which pyg-lib was ported from), built from source and put on
``PYTHONPATH``::

    SDKROOT=$(xcrun --show-sdk-path) uv pip install --target <dir> \\
        --no-build-isolation --no-deps torch-cluster

Results are appended to ``pointcloud_results.jsonl`` and summarized in
``pointcloud_results.md``.
"""

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA_DIR = os.environ.get("MLX_GRAPHS_BENCH_DATA") or os.path.join(
    REPO, ".mlx_graphs_data"
)
MLX_DIR = os.path.join(DATA_DIR, "quality_pointcloud")
RESULTS = os.path.join(HERE, "pointcloud_results.jsonl")
SUMMARY = os.path.join(HERE, "pointcloud_results.md")
NUM_POINTS = 1024
BATCH_SIZE = 32
LR = 1e-3


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def modelnet10(num_points: int = NUM_POINTS) -> dict:
    """Fixed point clouds sampled once from the ModelNet10 meshes, cached in
    an ``.npz`` file: ``{train,test}_pos`` [G, num_points, 3], ``{train,
    test}_y`` [G]."""
    path = os.path.join(MLX_DIR, f"modelnet10_{num_points}.npz")
    if not os.path.exists(path):
        import mlx.core as mx

        from mlx_graphs.datasets import ModelNet
        from mlx_graphs.transforms import NormalizeScale, SamplePoints

        mx.random.seed(0)
        sample = SamplePoints(num_points)
        arrays = {}
        for split in ("train", "test"):
            with contextlib.redirect_stdout(io.StringIO()):
                dataset = ModelNet(
                    "10",
                    train=split == "train",
                    base_dir=MLX_DIR,
                    pre_transform=NormalizeScale(),
                )
            graphs = [sample(g) for g in dataset]
            arrays[f"{split}_pos"] = np.stack([np.array(g.pos) for g in graphs])
            arrays[f"{split}_y"] = np.array(
                [g.graph_labels.item() for g in graphs], dtype=np.int64
            )
        os.makedirs(MLX_DIR, exist_ok=True)
        np.savez(path + ".tmp.npz", **arrays)
        os.replace(path + ".tmp.npz", path)
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def batches(num_graphs: int, shuffle_seed=None):
    order = np.arange(num_graphs)
    if shuffle_seed is not None:
        order = np.random.default_rng(shuffle_seed).permutation(num_graphs)
    for start in range(0, num_graphs, BATCH_SIZE):
        yield order[start : start + BATCH_SIZE]


# ---------------------------------------------------------------------------
# mlx-graphs
# ---------------------------------------------------------------------------


def run_mlx(model_name: str, epochs: int, seed: int, data: dict) -> dict:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    from mlx_graphs.nn import (
        MLP,
        DynamicEdgeConv,
        PointNetConv,
        global_max_pool,
    )
    from mlx_graphs.utils import fps, radius

    mx.random.seed(seed)

    class SAModule(nn.Module):
        def __init__(self, ratio, r, mlp):
            super().__init__()
            self.ratio, self.r = ratio, r
            self.conv = PointNetConv(mlp, add_self_loops=False)

        def __call__(self, x, pos, batch):
            idx = fps(pos, batch, ratio=self.ratio)
            row, col = radius(
                pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64
            )
            edge_index = mx.stack([col, row])
            x_dst = None if x is None else x[idx]
            x = self.conv(edge_index, (x, x_dst), (pos, pos[idx]))
            return x, pos[idx], batch[idx]

    class PointNet2(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.sa1 = SAModule(0.5, 0.2, MLP([3, 64, 64, 128], norm="batch"))
            self.sa2 = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256], norm="batch"))
            self.sa3 = MLP([256 + 3, 256, 512, 1024], norm="batch")
            self.mlp = MLP([1024, 512, 256, num_classes], dropout=0.5)

        def __call__(self, pos, batch, num_graphs):
            x, pos, batch = self.sa1(None, pos, batch)
            x, pos, batch = self.sa2(x, pos, batch)
            x = self.sa3(mx.concatenate([x, pos], axis=1))
            x = global_max_pool(x, batch, num_graphs)
            return self.mlp(x)

    class DGCNN(nn.Module):
        def __init__(self, num_classes, k=20):
            super().__init__()
            self.conv1 = DynamicEdgeConv(MLP([2 * 3, 64, 64, 64], norm="batch"), k)
            self.conv2 = DynamicEdgeConv(MLP([2 * 64, 128], norm="batch"), k)
            self.lin1 = nn.Linear(128 + 64, 1024)
            self.mlp = MLP([1024, 512, 256, num_classes], dropout=0.5)

        def __call__(self, pos, batch, num_graphs):
            x1 = self.conv1(pos, batch)
            x2 = self.conv2(x1, batch)
            out = self.lin1(mx.concatenate([x1, x2], axis=1))
            out = global_max_pool(out, batch, num_graphs)
            return self.mlp(out)

    num_classes = int(data["train_y"].max()) + 1
    model = PointNet2(num_classes) if model_name == "pointnet2" else DGCNN(num_classes)
    mx.eval(model.parameters())
    optimizer = optim.Adam(learning_rate=LR, bias_correction=True)

    def inputs(split, idx):
        pos = data[f"{split}_pos"][idx].reshape(-1, 3)
        batch = np.repeat(np.arange(len(idx)), data[f"{split}_pos"].shape[1])
        return mx.array(pos), mx.array(batch.astype(np.int32)), len(idx)

    def loss_fn(model, pos, batch, num_graphs, y):
        logits = model(pos, batch, num_graphs)
        return nn.losses.cross_entropy(logits, y, reduction="mean")

    step = nn.value_and_grad(model, loss_fn)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        start = time.perf_counter()
        total = 0.0
        num_train = len(data["train_y"])
        for idx in batches(num_train, shuffle_seed=seed * 1000 + epoch):
            pos, batch, n = inputs("train", idx)
            loss, grads = step(model, pos, batch, n, mx.array(data["train_y"][idx]))
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)
            total += loss.item() * n
        train_time = time.perf_counter() - start

        model.eval()
        correct = 0
        start = time.perf_counter()
        for idx in batches(len(data["test_y"])):
            pos, batch, n = inputs("test", idx)
            pred = mx.argmax(model(pos, batch, n), axis=1)
            correct += int((np.array(pred) == data["test_y"][idx]).sum())
        test_time = time.perf_counter() - start
        history.append(
            dict(
                epoch=epoch,
                loss=total / num_train,
                test_acc=correct / len(data["test_y"]),
                train_time=train_time,
                test_time=test_time,
            )
        )
        print(json.dumps(history[-1]), file=sys.stderr, flush=True)
    return dict(history=history, device="mlx gpu")


# ---------------------------------------------------------------------------
# PyG
# ---------------------------------------------------------------------------


def run_pyg(model_name: str, epochs: int, seed: int, data: dict, threads: int) -> dict:
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import MLP, EdgeConv, PointNetConv, global_max_pool

    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    try:
        import torch_cluster
    except ImportError as e:
        raise SystemExit(
            "The PyG models need torch_cluster (pyg-lib has no wheels for this "
            "torch): see the docstring of this script."
        ) from e

    class DynamicEdgeConv(EdgeConv):
        """PyG's DynamicEdgeConv, with torch_cluster's knn instead of
        pyg-lib's (same k nearest neighbors, the point itself included)."""

        def __init__(self, nn, k, aggr="max"):
            super().__init__(nn, aggr=aggr)
            self.k = k

        def forward(self, x, batch):
            edge_index = torch_cluster.knn(x, x, self.k, batch, batch).flip([0])
            return self.propagate(edge_index, x=(x, x))

    class SAModule(torch.nn.Module):
        def __init__(self, ratio, r, nn):
            super().__init__()
            self.ratio, self.r = ratio, r
            self.conv = PointNetConv(nn, add_self_loops=False)

        def forward(self, x, pos, batch):
            idx = torch_cluster.fps(pos, batch, ratio=self.ratio)
            row, col = torch_cluster.radius(
                pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64
            )
            edge_index = torch.stack([col, row], dim=0)
            x_dst = None if x is None else x[idx]
            x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
            return x, pos[idx], batch[idx]

    class PointNet2(torch.nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.sa1 = SAModule(0.5, 0.2, MLP([3, 64, 64, 128]))
            self.sa2 = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
            self.sa3 = MLP([256 + 3, 256, 512, 1024])
            self.mlp = MLP([1024, 512, 256, num_classes], dropout=0.5, norm=None)

        def forward(self, pos, batch, num_graphs):
            x, pos, batch = self.sa1(None, pos, batch)
            x, pos, batch = self.sa2(x, pos, batch)
            x = self.sa3(torch.cat([x, pos], dim=1))
            x = global_max_pool(x, batch, num_graphs)
            return self.mlp(x)

    class DGCNN(torch.nn.Module):
        def __init__(self, num_classes, k=20):
            super().__init__()
            self.conv1 = DynamicEdgeConv(MLP([2 * 3, 64, 64, 64]), k, "max")
            self.conv2 = DynamicEdgeConv(MLP([2 * 64, 128]), k, "max")
            self.lin1 = torch.nn.Linear(128 + 64, 1024)
            self.mlp = MLP([1024, 512, 256, num_classes], dropout=0.5, norm=None)

        def forward(self, pos, batch, num_graphs):
            x1 = self.conv1(pos, batch)
            x2 = self.conv2(x1, batch)
            out = self.lin1(torch.cat([x1, x2], dim=1))
            out = global_max_pool(out, batch, num_graphs)
            return self.mlp(out)

    num_classes = int(data["train_y"].max()) + 1
    model = PointNet2(num_classes) if model_name == "pointnet2" else DGCNN(num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    def inputs(split, idx):
        pos = torch.from_numpy(data[f"{split}_pos"][idx].reshape(-1, 3))
        batch = torch.arange(len(idx)).repeat_interleave(data[f"{split}_pos"].shape[1])
        return pos, batch, len(idx)

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        start = time.perf_counter()
        total = 0.0
        num_train = len(data["train_y"])
        for idx in batches(num_train, shuffle_seed=seed * 1000 + epoch):
            pos, batch, n = inputs("train", idx)
            optimizer.zero_grad()
            loss = F.cross_entropy(
                model(pos, batch, n), torch.from_numpy(data["train_y"][idx])
            )
            loss.backward()
            optimizer.step()
            total += loss.item() * n
        train_time = time.perf_counter() - start

        model.eval()
        correct = 0
        start = time.perf_counter()
        with torch.no_grad():
            for idx in batches(len(data["test_y"])):
                pos, batch, n = inputs("test", idx)
                pred = model(pos, batch, n).argmax(dim=1).numpy()
                correct += int((pred == data["test_y"][idx]).sum())
        test_time = time.perf_counter() - start
        history.append(
            dict(
                epoch=epoch,
                loss=total / num_train,
                test_acc=correct / len(data["test_y"]),
                train_time=train_time,
                test_time=test_time,
            )
        )
        print(json.dumps(history[-1]), file=sys.stderr, flush=True)
    return dict(history=history, device=f"torch cpu ({threads} threads)")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def worker(args):
    data = modelnet10()
    if args.worker == "mlx":
        result = run_mlx(args.model, args.epochs, args.seed, data)
    else:
        result = run_pyg(args.model, args.epochs, args.seed, data, args.torch_threads)
    print(json.dumps(result))


def load_results():
    if not os.path.exists(RESULTS):
        return []
    with open(RESULTS) as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize():
    rows = load_results()
    lines = [
        "# Point cloud classification: mlx-graphs vs PyG on ModelNet10",
        "",
        "Generated by `pointcloud_quality.py` (see its docstring for the setup):",
        f"{NUM_POINTS} points per shape sampled once, batch size {BATCH_SIZE}, "
        f"Adam lr {LR}. mlx-graphs on the GPU, PyG 2.8.1 on the CPU "
        "(`torch_cluster` for fps / radius / knn). Accuracy: test accuracy at "
        "the last epoch (and the best epoch). Time: mean training time per "
        "epoch (excluding the first epoch) and test evaluation time.",
        "",
        "| model | epochs | seed | framework | test acc (last / best) | "
        "train s/epoch | test s | final train loss |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(
        rows, key=lambda r: (r["model"], r["epochs"], r["seed"], r["framework"])
    ):
        h = r["history"]
        times = [e["train_time"] for e in h[1:]] or [h[0]["train_time"]]
        lines.append(
            f"| {r['model']} | {r['epochs']} | {r['seed']} | {r['framework']} "
            f"({r['device']}) | {100 * h[-1]['test_acc']:.1f} / "
            f"{100 * max(e['test_acc'] for e in h):.1f} | {np.mean(times):.1f} | "
            f"{np.mean([e['test_time'] for e in h]):.1f} | {h[-1]['loss']:.3f} |"
        )
    text = "\n".join(lines) + "\n"
    with open(SUMMARY, "w") as f:
        f.write(text)
    print(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--models", nargs="+", default=["pointnet2", "dgcnn"])
    parser.add_argument("--frameworks", nargs="+", default=["mlx", "pyg"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--worker", choices=["mlx", "pyg"], help=argparse.SUPPRESS)
    parser.add_argument("--model", help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        return worker(args)
    if args.summary:
        return summarize()

    modelnet10()  # prepares the cached point clouds once
    done = {
        (r["model"], r["framework"], r["epochs"], r["seed"]) for r in load_results()
    }
    for model in args.models:
        for seed in args.seeds:
            for framework in args.frameworks:
                if (model, framework, args.epochs, seed) in done:
                    continue
                print(f"{model} {framework} seed={seed}", flush=True)
                command = [
                    sys.executable,
                    __file__,
                    "--worker",
                    framework,
                    "--model",
                    model,
                    "--epochs",
                    str(args.epochs),
                    "--seed",
                    str(seed),
                    "--torch-threads",
                    str(args.torch_threads),
                ]
                process = subprocess.run(command, capture_output=True, text=True)
                if process.returncode != 0:
                    print(process.stderr[-3000:], file=sys.stderr)
                    continue
                result = json.loads(process.stdout.strip().splitlines()[-1])
                result.update(
                    model=model, framework=framework, epochs=args.epochs, seed=seed
                )
                with open(RESULTS, "a") as f:
                    f.write(json.dumps(result) + "\n")
    summarize()


if __name__ == "__main__":
    main()
