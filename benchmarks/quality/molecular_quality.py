"""Training quality of the molecular models of mlx-graphs vs PyTorch Geometric:
SchNet and DimeNet++ on a QM9 target, AttentiveFP on MoleculeNet ESOL.

    python benchmarks/quality/molecular_quality.py                   # everything
    python benchmarks/quality/molecular_quality.py --tasks attentivefp
    python benchmarks/quality/molecular_quality.py --summary         # table only

Each configuration (task, framework, seed) trains in its own subprocess
(``--worker``). mlx-graphs runs on the GPU and PyG on the CPU
(``--torch-threads`` threads). Results are appended to
``molecular_results.jsonl`` (finished configurations are skipped, so an
interrupted run resumes), and the table is written to
``molecular_results.md``.

Fairness rules (as in ``graph_quality.py``): same architecture,
hyperparameters and data (loaded once through mlx-graphs and handed to both
frameworks as NumPy arrays), same splits and same order of the mini-batches
(drawn from NumPy with the seed), MLX ``Adam(bias_correction=True)`` with
torch-style L2 weight decay, test metric taken at the best validation epoch.
The time per epoch is the training time only (forward, backward and update,
including the interaction graphs and the triplets).

PyG's ``radius_graph`` needs pyg-lib or torch-cluster, and DimeNet's
``triplets`` torch-sparse: when they are not installed, PyG gets vectorized
pure-torch versions of them (a brute-force radius graph per mini-batch, and
the triplets from sorted edges).

Budget (small, to compare the frameworks rather than to reach the published
accuracies): QM9 target 4 (HOMO-LUMO gap, eV) on a random subset of the
molecules, ESOL with a random 80/10/10 split.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
RESULTS = os.path.join(HERE, "molecular_results.jsonl")
SUMMARY = os.path.join(HERE, "molecular_results.md")

warnings.filterwarnings("ignore")

QM9_TARGET = 4  # HOMO-LUMO gap (eV)

# task -> (dataset, train/val/test sizes, epochs, batch size, lr, seeds)
TASKS = {
    "schnet": dict(
        dataset="QM9",
        sizes=(10000, 1000, 1000),
        epochs=10,
        batch=32,
        lr=5e-4,
        seeds=2,
    ),
    "dimenet++": dict(
        dataset="QM9",
        sizes=(5000, 1000, 1000),
        epochs=6,
        batch=32,
        lr=5e-4,
        seeds=1,
    ),
    "attentivefp": dict(
        dataset="ESOL",
        sizes=None,
        epochs=100,
        batch=64,
        lr=10**-2.5,
        seeds=3,
        weight_decay=1e-5,
    ),
}

SCHNET = dict(
    hidden_channels=64,
    num_filters=64,
    num_interactions=3,
    num_gaussians=50,
    cutoff=5.0,
)
DIMENET_PP = dict(
    hidden_channels=64,
    out_channels=1,
    num_blocks=2,
    int_emb_size=32,
    basis_emb_size=8,
    out_emb_channels=128,
    num_spherical=7,
    num_radial=6,
    cutoff=5.0,
)
ATTENTIVE_FP = dict(
    hidden_channels=200, out_channels=1, num_layers=2, num_timesteps=2, dropout=0.2
)


# ---------------------------------------------------------------------------
# Data, loaded through mlx-graphs and shared as NumPy arrays
# ---------------------------------------------------------------------------


def load_qm9(data_dir, sizes):
    from mlx_graphs.datasets import QM9

    dataset = QM9(base_dir=data_dir)
    perm = np.random.RandomState(0).permutation(len(dataset))
    n_train, n_val, n_test = sizes
    splits = {
        "train": perm[:n_train],
        "val": perm[n_train : n_train + n_val],
        "test": perm[n_train + n_val : n_train + n_val + n_test],
    }
    out = {}
    for name, idx in splits.items():
        out[name] = [
            dict(
                z=np.array(dataset[int(i)].z, dtype=np.int64),
                pos=np.array(dataset[int(i)].pos, dtype=np.float32),
                y=float(np.array(dataset[int(i)].graph_labels)[0, QM9_TARGET]),
            )
            for i in idx
        ]
    return out


def one_hot_features(values, sizes):
    """Concatenated one-hot encodings of integer categorical columns."""
    offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    out = np.zeros((values.shape[0], int(np.sum(sizes))), dtype=np.float32)
    rows = np.arange(values.shape[0])
    for column, offset in enumerate(offsets):
        out[rows, offset + values[:, column]] = 1.0
    return out


def load_esol(data_dir, seed):
    from mlx_graphs.datasets import MoleculeNet
    from mlx_graphs.utils import e_map, x_map

    dataset = MoleculeNet("ESOL", base_dir=data_dir)
    x_sizes = [len(v) for v in x_map.values()]
    e_sizes = [len(v) for v in e_map.values()]
    graphs = []
    for g in dataset:
        graphs.append(
            dict(
                x=one_hot_features(np.array(g.node_features), x_sizes),
                edge_index=np.array(g.edge_index, dtype=np.int64),
                edge_attr=one_hot_features(np.array(g.edge_features), e_sizes),
                y=float(np.array(g.graph_labels).reshape(-1)[0]),
            )
        )
    perm = np.random.RandomState(seed).permutation(len(graphs))
    n_val = n_test = len(graphs) // 10
    return {
        "test": [graphs[i] for i in perm[:n_test]],
        "val": [graphs[i] for i in perm[n_test : n_test + n_val]],
        "train": [graphs[i] for i in perm[n_test + n_val :]],
    }


def batches(n, batch_size, rng=None):
    order = np.arange(n) if rng is None else rng.permutation(n)
    return [order[i : i + batch_size] for i in range(0, n, batch_size)]


# ---------------------------------------------------------------------------
# PyG without pyg-lib / torch-cluster / torch-sparse
# ---------------------------------------------------------------------------


def torch_radius_graph(pos, r, batch=None, loop=False, max_num_neighbors=32, **_):
    import torch

    if batch is None:
        batch = torch.zeros(pos.size(0), dtype=torch.long)
    dist = torch.cdist(pos, pos)
    mask = (dist < r) & (batch[:, None] == batch[None, :])
    mask.fill_diagonal_(loop)
    center, neighbor = mask.nonzero(as_tuple=True)
    return torch.stack([neighbor, center])


def torch_triplets(edge_index, num_nodes):
    import torch

    row, col = edge_index  # j -> i
    order = torch.argsort(col * num_nodes + row)
    counts = torch.bincount(col, minlength=num_nodes)
    ptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
    num_triplets = counts[row]
    total = int(num_triplets.sum())
    first = torch.repeat_interleave(num_triplets.cumsum(0) - num_triplets, num_triplets)
    position = (
        torch.repeat_interleave(ptr[row], num_triplets) + torch.arange(total) - first
    )
    idx_kj = order[position]
    idx_ji = torch.repeat_interleave(torch.arange(row.size(0)), num_triplets)
    idx_i, idx_j, idx_k = col[idx_ji], row[idx_ji], row[idx_kj]
    mask = idx_i != idx_k
    return (col, row, idx_i[mask], idx_j[mask], idx_k[mask], idx_kj[mask], idx_ji[mask])


def patch_pyg():
    import torch_geometric.nn.models.dimenet as dimenet
    import torch_geometric.typing

    if not torch_geometric.typing.WITH_RADIUS:
        dimenet.radius_graph = torch_radius_graph
    if not torch_geometric.typing.WITH_TORCH_SPARSE:
        dimenet.triplets = torch_triplets


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------


class Selector:
    """Keeps the test metric of the epoch with the best validation metric
    (lower is better)."""

    def __init__(self):
        self.best = None
        self.curve = []

    def add(self, epoch, val, test, train_loss):
        self.curve.append(dict(epoch=epoch, val=val, test=test, loss=train_loss))
        if self.best is None or val < self.best[1]:
            self.best = (epoch, val, test)


def run_mlx(task, cfg, data, seed, stats):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_map

    from mlx_graphs.data import GraphData
    from mlx_graphs.data.batch import GraphDataBatch
    from mlx_graphs.nn import AttentiveFP, DimeNetPlusPlus, SchNet

    mx.random.seed(seed)
    mean, std = stats
    if task == "schnet":
        model = SchNet(**SCHNET)
    elif task == "dimenet++":
        model = DimeNetPlusPlus(**DIMENET_PP)
    else:
        in_dim = data["train"][0]["x"].shape[1]
        edge_dim = data["train"][0]["edge_attr"].shape[1]
        model = AttentiveFP(in_dim, edge_dim=edge_dim, **ATTENTIVE_FP)
    mx.eval(model.parameters())

    def to_graph(g):
        if task == "attentivefp":
            return GraphData(
                edge_index=mx.array(g["edge_index"].astype(np.int32)),
                node_features=mx.array(g["x"]),
                edge_features=mx.array(g["edge_attr"]),
                graph_labels=mx.array([[g["y"]]], dtype=mx.float32),
            )
        return GraphData(
            edge_index=mx.zeros((2, 0), dtype=mx.int32),
            z=mx.array(g["z"].astype(np.int32)),
            pos=mx.array(g["pos"]),
            graph_labels=mx.array([[g["y"]]], dtype=mx.float32),
            num_nodes=int(g["z"].size),
        )

    graphs = {name: [to_graph(g) for g in split] for name, split in data.items()}

    def forward(model, b):
        if task == "attentivefp":
            return model(
                b.node_features, b.edge_index, b.edge_features, b.batch_indices
            )
        return model(b.z, b.pos, b.batch_indices)

    def loss_fn(model, b):
        pred = forward(model, b)
        target = (b.graph_labels - mean) / std
        if task == "attentivefp":
            return nn.losses.mse_loss(pred, target)
        return nn.losses.l1_loss(pred, target)

    optimizer = optim.Adam(learning_rate=cfg["lr"], bias_correction=True)
    weight_decay = cfg.get("weight_decay", 0.0)
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    def evaluate(split):
        model.eval()
        errors = []
        for idx in batches(len(graphs[split]), 256):
            b = GraphDataBatch([graphs[split][i] for i in idx])
            pred = forward(model, b) * std + mean
            errors.append(np.array(pred - b.graph_labels).reshape(-1))
        model.train()
        errors = np.concatenate(errors)
        if task == "attentivefp":
            return float(np.sqrt(np.mean(errors**2)))
        return float(np.mean(np.abs(errors)))

    rng = np.random.RandomState(seed)
    selector, times = Selector(), []
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        start, total, count = time.perf_counter(), 0.0, 0
        for idx in batches(len(graphs["train"]), cfg["batch"], rng):
            b = GraphDataBatch([graphs["train"][i] for i in idx])
            loss, grads = loss_and_grad(model, b)
            if weight_decay:
                grads = tree_map(
                    lambda g, p: g + weight_decay * p, grads, model.parameters()
                )
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)
            total += loss.item() * len(idx)
            count += len(idx)
        times.append(time.perf_counter() - start)
        selector.add(epoch, evaluate("val"), evaluate("test"), total / count)
    return selector, times


def run_pyg(task, cfg, data, seed, stats, threads):
    import torch
    import torch.nn.functional as F
    from torch_geometric.data import Batch, Data
    from torch_geometric.nn.models import AttentiveFP, DimeNetPlusPlus, SchNet

    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    patch_pyg()
    mean, std = stats
    if task == "schnet":
        import torch_geometric.typing

        graph_fn = None
        if not torch_geometric.typing.WITH_RADIUS:

            def graph_fn(pos, batch):
                edge_index = torch_radius_graph(pos, SCHNET["cutoff"], batch)
                row, col = edge_index
                return edge_index, (pos[row] - pos[col]).norm(dim=-1)

        model = SchNet(interaction_graph=graph_fn, **SCHNET)
    elif task == "dimenet++":
        model = DimeNetPlusPlus(**DIMENET_PP)
    else:
        in_dim = data["train"][0]["x"].shape[1]
        edge_dim = data["train"][0]["edge_attr"].shape[1]
        model = AttentiveFP(in_dim, edge_dim=edge_dim, **ATTENTIVE_FP)

    def to_data(g):
        y = torch.tensor([[g["y"]]], dtype=torch.float)
        if task == "attentivefp":
            return Data(
                x=torch.from_numpy(g["x"]),
                edge_index=torch.from_numpy(g["edge_index"]),
                edge_attr=torch.from_numpy(g["edge_attr"]),
                y=y,
            )
        return Data(
            z=torch.from_numpy(g["z"]),
            pos=torch.from_numpy(g["pos"]),
            y=y,
            num_nodes=int(g["z"].size),
        )

    graphs = {name: [to_data(g) for g in split] for name, split in data.items()}

    def forward(b):
        if task == "attentivefp":
            return model(b.x, b.edge_index, b.edge_attr, b.batch)
        return model(b.z, b.pos, b.batch)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0)
    )

    @torch.no_grad()
    def evaluate(split):
        model.eval()
        errors = []
        for idx in batches(len(graphs[split]), 256):
            b = Batch.from_data_list([graphs[split][i] for i in idx])
            pred = forward(b) * std + mean
            errors.append((pred - b.y).view(-1).numpy())
        errors = np.concatenate(errors)
        if task == "attentivefp":
            return float(np.sqrt(np.mean(errors**2)))
        return float(np.mean(np.abs(errors)))

    rng = np.random.RandomState(seed)
    selector, times = Selector(), []
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        start, total, count = time.perf_counter(), 0.0, 0
        for idx in batches(len(graphs["train"]), cfg["batch"], rng):
            b = Batch.from_data_list([graphs["train"][i] for i in idx])
            optimizer.zero_grad()
            pred = forward(b)
            target = (b.y - mean) / std
            if task == "attentivefp":
                loss = F.mse_loss(pred, target)
            else:
                loss = F.l1_loss(pred, target)
            loss.backward()
            optimizer.step()
            total += float(loss) * len(idx)
            count += len(idx)
        times.append(time.perf_counter() - start)
        selector.add(epoch, evaluate("val"), evaluate("test"), total / count)
    return selector, times


def worker(args):
    cfg = dict(TASKS[args.task])
    if args.epochs:  # quick runs
        cfg["epochs"] = args.epochs
    if cfg["dataset"] == "QM9":
        data = load_qm9(args.data_dir, cfg["sizes"])
    else:
        data = load_esol(args.data_dir, args.seed)
    ys = np.array([g["y"] for g in data["train"]])
    stats = (float(ys.mean()), float(ys.std()))
    if args.framework == "mlx":
        selector, times = run_mlx(args.task, cfg, data, args.seed, stats)
    else:
        selector, times = run_pyg(
            args.task, cfg, data, args.seed, stats, args.torch_threads
        )
    epoch, val, test = selector.best
    print(
        json.dumps(
            dict(
                task=args.task,
                framework=args.framework,
                seed=args.seed,
                best_epoch=epoch,
                val=val,
                test=test,
                # the first epoch includes warm-up (e.g. kernel compilation)
                epoch_time=float(np.median(times[1:] if len(times) > 1 else times)),
                first_epoch_time=times[0],
                curve=selector.curve,
            )
        )
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def load_results():
    if not os.path.exists(RESULTS):
        return []
    with open(RESULTS) as f:
        return [json.loads(line) for line in f if line.strip()]


METRICS = {
    "schnet": ("QM9 (gap)", "test MAE (eV)"),
    "dimenet++": ("QM9 (gap)", "test MAE (eV)"),
    "attentivefp": ("ESOL", "test RMSE (log mol/L)"),
}


def summarize():
    results = load_results()
    lines = [
        "# Training quality: molecular models",
        "",
        "mlx-graphs (MLX, GPU) vs PyTorch Geometric (torch, CPU), trained by",
        "`molecular_quality.py` with the same architectures, hyperparameters,",
        "data and mini-batch order. Test metric at the best validation epoch,",
        "mean ± std over seeds; time per epoch is the median training time of",
        "an epoch (the first one, with warm-up, excluded).",
        "",
        "| model | dataset | metric | PyG | mlx-graphs | PyG s/epoch "
        "| mlx-graphs s/epoch | seeds |",
        "|---|---|---|---|---|---|---|---|",
    ]

    def fmt(values):
        if len(values) == 0:
            return "-"
        if len(values) == 1:
            return f"{values[0]:.4f}"
        return f"{np.mean(values):.4f} ± {np.std(values, ddof=1):.4f}"

    for task, cfg in TASKS.items():
        rows = {
            fw: [r for r in results if r["task"] == task and r["framework"] == fw]
            for fw in ("pyg", "mlx")
        }
        if not any(rows.values()):
            continue
        dataset, metric = METRICS[task]
        times = {
            fw: np.mean([r["epoch_time"] for r in rs]) if rs else float("nan")
            for fw, rs in rows.items()
        }
        if cfg["sizes"]:
            dataset += " {}/{}/{}".format(*cfg["sizes"])
        lines.append(
            f"| {task} | {dataset}, {cfg['epochs']} epochs | {metric} "
            f"| {fmt([r['test'] for r in rows['pyg']])} "
            f"| {fmt([r['test'] for r in rows['mlx']])} "
            f"| {times['pyg']:.1f} | {times['mlx']:.1f} "
            f"| {len(rows['pyg'])}/{len(rows['mlx'])} |"
        )
    text = "\n".join(lines) + "\n"
    with open(SUMMARY, "w") as f:
        f.write(text)
    print(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--frameworks", nargs="+", default=["pyg", "mlx"])
    parser.add_argument("--data-dir", default=os.path.join(REPO, ".mlx_graphs_data"))
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--task", help=argparse.SUPPRESS)
    parser.add_argument("--framework", help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--epochs", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        return worker(args)
    if args.summary:
        return summarize()

    done = {(r["task"], r["framework"], r["seed"]) for r in load_results()}
    for task in args.tasks:
        for seed in range(TASKS[task]["seeds"]):
            for framework in args.frameworks:
                if (task, framework, seed) in done:
                    continue
                print(f"{task} {framework} seed={seed} ...", flush=True)
                cmd = [
                    sys.executable,
                    __file__,
                    "--worker",
                    "--task",
                    task,
                    "--framework",
                    framework,
                    "--seed",
                    str(seed),
                    "--data-dir",
                    args.data_dir,
                    "--torch-threads",
                    str(args.torch_threads),
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
                if proc.returncode != 0 or not lines:
                    print(proc.stdout[-2000:], proc.stderr[-4000:])
                    continue
                result = json.loads(lines[-1])
                print(
                    f"  test={result['test']:.4f} (epoch {result['best_epoch']}), "
                    f"{result['epoch_time']:.1f} s/epoch",
                    flush=True,
                )
                with open(RESULTS, "a") as f:
                    f.write(json.dumps(result) + "\n")
    summarize()


if __name__ == "__main__":
    main()
