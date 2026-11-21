"""Training comparison of ``to_hetero`` models on DBLP: mlx-graphs vs PyG.

Trains ``to_hetero(GraphSAGE)`` and ``to_hetero(GAT)`` for author
classification on DBLP (full batch) in both frameworks with the same data,
architecture and optimizer (AdamW), for several seeds, and reports the test
accuracy at the best validation epoch. PyG runs on the CPU. Mini-batch runs
(``NeighborLoader``, ``HGTLoader``) are mlx-graphs only, since PyG's
heterogeneous samplers need pyg-lib or torch-sparse.

Every run is a separate process killed if its memory footprint exceeds
``--memory-gb`` (read with ``top``, which includes Metal allocations).

    python benchmarks/to_hetero/compare_dblp.py --data_dir .mlx_graphs_data
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
_UNITS = {"B": 1 / 1024**3, "K": 1 / 1024**2, "M": 1 / 1024, "G": 1.0, "T": 1024.0}


def footprint_gb(pid: int) -> float:
    out = subprocess.run(
        ["top", "-l", "1", "-pid", str(pid), "-stats", "mem"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    value = out.splitlines()[-1].strip().rstrip("+-") if out else ""
    try:
        return float(value[:-1]) * _UNITS[value[-1]]
    except (ValueError, KeyError, IndexError):
        return 0.0


def load_example():
    path = os.path.join(REPO, "examples", "to_hetero_dblp.py")
    spec = importlib.util.spec_from_file_location("to_hetero_dblp", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_mlx(args) -> dict:
    example = load_example()
    options = example.get_parser().parse_args(
        [
            "--model",
            args.model,
            "--loader",
            args.loader,
            "--epochs",
            str(args.epochs),
            "--seed",
            str(args.seed),
            "--log_every",
            str(args.epochs),
        ]
        + (["--data_dir", args.data_dir] if args.data_dir else [])
    )
    return example.main(options)


def run_pyg(args) -> dict:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GAT, GraphSAGE, to_hetero

    from mlx_graphs.datasets import DBLP

    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    data = DBLP(base_dir=args.data_dir)[0]
    num_nodes = data.num_nodes
    x_dict = {
        t: torch.from_numpy(np.array(v)) for t, v in data.node_features_dict.items()
    }
    for t, n in num_nodes.items():
        x_dict.setdefault(t, torch.ones(n, 1))
    edge_index_dict = {
        et: torch.from_numpy(np.array(v)).long()
        for et, v in data.edge_index_dict.items()
    }
    y = torch.from_numpy(np.array(data.node_labels_dict["author"])).long()
    masks = [
        torch.from_numpy(np.array(getattr(data, f"author_{s}_mask")))
        for s in ("train", "val", "test")
    ]
    metadata = (list(num_nodes), list(edge_index_dict))

    if args.model == "sage":
        model = GraphSAGE((-1, -1), 64, 2, out_channels=4, dropout=0.5)
    else:
        model = GAT(
            (-1, -1), 64, 2, out_channels=4, heads=8, dropout=0.5, add_self_loops=False
        )
    model = to_hetero(model, metadata, aggr="sum")
    with torch.no_grad():
        model(x_dict, edge_index_dict)  # initializes the lazy layers
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.001)

    best_val, best_state, times = -1.0, None, []
    for _ in range(args.epochs):
        start = time.time()
        model.train()
        optimizer.zero_grad()
        out = model(x_dict, edge_index_dict)["author"]
        loss = F.cross_entropy(out[masks[0]], y[masks[0]])
        loss.backward()
        optimizer.step()
        times.append(time.time() - start)
        model.eval()
        with torch.no_grad():
            pred = model(x_dict, edge_index_dict)["author"].argmax(-1)
        val_acc = (pred[masks[1]] == y[masks[1]]).float().mean().item()
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(x_dict, edge_index_dict)["author"].argmax(-1)
    test_acc = (pred[masks[2]] == y[masks[2]]).float().mean().item()
    return {
        "val_acc": best_val,
        "test_acc": test_acc,
        "epoch_time": float(np.median(times[1:])),
    }


def worker(args):
    result = run_mlx(args) if args.framework == "mlx" else run_pyg(args)
    print(json.dumps(result))


def unused_memory_gb() -> float:
    out = subprocess.run(["top", "-l", "1", "-n", "0"], capture_output=True, text=True)
    for line in out.stdout.splitlines():
        if line.startswith("PhysMem"):
            value = line.split(",")[-1].strip().split()[0]
            return float(value[:-1]) * _UNITS[value[-1]]
    return 0.0


def driver(args):
    configs = [
        ("mlx", "sage", "full"),
        ("pyg", "sage", "full"),
        ("mlx", "gat", "full"),
        ("pyg", "gat", "full"),
        ("mlx", "sage", "neighbor"),
        ("mlx", "sage", "hgt"),
    ]
    results = {}
    for framework, model, loader in configs:
        for seed in args.seeds:
            while unused_memory_gb() < args.min_free_gb:
                print("waiting for free memory...", file=sys.stderr)
                time.sleep(30)
            epochs = args.epochs if loader == "full" else args.minibatch_epochs
            cmd = [sys.executable, __file__, "--worker", "--framework", framework]
            cmd += ["--model", model, "--loader", loader, "--seed", str(seed)]
            cmd += ["--epochs", str(epochs)]
            if args.data_dir:
                cmd += ["--data_dir", args.data_dir]
            env = dict(os.environ, PYTHONPATH=REPO)
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env
            )
            peak = 0.0
            while proc.poll() is None:
                peak = max(peak, footprint_gb(proc.pid))
                if peak > args.memory_gb:
                    proc.kill()
                    break
                time.sleep(0.5)
            out, err = proc.communicate()
            key = f"{framework}/{model}/{loader}"
            if proc.returncode != 0:
                print(f"{key} seed {seed}: failed ({err.strip()[-300:]})")
                continue
            result = json.loads(out.strip().splitlines()[-1])
            result["peak_gb"] = peak
            results.setdefault(key, []).append(result)
            print(f"{key} seed {seed}: {result}", flush=True)

    print("\n| framework | model | training | test accuracy | epoch time |")
    print("|---|---|---|---|---|")
    for key, runs in results.items():
        acc = [r["test_acc"] for r in runs]
        t = [r["epoch_time"] for r in runs]
        mean, std = (
            sum(acc) / len(acc),
            (sum((a - sum(acc) / len(acc)) ** 2 for a in acc) / len(acc)) ** 0.5,
        )
        framework, model, loader = key.split("/")
        print(
            f"| {framework} | {model} | {loader} | {mean:.4f} ± {std:.4f} "
            f"({len(acc)} seeds) | {1000 * sum(t) / len(t):.0f} ms |"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--framework", choices=["mlx", "pyg"], default="mlx")
    parser.add_argument("--model", choices=["sage", "gat"], default="sage")
    parser.add_argument("--loader", choices=["full", "neighbor", "hgt"], default="full")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--minibatch_epochs", type=int, default=20)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--memory-gb", type=float, default=4.0)
    parser.add_argument("--min-free-gb", type=float, default=4.0)
    args = parser.parse_args()
    worker(args) if args.worker else driver(args)
