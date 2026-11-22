"""Trains ONE node classification configuration in ONE framework, over
several seeds, and prints the result as a JSON line.

Called by ``node_quality.py``, in its own subprocess under a memory cap:

    python node_worker.py --config arma --framework mlx --dataset cora --seeds 0 1 2

``--framework`` is ``mlx`` (mlx-graphs on the GPU) or ``pyg`` (PyTorch
Geometric on the CPU). Both frameworks get exactly the same arrays (loaded
through PyG, as in ``benchmarks/pyg_comparison``), the same public split, the
same model (``node_models.py``) and the same hyperparameters
(``node_quality.CONFIGS``). Feature transforms (``NormalizeFeatures``,
``SIGN``, ...) are applied by each framework's own implementation.
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "pyg_comparison"))

import node_models  # noqa: E402
from node_quality import CONFIGS  # noqa: E402
from worker import load_node_dataset  # noqa: E402  (pyg_comparison/worker.py)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def graph_info(data, hp, num_features):
    src, dst = data["edge_index"]
    n = data["x"].shape[0]
    in_deg = np.bincount(dst, minlength=n)
    return node_models.namespace(
        in_dim=num_features,
        num_classes=data["num_classes"],
        num_nodes=n,
        edge_dim=2,
        deg=np.bincount(in_deg).astype(np.int64),  # in-degree histogram (PNA)
    )


def synthetic_edge_features(edge_index, num_nodes):
    """Two deterministic edge features: 1/sqrt(1 + in-degree) of both ends."""
    src, dst = edge_index
    inv = 1.0 / np.sqrt(1.0 + np.bincount(dst, minlength=num_nodes))
    return np.stack([inv[src], inv[dst]], axis=1).astype(np.float32)


def mlx_features(data, features, hp):
    """Node features after mlx-graphs' own transforms: a list of arrays (one
    for most models, K + 1 for SIGN)."""
    import mlx.core as mx

    import mlx_graphs.transforms as T
    from mlx_graphs.data import GraphData

    if features == "raw":
        return [mx.array(data["x"])]
    graph = GraphData(
        edge_index=mx.array(data["edge_index"].astype(np.int32)),
        node_features=mx.array(data["x"]),
    )
    if features == "normalize":
        graph = T.NormalizeFeatures()(graph)
        return [graph.node_features]
    if features == "sign":
        graph = T.SIGN(hp.K)(T.NormalizeFeatures()(graph))
        return [graph.node_features] + [
            getattr(graph, f"x{i}") for i in range(1, hp.K + 1)
        ]
    if features == "pe_ldp":
        graph = T.AddLaplacianEigenvectorPE(
            hp.pe_k, attr_name=None, is_undirected=True
        )(graph)
        graph = T.LocalDegreeProfile()(graph)
        return [graph.node_features]
    raise ValueError(features)


def pyg_features(data, features, hp):
    import torch
    import torch_geometric.transforms as T
    from torch_geometric.data import Data

    x = torch.from_numpy(data["x"])
    if features == "raw":
        return [x]
    graph = Data(x=x, edge_index=torch.from_numpy(data["edge_index"]))
    if features == "normalize":
        return [T.NormalizeFeatures()(graph).x]
    if features == "sign":
        graph = T.SIGN(hp.K)(T.NormalizeFeatures()(graph))
        return [graph.x] + [graph[f"x{i}"] for i in range(1, hp.K + 1)]
    if features == "pe_ldp":
        graph = T.AddLaplacianEigenvectorPE(
            hp.pe_k, attr_name=None, is_undirected=True
        )(graph)
        return [T.LocalDegreeProfile()(graph).x]
    raise ValueError(features)


# ---------------------------------------------------------------------------
# Framework adapters: the same training loop drives both
# ---------------------------------------------------------------------------


class MLXRun:
    def __init__(self, data, cfg, hp, seed):
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim

        self.mx, self.nn = mx, nn
        mx.random.seed(seed)
        n = data["x"].shape[0]
        xs = mlx_features(data, cfg["features"], hp)
        self.info = graph_info(data, hp, xs[0].shape[1])
        masks = {}
        for k in ("train", "val", "test"):
            m = np.zeros(n, dtype=bool)
            m[data[k]] = True
            masks[k] = mx.array(m)
        self.masks = masks
        self.idx = {k: mx.array(data[k]) for k in ("train", "val", "test")}
        self.g = {
            "x": xs[0],
            "xs": xs,
            "ei": mx.array(data["edge_index"].astype(np.int32)),
            "ea": mx.array(synthetic_edge_features(data["edge_index"], n)),
            "y": mx.array(data["y"].astype(np.int32)),
            "visible": masks["train"],
        }
        mx.eval(self.g, self.idx, self.masks)
        self.model = node_models.MLX[cfg["model"]](self.info, hp)
        mx.eval(self.model.parameters())
        # bias_correction=True: the update rule of torch.optim.Adam
        self.opt = optim.Adam(learning_rate=hp.lr, bias_correction=True)
        self.wd, self.wd_conv = hp.wd, getattr(hp, "wd_conv", None)
        self.loss_and_grad = nn.value_and_grad(self.model, self._loss)
        self.loss_nodes = self.idx["train"]

    def _loss(self, model):
        logits = model(self.g)
        idx = self.loss_nodes
        return self.nn.losses.cross_entropy(
            logits[idx], self.g["y"][idx], reduction="mean"
        )

    def _weight_decay(self, grads):
        """L2 penalty added to the gradients: torch.optim.Adam's weight_decay."""
        from mlx.utils import tree_flatten, tree_map_with_path

        params = dict(tree_flatten(self.model.parameters()))

        def decay(name, grad):
            wd = self.wd
            if self.wd_conv is not None and name.startswith("convs."):
                wd = self.wd_conv
            return grad + wd * params[name] if wd else grad

        return tree_map_with_path(decay, grads)

    def set_visible(self, ratio):
        """MaskLabel: a random part of the training labels is visible to
        the model, the loss is taken on the other training nodes."""
        from mlx_graphs.nn import MaskLabel

        mx = self.mx
        visible = MaskLabel.ratio_mask(self.masks["train"], ratio)
        self.g["visible"] = visible
        hidden = np.nonzero(np.array(self.masks["train"] & ~visible))[0]
        self.loss_nodes = mx.array(hidden)

    def reset_visible(self):
        self.g["visible"] = self.masks["train"]
        self.loss_nodes = self.idx["train"]

    def train_step(self):
        self.model.train()
        loss, grads = self.loss_and_grad(self.model)
        if self.wd or self.wd_conv:
            grads = self._weight_decay(grads)
        self.opt.update(self.model, grads)
        self.mx.eval(loss, self.model.parameters(), self.opt.state)
        return loss.item()

    def logits(self):
        self.model.eval()
        out = self.model(self.g)
        self.mx.eval(out)
        return out

    def accuracies(self, logits):
        pred = self.mx.argmax(logits, axis=1)
        y = self.g["y"]
        return [
            (pred[self.idx[k]] == y[self.idx[k]]).astype(self.mx.float32).mean().item()
            for k in ("train", "val", "test")
        ]

    def softmax(self, logits):
        return self.mx.softmax(logits, axis=-1)

    def accuracy_of(self, probs, split):
        pred = self.mx.argmax(probs, axis=1)
        idx = self.idx[split]
        return (pred[idx] == self.g["y"][idx]).astype(self.mx.float32).mean().item()

    def label_propagation(self, num_layers, alpha):
        from mlx_graphs.nn import LabelPropagation

        lp = LabelPropagation(num_layers, alpha)
        return lp(self.g["ei"], self.g["y"], mask=self.masks["train"])

    def correct_and_smooth(self, y_soft, kw):
        from mlx_graphs.nn import CorrectAndSmooth

        post = CorrectAndSmooth(**kw)
        tr = self.idx["train"]
        y_train = self.g["y"][tr]
        y = post.correct(self.g["ei"], y_soft, y_train, tr)
        return post.smooth(self.g["ei"], y, y_train, tr)


class PyGRun:
    def __init__(self, data, cfg, hp, seed, device="cpu"):
        import torch
        import torch.nn.functional as F

        self.torch, self.F = torch, F
        torch.manual_seed(seed)
        dev = torch.device(device)
        n = data["x"].shape[0]
        xs = [x.to(dev) for x in pyg_features(data, cfg["features"], hp)]
        self.info = graph_info(data, hp, xs[0].shape[1])
        self.idx = {
            k: torch.from_numpy(data[k]).to(dev) for k in ("train", "val", "test")
        }
        self.masks = {}
        for k in ("train", "val", "test"):
            m = torch.zeros(n, dtype=torch.bool, device=dev)
            m[self.idx[k]] = True
            self.masks[k] = m
        ea = synthetic_edge_features(data["edge_index"], n)
        self.g = {
            "x": xs[0],
            "xs": xs,
            "ei": torch.from_numpy(data["edge_index"]).to(dev),
            "ea": torch.from_numpy(ea).to(dev),
            "y": torch.from_numpy(data["y"]).to(dev),
            "visible": self.masks["train"],
        }
        self.model = node_models.PYG[cfg["model"]](self.info, hp).to(dev)
        wd_conv = getattr(hp, "wd_conv", None)
        if wd_conv is None:
            groups = [{"params": list(self.model.parameters()), "weight_decay": hp.wd}]
        else:
            named = list(self.model.named_parameters())
            groups = [
                {
                    "params": [p for k, p in named if k.startswith("convs.")],
                    "weight_decay": wd_conv,
                },
                {
                    "params": [p for k, p in named if not k.startswith("convs.")],
                    "weight_decay": hp.wd,
                },
            ]
        self.opt = torch.optim.Adam(groups, lr=hp.lr)
        self.loss_nodes = self.idx["train"]

    def set_visible(self, ratio):
        from torch_geometric.nn import MaskLabel

        visible = MaskLabel.ratio_mask(self.masks["train"], ratio)
        self.g["visible"] = visible
        self.loss_nodes = (self.masks["train"] & ~visible).nonzero().view(-1)

    def reset_visible(self):
        self.g["visible"] = self.masks["train"]
        self.loss_nodes = self.idx["train"]

    def train_step(self):
        self.model.train()
        self.opt.zero_grad()
        logits = self.model(self.g)
        idx = self.loss_nodes
        loss = self.F.cross_entropy(logits[idx], self.g["y"][idx])
        loss.backward()
        self.opt.step()
        return float(loss.item())

    def logits(self):
        self.model.eval()
        with self.torch.no_grad():
            return self.model(self.g)

    def accuracies(self, logits):
        pred = logits.argmax(dim=1)
        y = self.g["y"]
        return [
            (pred[self.idx[k]] == y[self.idx[k]]).float().mean().item()
            for k in ("train", "val", "test")
        ]

    def softmax(self, logits):
        return logits.softmax(dim=-1)

    def accuracy_of(self, probs, split):
        idx = self.idx[split]
        return (probs.argmax(dim=1)[idx] == self.g["y"][idx]).float().mean().item()

    def label_propagation(self, num_layers, alpha):
        from torch_geometric.nn import LabelPropagation

        lp = LabelPropagation(num_layers, alpha)
        return lp(self.g["y"], self.g["ei"], mask=self.masks["train"])

    def correct_and_smooth(self, y_soft, kw):
        from torch_geometric.nn import CorrectAndSmooth

        post = CorrectAndSmooth(**kw)
        tr = self.masks["train"]
        y_train = self.g["y"][tr]
        y = post.correct(y_soft, y_train, tr, self.g["ei"])
        return post.smooth(y, y_train, tr, self.g["ei"])


# ---------------------------------------------------------------------------
# One seed
# ---------------------------------------------------------------------------


def run_seed(data, cfg, hp, framework, seed, device="cpu") -> dict:
    if framework == "mlx":
        run = MLXRun(data, cfg, hp, seed)
    else:
        run = PyGRun(data, cfg, hp, seed, device)
    kind = cfg.get("kind", "train")
    start = time.perf_counter()

    if kind == "label_prop":
        out = run.label_propagation(hp.num_layers, hp.alpha)
        return {
            "val_acc": run.accuracy_of(out, "val"),
            "test_acc": run.accuracy_of(out, "test"),
            "seconds": time.perf_counter() - start,
        }

    losses, best = [], None
    for epoch in range(1, hp.epochs + 1):
        if kind == "masklabel":
            run.set_visible(hp.mask_ratio)
        losses.append(run.train_step())
        if kind == "masklabel":
            run.reset_visible()
        if epoch % hp.eval_every and epoch != hp.epochs:
            continue
        logits = run.logits()
        train_acc, val_acc, test_acc = run.accuracies(logits)
        if best is None or val_acc > best["val_acc"]:  # earliest on ties
            best = {
                "best_epoch": epoch,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "test_acc": test_acc,
            }
            if kind == "correct_and_smooth":
                best_probs = run.softmax(logits)

    result = {
        **best,
        "loss_curve": [round(v, 5) for v in losses],
        "final_train_loss": float(np.mean(losses[-10:])),
        "seconds": time.perf_counter() - start,
    }
    if kind == "correct_and_smooth":
        out = run.correct_and_smooth(best_probs, hp.cs)
        result["base_test_acc"] = result["test_acc"]
        result["base_val_acc"] = result["val_acc"]
        result["val_acc"] = run.accuracy_of(out, "val")
        result["test_acc"] = run.accuracy_of(out, "test")
    return result


# ---------------------------------------------------------------------------
# Admission control, shared by every worker of the harness: at most
# NODE_QUALITY_SLOTS workers (10) train at once, and a worker only starts
# when the machine has NODE_QUALITY_MIN_FREE_GB (3 GB) of unused memory.
# A waiting worker is a bare Python process (no torch / MLX imported yet).
# ---------------------------------------------------------------------------

SLOT_DIR = os.path.join(
    os.environ.get("TMPDIR", "/tmp"), "mlx_graphs_node_quality_slots"
)
_slot = None  # the lock file held for the life of the process


def unused_memory_gb():
    """Unused physical memory, from ``top -l 1 -n 0`` ("PhysMem: ...,
    1011M unused.")."""
    import re
    import subprocess

    out = subprocess.run(
        ["top", "-l", "1", "-n", "0"], capture_output=True, text=True
    ).stdout
    match = re.search(r"([\d.]+)([KMGT]) unused", out)
    if match is None:
        return float("inf")
    scale = {"K": 1 / 1024**2, "M": 1 / 1024, "G": 1.0, "T": 1024.0}
    return float(match.group(1)) * scale[match.group(2)]


# Configurations whose workers can take more than ~1.5 GB: at most
# NODE_QUALITY_HEAVY_SLOTS (2) of them run at once.
HEAVY_CONFIGS = {"pna_conv", "pna", "gcn2", "deepgcn"}
# Footprint assumed for a worker that has not allocated yet, in GB.
EXPECTED_GB = {"heavy": 3.5, "light": 1.0}


def workers_footprint_gb():
    """Total physical footprint (incl. Metal memory) of the running workers
    of this harness, other than this process."""
    import subprocess

    sys.path.insert(0, os.path.join(HERE, "..", "pyg_comparison"))
    from common import footprint_gb

    out = subprocess.run(
        ["pgrep", "-f", "node_worker.py"], capture_output=True, text=True
    ).stdout
    pids = [int(pid) for pid in out.split() if int(pid) != os.getpid()]
    return sum(footprint_gb(pid) for pid in pids)


def _try_lock(path):
    import fcntl

    f = open(path, "w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return f
    except OSError:
        f.close()
        return None


def _first_free(prefix, count):
    for i in range(count):
        f = _try_lock(os.path.join(SLOT_DIR, f"{prefix}{i}.lock"))
        if f is not None:
            return f
    return None


def wait_for_slot(config):
    """Blocks until this worker may start: it holds one of the
    NODE_QUALITY_SLOTS slots (and, for a heavy configuration, one of the
    NODE_QUALITY_HEAVY_SLOTS heavy slots), the machine has
    NODE_QUALITY_MIN_FREE_GB of unused memory, and the workers' total
    footprint stays under NODE_QUALITY_BUDGET_GB."""
    import fcntl

    global _slot
    slots = int(os.environ.get("NODE_QUALITY_SLOTS", "10"))
    heavy_slots = int(os.environ.get("NODE_QUALITY_HEAVY_SLOTS", "2"))
    min_free = float(os.environ.get("NODE_QUALITY_MIN_FREE_GB", "3"))
    budget = float(os.environ.get("NODE_QUALITY_BUDGET_GB", "12"))
    heavy = config in HEAVY_CONFIGS
    expected = EXPECTED_GB["heavy" if heavy else "light"]
    os.makedirs(SLOT_DIR, exist_ok=True)
    while True:
        # `touch $SLOT_DIR/drain` makes waiting and new workers give up (the
        # runner records them as failed, so a later run redoes them), e.g.
        # to restart the runner with fewer workers without losing work.
        if os.path.exists(os.path.join(SLOT_DIR, "drain")):
            sys.exit("drained: not started")
        held = [_first_free("slot", slots)]
        if heavy and held[0] is not None:
            held.append(_first_free("heavy", heavy_slots))
        if all(f is not None for f in held):
            # one launch at a time, so that two workers do not both see the
            # same free memory before either has allocated anything
            with open(os.path.join(SLOT_DIR, "launch.lock"), "w") as launch:
                fcntl.flock(launch, fcntl.LOCK_EX)
                if (
                    unused_memory_gb() >= min_free
                    and workers_footprint_gb() + expected <= budget
                ):
                    _slot = held
                    time.sleep(5)  # let this worker allocate first
                    return
        for f in held:
            if f is not None:
                f.close()
        time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, choices=sorted(CONFIGS))
    parser.add_argument("--framework", required=True, choices=["mlx", "pyg"])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "mps"], help="PyG's device"
    )
    parser.add_argument("--threads", type=int, default=1, help="PyG's CPU threads")
    args = parser.parse_args()
    wait_for_slot(args.config)

    cfg = CONFIGS[args.config]
    hp = node_models.namespace(**cfg["hp"])
    for key, value in cfg.get("per_dataset", {}).get(args.dataset, {}).items():
        setattr(hp, key, value)

    if args.framework == "mlx":
        import mlx.core as mx

        mx.set_cache_limit(2 * 1024**3)
    else:
        import torch

        torch.set_num_threads(args.threads)

    data = load_node_dataset(args.dataset, 0)
    runs = [
        {"seed": s, **run_seed(data, cfg, hp, args.framework, s, args.device)}
        for s in args.seeds
    ]
    print(
        json.dumps(
            {
                "config": args.config,
                "framework": args.framework,
                "dataset": args.dataset,
                "seeds": args.seeds,
                "device": "gpu" if args.framework == "mlx" else args.device,
                "hp": vars(hp),
                "runs": runs,
            }
        )
    )


if __name__ == "__main__":
    main()
