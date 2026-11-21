"""Statistical and training checks of mlx-graphs' samplers and loaders.

PyG's own samplers need compiled extensions (pyg-lib, torch_sparse,
torch_cluster); these checks compare against ground truth that does not:

- ``saint``: GraphSAINT normalization. For every sampler, the expectation over
  fresh subgraphs of the ``edge_norm``-weighted aggregation of a sampled node
  is compared with its full-graph aggregation (library normalization, which
  divides the count of the aggregating node, vs PyG's, which divides the count
  of the source node), and the ``node_norm``-weighted loss with the full-graph
  mean loss. Also compares the edge sampler's probabilities (paper's
  ``1/deg(u) + 1/deg(v)`` vs PyG's ``deg(u) + deg(v)``) by the variance of the
  estimator.
- ``stats``: ImbalancedSampler class balance, DynamicBatchSampler budgets and
  coverage, epoch coverage of the seeds of every loader, and uniformity of the
  neighbor sampling (NeighborLoader and ShaDowKHopSampler).
- ``train``: a 2-layer GraphSAGE trained full-batch (mlx-graphs and PyG) and
  with NeighborLoader, GraphSAINTRandomWalkSampler, ClusterLoader and
  ShaDowKHopSampler (node classification), and with LinkNeighborLoader vs
  full-batch (link prediction, AUC).

Run it under a memory cap, from the repository root::

    python benchmarks/run_capped.py --gb 6 -- \\
        python benchmarks/quality/sampler_quality.py --data-dir .mlx_graphs_data

Results are printed as markdown tables (see ``sampler_results.md``).
"""

import argparse
import contextlib
import io
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import mlx.optimizers as optim  # noqa: E402

from mlx_graphs.data import GraphData  # noqa: E402
from mlx_graphs.loaders import (  # noqa: E402
    ClusterData,
    ClusterLoader,
    DynamicBatchSampler,
    GraphSAINTEdgeSampler,
    GraphSAINTNodeSampler,
    GraphSAINTRandomWalkSampler,
    ImbalancedSampler,
    LinkNeighborLoader,
    NeighborLoader,
    RandomNodeLoader,
    ShaDowKHopSampler,
)
from mlx_graphs.nn import SAGEConv  # noqa: E402


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    print("\n".join(lines) + "\n", flush=True)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_planetoid(data_dir, name):
    """Cora / CiteSeer / PubMed (public split) as numpy arrays, through PyG's
    pure-Python Planetoid loader (cached under ``<data_dir>/pyg``)."""
    from torch_geometric.datasets import Planetoid

    pyg_name = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}[name]
    with contextlib.redirect_stdout(io.StringIO()):
        data = Planetoid(os.path.join(data_dir, "pyg"), pyg_name)[0]
    return {
        "x": data.x.numpy().astype(np.float32),
        "edge_index": data.edge_index.numpy().astype(np.int64),
        "y": data.y.numpy().astype(np.int64),
        "train": np.nonzero(data.train_mask.numpy())[0],
        "val": np.nonzero(data.val_mask.numpy())[0],
        "test": np.nonzero(data.test_mask.numpy())[0],
    }


def to_graph(d, edge_index=None):
    n = d["x"].shape[0]
    g = GraphData(
        edge_index=mx.array(d["edge_index"] if edge_index is None else edge_index),
        node_features=mx.array(d["x"]),
        node_labels=mx.array(d["y"]),
    )
    mask = np.zeros(n, dtype=bool)
    mask[d["train"]] = True
    g.train_mask = mx.array(mask)
    return g


# ---------------------------------------------------------------------------
# GraphSAINT normalization
# ---------------------------------------------------------------------------


class _PyGProbEdgeSampler(GraphSAINTEdgeSampler):
    """Edge sampler with PyG's edge probabilities,
    ``in_degree(source) + out_degree(target)``."""

    def _sample_nodes(self):
        if self._edge_prob is None:
            adj = self._adj
            self._edge_prob = (adj.in_degree[adj.src] + adj.out_degree[adj.dst]).astype(
                np.float64
            )
        return super()._sample_nodes()


def _counts(sampler, num_samples):
    N, E = sampler.N, sampler.E
    node_count, edge_count = np.zeros(N), np.zeros(E)
    for _ in range(num_samples):
        n_id, _, e_id = sampler._sample()
        node_count[n_id] += 1
        edge_count[e_id] += 1
    return node_count, edge_count


def saint_estimates(sampler, x, loss, num_eval):
    """Monte-Carlo estimates of the normalized aggregation of every node
    (conditional on the node being sampled) and of the normalized loss."""
    adj = sampler._adj
    src, dst = adj.src, adj.dst
    with np.errstate(divide="ignore", invalid="ignore"):
        # PyG's normalization, for comparison: count(source) / count(edge)
        node_count, edge_count = _counts(sampler, num_eval)
        pyg_norm = np.nan_to_num(np.clip(node_count[src] / edge_count, 0, 1e4), nan=0.1)
    norms = {
        "library (target count)": sampler.edge_norm,
        "PyG (source count)": pyg_norm,
    }
    sums = {k: np.zeros(sampler.N) for k in norms}
    sq = {k: np.zeros(sampler.N) for k in norms}
    times = np.zeros(sampler.N)
    losses, sizes = [], []
    for _ in range(num_eval):
        n_id, _, e_id = sampler._sample()
        times[n_id] += 1
        sizes.append(n_id.size)
        for k, norm in norms.items():
            agg = np.bincount(
                dst[e_id], weights=norm[e_id] * x[src[e_id]], minlength=sampler.N
            )
            sums[k] += agg
            sq[k] += agg**2
        losses.append((sampler.node_norm[n_id] * loss[n_id]).sum())
    return norms, sums, sq, times, np.array(losses), np.mean(sizes)


def run_saint(data_dir, num_eval):
    print("## GraphSAINT normalization (Cora)\n")
    d = load_planetoid(data_dir, "cora")
    graph = to_graph(d)
    N = graph.num_nodes
    src, dst = d["edge_index"]
    rng = np.random.default_rng(0)
    x = rng.random(N) + 0.5
    full = np.bincount(dst, weights=x[src], minlength=N)
    loss = rng.random(N)

    samplers = {
        "Node (batch_size=1000)": lambda cov: GraphSAINTNodeSampler(
            graph, batch_size=1000, num_steps=10, sample_coverage=cov
        ),
        "Edge (batch_size=400)": lambda cov: GraphSAINTEdgeSampler(
            graph, batch_size=400, num_steps=10, sample_coverage=cov
        ),
        "RandomWalk (300 roots, length 2)": lambda cov: GraphSAINTRandomWalkSampler(
            graph, batch_size=300, walk_length=2, num_steps=10, sample_coverage=cov
        ),
    }
    rows, loss_rows = [], []
    for name, make in samplers.items():
        for coverage in (100, 1000):
            np.random.seed(0)
            sampler = make(coverage)
            _, sums, _, times, losses, size = saint_estimates(
                sampler, x, loss, num_eval
            )
            ok = (times >= 100) & (full > 0)
            for k in sums:
                if coverage != 100 and k.startswith("PyG"):
                    continue
                rel = np.abs(sums[k][ok] / times[ok] - full[ok]) / full[ok]
                rows.append(
                    [
                        name,
                        coverage,
                        k,
                        ok.sum(),
                        f"{np.mean(rel):.4f}",
                        f"{np.median(rel):.4f}",
                        f"{np.max(rel):.3f}",
                    ]
                )
            sem = losses.std() / np.sqrt(losses.size)
            loss_rows.append(
                [
                    name,
                    coverage,
                    f"{size:.0f}",
                    f"{losses.mean():.5f} ± {sem:.5f}",
                    f"{loss.mean():.5f}",
                ]
            )
    print(
        f"Relative error of E[normalized aggregation | node sampled] vs the "
        f"full-graph sum, over {num_eval} fresh subgraphs (nodes sampled >= 100 "
        f"times; the PyG normalization is computed from the counts of another "
        f"{num_eval} subgraphs):\n"
    )
    table(
        [
            "sampler",
            "sample_coverage",
            "edge_norm",
            "nodes",
            "mean rel. err",
            "median",
            "max",
        ],
        rows,
    )
    print(
        "Normalized loss `sum(node_norm * loss)` per subgraph vs full-graph mean "
        "loss (± standard error):\n"
    )
    table(
        [
            "sampler",
            "sample_coverage",
            "avg. subgraph nodes",
            "E[normalized loss]",
            "full-graph mean",
        ],
        loss_rows,
    )

    # Edge probabilities: paper vs PyG. GraphSAINT's edge probabilities
    # minimize the variance of the estimator of the (random-walk normalized)
    # aggregation summed over all nodes, sum_(u,v) x_u / deg(v), from the
    # sampled edges e weighted by 1 / P(e sampled), for a budget of edges.
    deg = np.maximum(np.bincount(dst, minlength=N), 1)
    contrib = x[src] / deg[dst]
    total = contrib.sum()

    def mean_nodes(cls, batch_size):
        np.random.seed(1)
        sampler = cls(graph, batch_size=batch_size)
        return np.mean([sampler._sample()[0].size for _ in range(30)])

    target = mean_nodes(GraphSAINTEdgeSampler, 400)
    pyg_batch = min(
        range(400, 1600, 20),
        key=lambda b: abs(mean_nodes(_PyGProbEdgeSampler, b) - target),
    )
    rows = []
    for name, cls, batch_size in (
        ("paper: 1/deg(u) + 1/deg(v)", GraphSAINTEdgeSampler, 400),
        ("PyG: deg(u) + deg(v)", _PyGProbEdgeSampler, 400),
        ("PyG: deg(u) + deg(v)", _PyGProbEdgeSampler, pyg_batch),
    ):
        np.random.seed(0)
        sampler = cls(graph, batch_size=batch_size)
        _, edge_count = _counts(sampler, num_eval)
        p = edge_count / num_eval
        estimates, sizes = [], []
        for _ in range(num_eval):
            n_id, _, e_id = sampler._sample()
            e_id = e_id[p[e_id] > 0]
            estimates.append((contrib[e_id] / p[e_id]).sum())
            sizes.append(n_id.size)
        estimates = np.array(estimates)
        rows.append(
            [
                name,
                batch_size,
                f"{np.mean(sizes):.0f}",
                f"{(estimates.mean() - total) / total:+.4f}",
                f"{estimates.std() / total:.4f}",
                f"{np.mean(edge_count == 0):.4f}",
            ]
        )
    print(
        "Edge sampler probabilities: one subgraph's estimate of the summed "
        "mean-aggregation (sampled edges weighted by 1 / their inclusion "
        f"probability, estimated from {num_eval} other subgraphs):\n"
    )
    table(
        [
            "edge probability",
            "edges / subgraph",
            "avg. subgraph nodes",
            "rel. bias",
            "rel. std",
            "edges never sampled",
        ],
        rows,
    )


# ---------------------------------------------------------------------------
# Statistical properties
# ---------------------------------------------------------------------------


def chi2_uniform(counts):
    from scipy.stats import chisquare

    return chisquare(counts).pvalue


def run_stats(data_dir):
    print("## Statistical properties\n")
    d = load_planetoid(data_dir, "cora")
    graph = to_graph(d)
    N = graph.num_nodes
    y = d["y"]

    # ImbalancedSampler
    rows = []
    rng = np.random.default_rng(0)
    # Strongly imbalanced subset: keep 100% of class 3, 5% of the others
    subset = np.nonzero((y == 3) | (rng.random(N) < 0.05))[0]
    for name, nodes in (("all nodes", None), ("imbalanced subset", subset)):
        np.random.seed(0)
        sampler = ImbalancedSampler(
            graph, input_nodes=None if nodes is None else mx.array(nodes)
        )
        labels = y if nodes is None else y[nodes]
        drawn = np.concatenate([labels[np.array(list(sampler))] for _ in range(20)])
        before = np.bincount(labels, minlength=7) / labels.size
        after = np.bincount(drawn, minlength=7) / drawn.size
        rows.append(
            [
                name,
                labels.size,
                f"{before.min():.3f}-{before.max():.3f}",
                f"{after.min():.3f}-{after.max():.3f}",
                f"{1 / 7:.3f}",
                f"{chi2_uniform(np.bincount(drawn, minlength=7)):.2f}",
            ]
        )
    print("ImbalancedSampler on Cora labels (20 epochs of draws):\n")
    table(
        [
            "input",
            "labels",
            "class freq. before",
            "class freq. drawn",
            "target",
            "chi2 p (uniform)",
        ],
        rows,
    )

    # DynamicBatchSampler
    from mlx_graphs.datasets import TUDataset

    with contextlib.redirect_stdout(io.StringIO()):
        dataset = TUDataset("PROTEINS", base_dir=data_dir)
    graphs = [dataset[i] for i in range(len(dataset))]
    sizes = {
        "node": np.array([g.num_nodes for g in graphs]),
        "edge": np.array([g.num_edges for g in graphs]),
    }
    rows = []
    for mode, max_num in (("node", 300), ("node", 1000), ("edge", 2000)):
        for skip in (False, True):
            np.random.seed(0)
            batches = list(
                DynamicBatchSampler(
                    graphs, max_num, mode, shuffle=True, skip_too_big=skip
                )
            )
            totals = np.array([sizes[mode][b].sum() for b in batches])
            flat = np.concatenate(batches)
            too_big = np.nonzero(sizes[mode] > max_num)[0]
            over = [b for b, t in zip(batches, totals) if t > max_num]
            expected = np.setdiff1d(np.arange(len(graphs)), too_big if skip else [])
            rows.append(
                [
                    mode,
                    max_num,
                    skip,
                    len(batches),
                    totals.max(),
                    f"{totals.mean():.0f}",
                    len(too_big),
                    all(len(b) == 1 for b in over),
                    np.array_equal(np.sort(flat), expected),
                ]
            )
    print(f"DynamicBatchSampler on PROTEINS ({len(graphs)} graphs, shuffled):\n")
    table(
        [
            "mode",
            "max_num",
            "skip_too_big",
            "batches",
            "largest batch",
            "mean batch",
            "graphs > max_num",
            "over-budget batches are single graphs",
            "every graph exactly once",
        ],
        rows,
    )

    # Epoch coverage of the seeds
    rows = []
    train = d["train"]
    ei = d["edge_index"]
    np.random.seed(0)
    seen = [
        np.asarray(b.n_id[: b.batch_size])
        for b in NeighborLoader(
            graph, [10, 5], input_nodes=mx.array(train), batch_size=32, shuffle=True
        )
    ]
    rows.append(["NeighborLoader (train seeds)", train.size, _coverage(seen, train)])
    seen = [
        np.asarray(b.input_id)
        for b in LinkNeighborLoader(
            graph, [5], neg_sampling_ratio=1.0, batch_size=512, shuffle=True
        )
    ]
    rows.append(
        [
            "LinkNeighborLoader (edges)",
            ei.shape[1],
            _coverage(seen, np.arange(ei.shape[1])),
        ]
    )
    seen = [
        np.asarray(b.n_id)[np.asarray(b.root_n_id)]
        for b in ShaDowKHopSampler(
            graph, 2, 5, node_idx=mx.array(train), batch_size=32, shuffle=True
        )
    ]
    rows.append(["ShaDowKHopSampler (train roots)", train.size, _coverage(seen, train)])
    seen = [np.asarray(b.n_id) for b in RandomNodeLoader(graph, 7, shuffle=True)]
    rows.append(["RandomNodeLoader (nodes)", N, _coverage(seen, np.arange(N))])
    cluster_data = ClusterData(graph, 20, method="balanced")
    seen = [np.asarray(b.n_id) for b in ClusterLoader(cluster_data, 3, shuffle=True)]
    rows.append(["ClusterLoader (nodes)", N, _coverage(seen, np.arange(N))])
    print("Epoch coverage (each seed / node exactly once per epoch):\n")
    table(["loader", "items", "covered exactly once"], rows)

    # GraphSAINT coverage
    rows = []
    for name, sampler in (
        ("Node", GraphSAINTNodeSampler(graph, 1000, num_steps=20)),
        ("Edge", GraphSAINTEdgeSampler(graph, 400, num_steps=20)),
        ("RandomWalk", GraphSAINTRandomWalkSampler(graph, 300, 2, num_steps=20)),
    ):
        np.random.seed(0)
        batches = list(sampler)
        times = np.bincount(
            np.concatenate([np.asarray(b.n_id) for b in batches]), minlength=N
        )
        rows.append(
            [
                name,
                len(batches),
                f"{np.mean([b.num_nodes for b in batches]):.0f}",
                f"{np.mean(times > 0):.3f}",
            ]
        )
    print("GraphSAINT node coverage over 20 subgraphs:\n")
    table(["sampler", "subgraphs", "avg. nodes", "fraction of nodes seen"], rows)

    # Neighbor sampling uniformity
    in_deg = np.bincount(ei[1], minlength=N)
    rows = []
    for v in np.argsort(-in_deg)[:3]:
        v = int(v)
        nbrs = ei[0, ei[1] == v]
        k = 5
        trials = 3000
        counts = {"NeighborLoader": np.zeros(N), "ShaDowKHopSampler": np.zeros(N)}
        loader = NeighborLoader(graph, [k], input_nodes=mx.array([v]), batch_size=1)
        shadow = ShaDowKHopSampler(graph, 1, k, node_idx=mx.array([v]))
        for _ in range(trials):
            b = next(iter(loader))
            n_id = np.asarray(b.n_id)
            counts["NeighborLoader"][n_id[np.asarray(b.edge_index)[0]]] += 1
            b = next(iter(shadow))
            counts["ShaDowKHopSampler"][np.asarray(b.n_id)] += 1
        for name, c in counts.items():
            freq = c[nbrs] / trials
            rows.append(
                [
                    name,
                    v,
                    nbrs.size,
                    k,
                    f"{k / nbrs.size:.3f}",
                    f"{freq.min():.3f}-{freq.max():.3f}",
                    f"{chi2_uniform(c[nbrs]):.2f}",
                ]
            )
    print(
        "Neighbor sampling uniformity (fan-out 5, 3000 draws, highest in-degree "
        "nodes of Cora):\n"
    )
    table(
        [
            "sampler",
            "node",
            "in-degree",
            "k",
            "expected inclusion",
            "observed inclusion",
            "chi2 p (uniform)",
        ],
        rows,
    )


def _coverage(seen, expected):
    seen = np.sort(np.concatenate(seen))
    return np.array_equal(seen, np.sort(expected))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


class SAGE(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout=0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, out_dim)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, edge_index, x):
        x = self.dropout(x)
        x = nn.relu(self.conv1(edge_index, x))
        x = self.dropout(x)
        return self.conv2(edge_index, x)


def _adam(lr):
    # bias_correction=True: same update rule as torch.optim.Adam
    return optim.Adam(learning_rate=lr, bias_correction=True)


class _L2Adam:
    """Adam with PyTorch's (coupled) L2 weight decay."""

    def __init__(self, lr, weight_decay):
        self.opt = optim.Adam(learning_rate=lr, bias_correction=True)
        self.wd = weight_decay

    def update(self, model, grads):
        from mlx.utils import tree_map

        grads = tree_map(
            lambda g, p: g + self.wd * p, grads, model.trainable_parameters()
        )
        self.opt.update(model, grads)

    @property
    def state(self):
        return self.opt.state


def _accuracy(model, graph, idx, y):
    model.eval()
    pred = np.asarray(mx.argmax(model(graph.edge_index, graph.node_features), axis=-1))
    model.train()
    return [float((pred[i] == y[i]).mean()) for i in idx]


def train_node_mlx(d, graph, method, seed, epochs, hidden=64, lr=0.01, wd=5e-4):
    mx.random.seed(seed)
    np.random.seed(seed)
    y = d["y"]
    C = int(y.max()) + 1
    model = SAGE(d["x"].shape[1], hidden, C)
    mx.eval(model.parameters())
    opt = _L2Adam(lr, wd)
    train = d["train"]

    def ce(model, edge_index, x, labels, weight):
        logits = model(edge_index, x)
        return (
            nn.losses.cross_entropy(logits, labels, reduction="none") * weight
        ).sum()

    step = nn.value_and_grad(model, ce)

    def update(edge_index, x, labels, weight):
        loss, grads = step(model, edge_index, x, labels, weight)
        opt.update(model, grads)
        mx.eval(loss, model.parameters(), opt.state)

    def batches():
        if method == "full":
            w = np.zeros(graph.num_nodes, np.float32)
            w[train] = 1.0 / train.size
            yield graph.edge_index, graph.node_features, graph.node_labels, mx.array(w)
        elif method == "neighbor":
            for b in loaders["neighbor"]:
                n = b.batch_size
                w = np.zeros(b.num_nodes, np.float32)
                w[:n] = 1.0 / n
                yield b.edge_index, b.node_features, b.node_labels, mx.array(w)
        elif method == "saint":
            for b in loaders["saint"]:
                # sum(node_norm * loss) over the sampled train nodes is an
                # unbiased estimate of sum(loss over train nodes) / N: scale
                # it to the mean train loss, the full-batch objective (else
                # weight decay dominates on graphs with few train nodes).
                w = np.asarray(b.node_norm) * np.asarray(b.train_mask)
                w *= graph.num_nodes / train.size
                yield b.edge_index, b.node_features, b.node_labels, mx.array(w)
        elif method == "cluster":
            for b in loaders["cluster"]:
                m = np.asarray(b.train_mask).astype(np.float32)
                if m.sum() == 0:
                    continue
                yield (
                    b.edge_index,
                    b.node_features,
                    b.node_labels,
                    mx.array(m / m.sum()),
                )
        elif method == "shadow":
            for b in loaders["shadow"]:
                # Loss on the root of every subgraph
                w = np.zeros(b.num_nodes, np.float32)
                root = np.asarray(b.root_n_id)
                w[root] = 1.0 / root.size
                labels = np.zeros(b.num_nodes, np.int64)
                labels[root] = np.asarray(b.node_labels)
                yield b.edge_index, b.node_features, mx.array(labels), mx.array(w)

    loaders = {}
    if method == "neighbor":
        loaders["neighbor"] = NeighborLoader(
            graph, [10, 10], input_nodes=mx.array(train), batch_size=32, shuffle=True
        )
    elif method == "saint":
        loaders["saint"] = GraphSAINTRandomWalkSampler(
            graph,
            batch_size=max(300, graph.num_nodes // 10),
            walk_length=2,
            num_steps=5,
            sample_coverage=50,
        )
    elif method == "cluster":
        cluster_data = ClusterData(graph, num_parts=20, method="balanced")
        loaders["cluster"] = ClusterLoader(cluster_data, batch_size=5, shuffle=True)
    elif method == "shadow":
        loaders["shadow"] = ShaDowKHopSampler(
            graph,
            depth=2,
            num_neighbors=10,
            node_idx=mx.array(train),
            batch_size=32,
            shuffle=True,
        )

    best_val, test_at_best = -1.0, 0.0
    for epoch in range(epochs):
        for args in batches():
            update(*args)
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            val, test = _accuracy(model, graph, [d["val"], d["test"]], y)
            if val > best_val:
                best_val, test_at_best = val, test
    return best_val, test_at_best


def train_node_pyg(d, seed, epochs, hidden=64, lr=0.01, wd=5e-4):
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import SAGEConv as PyGSAGEConv

    torch.manual_seed(seed)

    class Net(torch.nn.Module):
        def __init__(self, i, h, o):
            super().__init__()
            self.conv1, self.conv2 = PyGSAGEConv(i, h), PyGSAGEConv(h, o)

        def forward(self, x, ei):
            x = F.dropout(x, 0.5, self.training)
            x = F.relu(self.conv1(x, ei))
            x = F.dropout(x, 0.5, self.training)
            return self.conv2(x, ei)

    x = torch.from_numpy(d["x"])
    ei = torch.from_numpy(d["edge_index"])
    y = torch.from_numpy(d["y"])
    train = torch.from_numpy(d["train"])
    model = Net(x.shape[1], hidden, int(y.max()) + 1)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_val, test_at_best = -1.0, 0.0
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        F.cross_entropy(model(x, ei)[train], y[train]).backward()
        opt.step()
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                pred = model(x, ei).argmax(-1).numpy()
            val = float((pred[d["val"]] == d["y"][d["val"]]).mean())
            test = float((pred[d["test"]] == d["y"][d["test"]]).mean())
            if val > best_val:
                best_val, test_at_best = val, test
    return best_val, test_at_best


NODE_METHODS = {
    # name: (framework, method, epochs)
    "full-batch (PyG)": ("pyg", "full", 200),
    "full-batch": ("mlx", "full", 200),
    "NeighborLoader [10, 10]": ("mlx", "neighbor", 50),
    "GraphSAINTRandomWalkSampler": ("mlx", "saint", 100),
    "ClusterLoader (20 parts, 5 per batch)": ("mlx", "cluster", 100),
    "ShaDowKHopSampler (depth 2, 10)": ("mlx", "shadow", 50),
}


def run_train(data_dir, datasets, seeds):
    print("## Node classification (2-layer GraphSAGE, public split)\n")
    rows = []
    for name in datasets:
        d = load_planetoid(data_dir, name)
        graph = to_graph(d)
        for label, (framework, method, epochs) in NODE_METHODS.items():
            start = time.time()
            results = []
            for seed in seeds:
                if framework == "pyg":
                    results.append(train_node_pyg(d, seed, epochs))
                else:
                    results.append(train_node_mlx(d, graph, method, seed, epochs))
            val, test = np.array(results).T
            rows.append(
                [
                    name,
                    label,
                    epochs,
                    f"{100 * val.mean():.1f} ± {100 * val.std():.1f}",
                    f"{100 * test.mean():.1f} ± {100 * test.std():.1f}",
                    f"{(time.time() - start) / len(seeds):.1f}",
                ]
            )
            print(rows[-1], file=sys.stderr, flush=True)
    print(
        f"Seeds {seeds}; test accuracy at the best validation accuracy "
        f"(evaluated every 5 epochs, full-graph inference):\n"
    )
    table(
        [
            "dataset",
            "training",
            "epochs",
            "val acc. (%)",
            "test acc. (%)",
            "time / run (s)",
        ],
        rows,
    )


# ---------------------------------------------------------------------------
# Link prediction
# ---------------------------------------------------------------------------


def link_split(d, seed):
    """Undirected edges split 85 / 5 / 10, with fixed negatives (non-edges)
    for validation and test."""
    rng = np.random.default_rng(seed)
    n = d["x"].shape[0]
    src, dst = d["edge_index"]
    keys = np.unique(np.minimum(src, dst) * n + np.maximum(src, dst))
    keys = keys[keys // n != keys % n]
    keys = rng.permutation(keys)
    num_val, num_test = int(0.05 * keys.size), int(0.10 * keys.size)
    parts = np.split(keys, [num_val, num_val + num_test])
    pairs = [np.stack([k // n, k % n]) for k in parts]
    existing = set(keys.tolist())

    def negatives(count):
        out = []
        while len(out) < count:
            u, v = rng.integers(0, n, 2)
            key = min(u, v) * n + max(u, v)
            if u != v and key not in existing:
                out.append((u, v))
        return np.array(out).T

    val, test, train = pairs
    return {
        "train_pos": train,
        "mp_edges": np.concatenate([train, train[::-1]], axis=1),
        "val": (val, negatives(val.shape[1])),
        "test": (test, negatives(test.shape[1])),
    }


def _auc(pos, neg):
    from sklearn.metrics import roc_auc_score

    return float(
        roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg])
    )


def _bce(logits, labels):
    return nn.losses.binary_cross_entropy(
        logits, labels, with_logits=True, reduction="mean"
    )


def train_link_mlx(d, split, method, seed, epochs, hidden=128, out=64, lr=0.01):
    mx.random.seed(seed)
    np.random.seed(seed)
    n = d["x"].shape[0]
    graph = to_graph(d, split["mp_edges"])
    model = SAGE(d["x"].shape[1], hidden, out, dropout=0.0)
    mx.eval(model.parameters())
    opt = _adam(lr)

    def loss_fn(model, edge_index, x, label_index, labels):
        h = model(edge_index, x)
        logits = (h[label_index[0]] * h[label_index[1]]).sum(-1)
        return _bce(logits, labels)

    step = nn.value_and_grad(model, loss_fn)

    def batches():
        if method == "full":
            pos = split["train_pos"]
            neg = np.random.randint(0, n, pos.shape)
            label_index = np.concatenate([pos, neg], axis=1)
            labels = np.r_[np.ones(pos.shape[1]), np.zeros(pos.shape[1])]
            yield (
                graph.edge_index,
                graph.node_features,
                mx.array(label_index),
                mx.array(labels),
            )
        else:
            for b in loader:
                yield b.edge_index, b.node_features, b.edge_label_index, b.edge_label

    loader = None
    if method == "link-neighbor":
        loader = LinkNeighborLoader(
            graph,
            [10, 10],
            edge_label_index=mx.array(split["train_pos"]),
            neg_sampling_ratio=1.0,
            batch_size=256,
            shuffle=True,
        )

    def evaluate(which):
        model.eval()
        h = np.asarray(model(graph.edge_index, graph.node_features))
        model.train()
        pos, neg = split[which]
        score = lambda e: (h[e[0]] * h[e[1]]).sum(-1)  # noqa: E731
        return _auc(score(pos), score(neg))

    best_val, test_at_best = -1.0, 0.0
    for epoch in range(epochs):
        for args in batches():
            loss, grads = step(model, *args)
            opt.update(model, grads)
            mx.eval(loss, model.parameters(), opt.state)
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            val = evaluate("val")
            if val > best_val:
                best_val, test_at_best = val, evaluate("test")
    return best_val, test_at_best


def train_link_pyg(d, split, seed, epochs, hidden=128, out=64, lr=0.01):
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import SAGEConv as PyGSAGEConv

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = d["x"].shape[0]

    class Net(torch.nn.Module):
        def __init__(self, i, h, o):
            super().__init__()
            self.conv1, self.conv2 = PyGSAGEConv(i, h), PyGSAGEConv(h, o)

        def forward(self, x, ei):
            return self.conv2(F.relu(self.conv1(x, ei)), ei)

    x = torch.from_numpy(d["x"])
    ei = torch.from_numpy(split["mp_edges"])
    pos = torch.from_numpy(split["train_pos"])
    model = Net(x.shape[1], hidden, out)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def evaluate(which):
        model.eval()
        with torch.no_grad():
            h = model(x, ei).numpy()
        p, q = split[which]
        score = lambda e: (h[e[0]] * h[e[1]]).sum(-1)  # noqa: E731
        return _auc(score(p), score(q))

    best_val, test_at_best = -1.0, 0.0
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        neg = torch.from_numpy(rng.integers(0, n, tuple(pos.shape)))
        label_index = torch.cat([pos, neg], dim=1)
        labels = torch.cat([torch.ones(pos.shape[1]), torch.zeros(pos.shape[1])])
        h = model(x, ei)
        logits = (h[label_index[0]] * h[label_index[1]]).sum(-1)
        F.binary_cross_entropy_with_logits(logits, labels).backward()
        opt.step()
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            val = evaluate("val")
            if val > best_val:
                best_val, test_at_best = val, evaluate("test")
    return best_val, test_at_best


def run_link(data_dir, datasets, seeds):
    print("## Link prediction (2-layer GraphSAGE encoder, dot-product decoder)\n")
    rows = []
    methods = {
        "full-batch (PyG)": ("pyg", "full", 200),
        "full-batch": ("mlx", "full", 200),
        "LinkNeighborLoader [10, 10], 1 negative / positive": (
            "mlx",
            "link-neighbor",
            30,
        ),
    }
    for name in datasets:
        d = load_planetoid(data_dir, name)
        for label, (framework, method, epochs) in methods.items():
            start = time.time()
            results = []
            for seed in seeds:
                split = link_split(d, seed)
                if framework == "pyg":
                    results.append(train_link_pyg(d, split, seed, epochs))
                else:
                    results.append(train_link_mlx(d, split, method, seed, epochs))
            val, test = np.array(results).T
            rows.append(
                [
                    name,
                    label,
                    epochs,
                    f"{100 * val.mean():.1f} ± {100 * val.std():.1f}",
                    f"{100 * test.mean():.1f} ± {100 * test.std():.1f}",
                    f"{(time.time() - start) / len(seeds):.1f}",
                ]
            )
            print(rows[-1], file=sys.stderr, flush=True)
    print(
        f"Seeds {seeds} (each with its own 85/5/10 edge split, shared by all "
        f"methods); test ROC-AUC at the best validation AUC:\n"
    )
    table(
        [
            "dataset",
            "training",
            "epochs",
            "val AUC (%)",
            "test AUC (%)",
            "time / run (s)",
        ],
        rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--sections",
        nargs="+",
        default=["saint", "stats", "train", "link"],
        choices=["saint", "stats", "train", "link"],
    )
    parser.add_argument(
        "--data-dir", default=os.path.join(REPO_ROOT, ".mlx_graphs_data")
    )
    parser.add_argument("--datasets", nargs="+", default=["cora", "pubmed"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--num-eval",
        type=int,
        default=3000,
        help="fresh GraphSAINT subgraphs per estimate",
    )
    args = parser.parse_args()
    # Mini-batches have different shapes: bound MLX's buffer cache, which
    # otherwise keeps growing (several GB on PubMed's LinkNeighborLoader).
    mx.set_cache_limit(1 << 30)
    data_dir = os.path.abspath(args.data_dir)
    if "saint" in args.sections:
        run_saint(data_dir, args.num_eval)
    if "stats" in args.sections:
        run_stats(data_dir)
    if "train" in args.sections:
        run_train(data_dir, args.datasets, args.seeds)
    if "link" in args.sections:
        run_link(data_dir, args.datasets, args.seeds)


if __name__ == "__main__":
    main()
