"""Train a GCN on BA-Shapes and check that explainers find the house motifs.

Run with:  python examples/explain_ba_shapes.py
"""

import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from mlx_graphs.explain import Explainer
from mlx_graphs.explain.algorithm import GNNExplainer, IntegratedGradients, Saliency
from mlx_graphs.explain.datasets import ba_shapes
from mlx_graphs.explain.metric import characterization_score, fidelity, groundtruth_auc
from mlx_graphs.nn import GCNConv
from mlx_graphs.utils import k_hop_subgraph


class GCN(nn.Module):
    """A 3-layer GCN that forwards `edge_weights` so its edges can be explained."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.convs = [
            GCNConv(in_dim, hidden),
            GCNConv(hidden, hidden),
            GCNConv(hidden, hidden),
        ]
        self.head = nn.Linear(hidden, out_dim)

    def __call__(self, edge_index, node_features, edge_weights=None):
        h = node_features
        for conv in self.convs:
            h = nn.relu(conv(edge_index, h, edge_weights))
        return self.head(h)


def train(model, graph, epochs: int = 1000, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = graph.node_features.shape[0]
    perm = rng.permutation(n)
    train_idx, test_idx = mx.array(perm[: int(0.8 * n)]), mx.array(perm[int(0.8 * n) :])
    y = graph.node_labels

    def loss_fn(m):
        out = m(graph.edge_index, graph.node_features)
        return nn.losses.cross_entropy(out[train_idx], y[train_idx], reduction="mean")

    opt = optim.Adam(learning_rate=0.01)
    step = nn.value_and_grad(model, loss_fn)
    for _ in range(epochs):
        loss, grads = step(model)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)

    pred = mx.argmax(model(graph.edge_index, graph.node_features), axis=-1)
    acc = lambda idx: float((pred[idx] == y[idx]).astype(mx.float32).mean())  # noqa: E731
    print(
        f"GCN trained: loss={float(loss):.3f} train acc={acc(train_idx):.3f} "
        f"test acc={acc(test_idx):.3f}"
    )


def main():
    mx.random.seed(0)
    graph = ba_shapes(seed=0)
    model = GCN(graph.node_features.shape[1], 20, 4)
    train(model, graph)

    model_config = dict(
        mode="multiclass_classification", task_level="node", return_type="raw"
    )
    algorithms = {
        "Saliency": Saliency(),
        "IntegratedGradients": IntegratedGradients(n_steps=50, absolute=True),
        "GNNExplainer": GNNExplainer(epochs=200, lr=0.01, seed=0),
    }

    # explain a few nodes that sit inside house motifs
    motif_nodes = np.nonzero(np.asarray(graph.node_labels) > 0)[0]
    nodes = motif_nodes[:: max(1, len(motif_nodes) // 20)][:20]
    num_nodes = graph.node_features.shape[0]

    for name, algorithm in algorithms.items():
        explainer = Explainer(
            model=model,
            algorithm=algorithm,
            explanation_type="model",
            edge_mask_type="object",  # BA-Shapes has constant features: explain edges
            model_config=model_config,
        )
        aucs, fid_pos, fid_neg = [], [], []
        start = time.perf_counter()
        for node in nodes:
            exp = explainer(graph.edge_index, graph.node_features, index=int(node))
            # AUC over the edges in the node's 3-hop receptive field, as in the paper
            _, _, _, in_field = k_hop_subgraph(
                int(node), 3, graph.edge_index, num_nodes=num_nodes
            )
            keep = mx.array(np.nonzero(np.asarray(in_field))[0])
            auc = groundtruth_auc(exp.edge_mask[keep], graph.edge_groundtruth[keep])
            if auc is not None:
                aucs.append(auc)
            # a house has 6 undirected = 12 directed edges
            pos, neg = fidelity(explainer, exp.threshold("topk_hard", 12))
            fid_pos.append(pos)
            fid_neg.append(neg)
        elapsed = time.perf_counter() - start
        fp, fn = float(np.mean(fid_pos)), float(np.mean(fid_neg))
        print(
            f"{name:20s} edge AUC={np.mean(aucs):.3f}  fidelity+={fp:.2f}  "
            f"fidelity-={fn:.2f}  charact.={characterization_score(fp, fn):.2f}  "
            f"({elapsed / len(nodes) * 1000:.0f} ms/node)"
        )

    save_picture(model, graph, model_config)


def save_picture(model, graph, model_config, path="ba_shapes_explanation.png"):
    """Draws the edges GNNExplainer finds important (mask > 0.1) for a roof node."""
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200, lr=0.01, seed=0),
        explanation_type="model",
        node_mask_type="object",
        edge_mask_type="object",
        model_config=model_config,
    )
    node = int(np.nonzero(np.asarray(graph.node_labels) == 3)[0][0])  # a roof node
    exp = explainer(graph.edge_index, graph.node_features, index=node)
    names = ["base", "middle", "bottom", "top"]
    labels = [f"{i}\n{names[c]}" for i, c in enumerate(np.asarray(graph.node_labels))]
    # keep only edges that clearly matter; fall back to the top 12 if none do
    shown = exp.threshold("hard", 0.1)
    if int((shown.edge_mask > 0).sum()) == 0:
        shown = exp.threshold("topk", 12)
    try:
        shown.visualize_graph(
            path, backend="networkx", node_labels=labels, edge_labels=True
        )
        print(f"Saved the explanation of node {node} to {path}")
    except ImportError:
        print('Install `pip install "mlx-graphs[viz]"` to save a picture')


if __name__ == "__main__":
    main()
