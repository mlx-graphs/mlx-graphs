"""Graph classification on TU datasets: the same models in both frameworks.

Every model maps a batch of graphs to class logits. The MLX models take an
mlx-graphs ``GraphDataBatch``; the PyG models a PyG ``Batch``.

Models (hidden size ``H``):

- ``gine``: 3 x GINEConv (MLP H-H, edge features) + sum pooling (MUTAG).
- ``pna``: node embedding, 3 x PNAConv (mean/min/max/std, identity/
  amplification/attenuation, 4 towers, degree histogram of the training
  graphs, edge features when the dataset has them) + BatchNorm + ReLU, sum
  pooling.
- ``topk`` / ``sag``: PyG's ``proteins_topk_pool.py``: 3 x (GraphConv 128 +
  TopKPooling / SAGPooling ratio 0.8 + [max || mean] readout), summed.
- ``dgcnn``: DGCNN (GCNConv 32-32-32-1, tanh, SortAggregation k, Conv1d,
  MaxPool1d, Conv1d, MLP), as in PyG's ``seal_link_pred.py``.
- ``multiaggr``: 3 x GINConv + MultiAggregation([mean, max, sum, std]) readout.
- ``graphnorm``: 3 x (GCNConv + GraphNorm + ReLU) + mean pooling, trained
  with ``Dataloader(drop_last=True, follow_batch=["node_features"])`` and
  the ``node_features_batch`` vector (PyG: ``follow_batch=["x"]``).
- ``set2set``: 3 x GINConv + Set2Set(3 steps) readout.
- ``deepsets``: 3 x GINConv + DeepSetsAggregation(MLP, MLP) readout.
"""

import numpy as np

HIDDEN = 64
GRAPH_MODELS = (
    "gine",
    "pna",
    "topk",
    "sag",
    "dgcnn",
    "multiaggr",
    "graphnorm",
    "set2set",
    "deepsets",
)
# lr, batch size, epochs (the pooling models follow proteins_topk_pool.py)
TRAIN_ARGS = {
    "topk": dict(lr=0.0005, batch_size=60),
    "sag": dict(lr=0.0005, batch_size=60),
    "dgcnn": dict(lr=0.001, batch_size=32),
}
DEFAULT_TRAIN_ARGS = dict(lr=0.01, batch_size=32)


def sort_k(graphs, train_idx) -> int:
    """DGCNN's k: 60% of the training graphs have at most k nodes."""
    sizes = sorted(len(graphs[i]["x"]) for i in train_idx)
    return max(10, sizes[int(np.ceil(0.6 * len(sizes))) - 1])


# ---------------------------------------------------------------------------
# MLX models
# ---------------------------------------------------------------------------


def mlx_model(name, in_dim, edge_dim, num_classes, deg=None, k=None):
    import mlx.core as mx
    import mlx.nn as nn

    import mlx_graphs.nn as gnn

    H = HIDDEN

    def mlp(i, o):
        return nn.Sequential(nn.Linear(i, o), nn.ReLU(), nn.Linear(o, o))

    def head(i):
        return nn.Sequential(
            nn.Linear(i, H), nn.ReLU(), nn.Dropout(0.5), nn.Linear(H, num_classes)
        )

    class GIN(nn.Module):
        """3 x GINConv / GINEConv + a readout."""

        def __init__(self, readout, readout_dim, edge=False):
            super().__init__()
            if edge:
                self.convs = [
                    gnn.GINEConv(
                        mlp(in_dim if i == 0 else H, H), edge_features_dim=edge_dim
                    )
                    for i in range(3)
                ]
            else:
                self.convs = [
                    gnn.GINConv(mlp(in_dim if i == 0 else H, H)) for i in range(3)
                ]
            self.edge = edge
            self.readout = readout
            self.head = head(readout_dim)

        def __call__(self, b):
            x = b.node_features
            for conv in self.convs:
                if self.edge:
                    x = nn.relu(conv(b.edge_index, x, b.edge_features))
                else:
                    x = nn.relu(conv(b.edge_index, x))
            return self.head(self.readout(x, b.batch_indices, b.num_graphs))

    if name == "gine":
        return GIN(gnn.global_add_pool, H, edge=True)
    if name == "multiaggr":
        aggr = gnn.MultiAggregation(["mean", "max", "sum", "std"])
        return GIN(lambda x, b, n: aggr(x, b, n), 4 * H)
    if name == "set2set":
        s2s = gnn.Set2Set(H, processing_steps=3)
        model = GIN(None, 2 * H)
        model.s2s = s2s
        model.readout = lambda x, b, n: model.s2s(x, b, n)
        return model
    if name == "deepsets":
        model = GIN(None, H)
        model.aggr = gnn.DeepSetsAggregation(
            local_nn=mlp(H, H), global_nn=nn.Sequential(nn.Linear(H, H), nn.ReLU())
        )
        model.readout = lambda x, b, n: model.aggr(x, b, n)
        return model

    if name == "pna":

        class PNANet(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Linear(in_dim, H)
                self.convs = [
                    gnn.PNAConv(
                        H,
                        H,
                        aggregators=["mean", "min", "max", "std"],
                        scalers=["identity", "amplification", "attenuation"],
                        deg=deg,
                        edge_features_dim=edge_dim,
                        towers=4,
                        pre_layers=1,
                        post_layers=1,
                        divide_input=False,
                    )
                    for _ in range(3)
                ]
                self.norms = [nn.BatchNorm(H) for _ in range(3)]
                self.head = head(H)

            def __call__(self, b):
                x = self.emb(b.node_features)
                for conv, norm in zip(self.convs, self.norms):
                    x = nn.relu(norm(conv(b.edge_index, x, b.edge_features)))
                return self.head(gnn.global_add_pool(x, b.batch_indices, b.num_graphs))

        return PNANet()

    if name in ("topk", "sag"):

        class PoolNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.convs = [gnn.GraphConv(in_dim, 128)] + [
                    gnn.GraphConv(128, 128) for _ in range(2)
                ]
                if name == "topk":
                    self.pools = [gnn.TopKPooling(128, ratio=0.8) for _ in range(3)]
                else:
                    self.pools = [gnn.SAGPooling(128, ratio=0.8) for _ in range(3)]
                self.lin1 = nn.Linear(256, 128)
                self.lin2 = nn.Linear(128, 64)
                self.lin3 = nn.Linear(64, num_classes)
                self.dropout = nn.Dropout(0.5)

            def __call__(self, b):
                x, ei, batch = b.node_features, b.edge_index, b.batch_indices
                n = b.num_graphs
                out = 0
                for conv, pool in zip(self.convs, self.pools):
                    x = nn.relu(conv(ei, x))
                    x, ei, _, batch, _, _ = pool(ei, x, None, batch)
                    out = out + mx.concatenate(
                        [
                            gnn.global_max_pool(x, batch, n),
                            gnn.global_mean_pool(x, batch, n),
                        ],
                        axis=1,
                    )
                x = nn.relu(self.lin1(out))
                x = self.dropout(x)
                x = nn.relu(self.lin2(x))
                return self.lin3(x)

        return PoolNet()

    if name == "dgcnn":

        class DGCNN(nn.Module):
            def __init__(self):
                super().__init__()
                dims = [in_dim, 32, 32, 32, 1]
                self.convs = [
                    gnn.GCNConv(dims[i], dims[i + 1], add_self_loops=True)
                    for i in range(4)
                ]
                self.sort = gnn.SortAggregation(k)
                total = 97
                self.conv1 = nn.Conv1d(1, 16, total, stride=total)
                self.pool = nn.MaxPool1d(2, 2)
                self.conv2 = nn.Conv1d(16, 32, 5, stride=1)
                # MLX's Conv1d starts with zero biases, torch's with uniform
                # ones in +-1/sqrt(fan_in): initialize them like torch
                for conv, fan_in in ((self.conv1, total), (self.conv2, 16 * 5)):
                    bound = 1 / fan_in**0.5
                    conv.bias = mx.random.uniform(-bound, bound, conv.bias.shape)
                dense = (int((k - 2) / 2 + 1) - 5 + 1) * 32
                self.lin1 = nn.Linear(dense, 128)
                self.dropout = nn.Dropout(0.5)
                self.lin2 = nn.Linear(128, num_classes)

            def __call__(self, b):
                xs, x = [], b.node_features
                for conv in self.convs:
                    x = mx.tanh(conv(b.edge_index, x))
                    xs.append(x)
                x = mx.concatenate(xs, axis=-1)
                x = self.sort(x, b.batch_indices, b.num_graphs)  # [B, k * 97]
                x = x[:, :, None]  # channels last: [B, k * 97, 1]
                x = nn.relu(self.conv1(x))
                x = self.pool(x)
                x = nn.relu(self.conv2(x))
                # flatten in PyTorch's [B, C, L] order
                x = x.transpose(0, 2, 1).reshape(x.shape[0], -1)
                x = nn.relu(self.lin1(x))
                return self.lin2(self.dropout(x))

        return DGCNN()

    if name == "graphnorm":

        class GNNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.convs = [
                    gnn.GCNConv(in_dim if i == 0 else H, H, add_self_loops=True)
                    for i in range(3)
                ]
                self.norms = [gnn.GraphNorm(H) for _ in range(3)]
                self.head = head(H)

            def __call__(self, b):
                # batch vector given by follow_batch=["node_features"]
                batch, n = b.node_features_batch, b.num_graphs
                x = b.node_features
                for conv, norm in zip(self.convs, self.norms):
                    x = nn.relu(norm(conv(b.edge_index, x), batch, n))
                return self.head(gnn.global_mean_pool(x, batch, n))

        return GNNet()

    raise ValueError(name)


# ---------------------------------------------------------------------------
# PyG models
# ---------------------------------------------------------------------------


def pyg_model(name, in_dim, edge_dim, num_classes, deg=None, k=None):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch_geometric.nn as pnn

    H = HIDDEN

    def mlp(i, o):
        return nn.Sequential(nn.Linear(i, o), nn.ReLU(), nn.Linear(o, o))

    def head(i):
        return nn.Sequential(
            nn.Linear(i, H), nn.ReLU(), nn.Dropout(0.5), nn.Linear(H, num_classes)
        )

    class GIN(nn.Module):
        def __init__(self, readout, readout_dim, edge=False):
            super().__init__()
            if edge:
                self.convs = nn.ModuleList(
                    pnn.GINEConv(mlp(in_dim if i == 0 else H, H), edge_dim=edge_dim)
                    for i in range(3)
                )
            else:
                self.convs = nn.ModuleList(
                    pnn.GINConv(mlp(in_dim if i == 0 else H, H)) for i in range(3)
                )
            self.edge = edge
            self.readout = readout
            self.head = head(readout_dim)

        def forward(self, b):
            x = b.x
            for conv in self.convs:
                if self.edge:
                    x = F.relu(conv(x, b.edge_index, b.edge_attr))
                else:
                    x = F.relu(conv(x, b.edge_index))
            return self.head(self.readout(x, b.batch, b.num_graphs))

    if name == "gine":
        return GIN(lambda x, b, n: pnn.global_add_pool(x, b, n), H, edge=True)
    if name == "multiaggr":
        model = GIN(None, 4 * H)
        model.aggr = pnn.aggr.MultiAggregation(["mean", "max", "sum", "std"])
        model.readout = lambda x, b, n: model.aggr(x, b, dim_size=n)
        return model
    if name == "set2set":
        model = GIN(None, 2 * H)
        model.s2s = pnn.aggr.Set2Set(H, processing_steps=3)
        model.readout = lambda x, b, n: model.s2s(x, b, dim_size=n)
        return model
    if name == "deepsets":
        model = GIN(None, H)
        model.aggr = pnn.aggr.DeepSetsAggregation(
            local_nn=mlp(H, H), global_nn=nn.Sequential(nn.Linear(H, H), nn.ReLU())
        )
        model.readout = lambda x, b, n: model.aggr(x, b, dim_size=n)
        return model

    if name == "pna":

        class PNANet(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Linear(in_dim, H)
                self.convs = nn.ModuleList(
                    pnn.PNAConv(
                        H,
                        H,
                        aggregators=["mean", "min", "max", "std"],
                        scalers=["identity", "amplification", "attenuation"],
                        deg=torch.as_tensor(deg),
                        edge_dim=edge_dim,
                        towers=4,
                        pre_layers=1,
                        post_layers=1,
                        divide_input=False,
                    )
                    for _ in range(3)
                )
                self.norms = nn.ModuleList(nn.BatchNorm1d(H) for _ in range(3))
                self.head = head(H)

            def forward(self, b):
                x = self.emb(b.x)
                ea = b.edge_attr if edge_dim else None
                for conv, norm in zip(self.convs, self.norms):
                    x = F.relu(norm(conv(x, b.edge_index, ea)))
                return self.head(pnn.global_add_pool(x, b.batch, b.num_graphs))

        return PNANet()

    if name in ("topk", "sag"):

        class PoolNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.convs = nn.ModuleList(
                    [pnn.GraphConv(in_dim, 128)]
                    + [pnn.GraphConv(128, 128) for _ in range(2)]
                )
                Pool = pnn.TopKPooling if name == "topk" else pnn.SAGPooling
                self.pools = nn.ModuleList(Pool(128, ratio=0.8) for _ in range(3))
                self.lin1 = nn.Linear(256, 128)
                self.lin2 = nn.Linear(128, 64)
                self.lin3 = nn.Linear(64, num_classes)

            def forward(self, b):
                x, ei, batch, n = b.x, b.edge_index, b.batch, b.num_graphs
                out = 0
                for conv, pool in zip(self.convs, self.pools):
                    x = F.relu(conv(x, ei))
                    x, ei, _, batch, _, _ = pool(x, ei, None, batch)
                    out = out + torch.cat(
                        [
                            pnn.global_max_pool(x, batch, n),
                            pnn.global_mean_pool(x, batch, n),
                        ],
                        dim=1,
                    )
                x = F.relu(self.lin1(out))
                x = F.dropout(x, p=0.5, training=self.training)
                x = F.relu(self.lin2(x))
                return self.lin3(x)

        return PoolNet()

    if name == "dgcnn":

        class DGCNN(nn.Module):
            def __init__(self):
                super().__init__()
                dims = [in_dim, 32, 32, 32, 1]
                self.convs = nn.ModuleList(
                    pnn.GCNConv(dims[i], dims[i + 1]) for i in range(4)
                )
                self.sort = pnn.aggr.SortAggregation(k)
                total = 97
                self.conv1 = nn.Conv1d(1, 16, total, total)
                self.pool = nn.MaxPool1d(2, 2)
                self.conv2 = nn.Conv1d(16, 32, 5, 1)
                dense = (int((k - 2) / 2 + 1) - 5 + 1) * 32
                self.lin1 = nn.Linear(dense, 128)
                self.dropout = nn.Dropout(0.5)
                self.lin2 = nn.Linear(128, num_classes)

            def forward(self, b):
                xs, x = [], b.x
                for conv in self.convs:
                    x = torch.tanh(conv(x, b.edge_index))
                    xs.append(x)
                x = torch.cat(xs, dim=-1)
                x = self.sort(x, b.batch, dim_size=b.num_graphs)
                x = x.unsqueeze(1)  # [B, 1, k * 97]
                x = F.relu(self.conv1(x))
                x = self.pool(x)
                x = F.relu(self.conv2(x))
                x = x.reshape(x.size(0), -1)
                x = F.relu(self.lin1(x))
                return self.lin2(self.dropout(x))

        return DGCNN()

    if name == "graphnorm":

        class GNNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.convs = nn.ModuleList(
                    pnn.GCNConv(in_dim if i == 0 else H, H) for i in range(3)
                )
                self.norms = nn.ModuleList(pnn.GraphNorm(H) for _ in range(3))
                self.head = head(H)

            def forward(self, b):
                batch, n = b.x_batch, b.num_graphs  # follow_batch=["x"]
                x = b.x
                for conv, norm in zip(self.convs, self.norms):
                    x = F.relu(norm(conv(x, b.edge_index), batch, n))
                return self.head(pnn.global_mean_pool(x, batch, n))

        return GNNet()

    raise ValueError(name)
