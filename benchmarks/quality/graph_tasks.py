"""Heterogeneous, link prediction, recommendation, self-supervised and
knowledge graph tasks, each in both frameworks (``run_<task>(args)``).

They follow PyG's examples (``hetero/han_imdb.py``, ``hetero/hgt_dblp.py``,
``rgat.py``, ``rgcn.py``, ``autoencoder.py``, ``lightgcn.py``,
``infomax_transductive.py``, ``kge_fb15k_237.py``) with the same
hyperparameters in both frameworks.
"""

import graph_data
import numpy as np
from graph_worker import Selector, mlx_adam

# ---------------------------------------------------------------------------
# Shared training loop of the full-batch node classification tasks
# ---------------------------------------------------------------------------


def _node_classification(args, mlx_build, pyg_build, lr, weight_decay, y, idx):
    """``*_build()`` return ``forward(training) -> logits of the labelled
    nodes`` and the model. ``idx`` are the train / val / test positions in
    those logits."""
    train, val, test = idx
    if args.framework == "mlx":
        import mlx.core as mx
        import mlx.nn as nn

        forward, model = mlx_build()
        optimizer, decay = mlx_adam(lr, weight_decay)
        ym = mx.array(y.astype(np.int32))
        tr, va, te = (mx.array(i.astype(np.int32)) for i in idx)

        def loss_fn(model):
            out = forward()
            return nn.losses.cross_entropy(out[tr], ym[tr], reduction="mean")

        loss_and_grad = nn.value_and_grad(model, loss_fn)

        def step():
            model.train()
            loss, grads = loss_and_grad(model)
            optimizer.update(model, decay(grads, model.trainable_parameters()))
            mx.eval(loss, model.parameters(), optimizer.state)
            return loss.item()

        def evaluate():
            model.eval()
            out = forward()
            pred = mx.argmax(out, axis=1)
            val_loss = nn.losses.cross_entropy(out[va], ym[va], reduction="mean")
            accs = [
                (pred[i] == ym[i]).astype(mx.float32).mean().item() for i in (va, te)
            ]
            return (*accs, val_loss.item())

    else:
        import torch
        import torch.nn.functional as F

        forward, model = pyg_build()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        yt = torch.from_numpy(y)
        tr, va, te = (torch.from_numpy(i) for i in idx)

        def step():
            model.train()
            optimizer.zero_grad()
            loss = F.cross_entropy(forward()[tr], yt[tr])
            loss.backward()
            optimizer.step()
            return loss.item()

        @torch.no_grad()
        def evaluate():
            model.eval()
            out = forward()
            pred = out.argmax(1)
            val_loss = F.cross_entropy(out[va], yt[va]).item()
            accs = [(pred[i] == yt[i]).float().mean().item() for i in (va, te)]
            return (*accs, val_loss)

    selector, losses = Selector(), []
    for epoch in range(1, args.epochs + 1):
        losses.append(step())
        val_acc, test_acc, val_loss = evaluate()
        selector.add(epoch, val_acc, test_acc, val_loss)
    return {"metric": "accuracy", "loss_curve": losses, **selector.summary()}


# ---------------------------------------------------------------------------
# Heterogeneous graphs: HAN on IMDB, HGT on DBLP
# ---------------------------------------------------------------------------


def run_han(args):
    """PyG's han_imdb.py: HANConv(-1, 128, heads=8, dropout=0.6) on the
    movie-actor-movie / movie-director-movie metapaths + Linear(128, 3);
    Adam(lr=0.005, weight_decay=0.001)."""
    data = graph_data.imdb()
    x = data["x"]
    ets = [("movie", "metapath_0", "movie"), ("movie", "metapath_1", "movie")]
    eis = [data["metapath_actor"], data["metapath_director"]]
    metadata = (["movie"], ets)
    masks = [np.nonzero(data[f"{s}_mask"])[0] for s in ("train", "val", "test")]

    def mlx_build():
        import mlx.core as mx
        import mlx.nn as nn

        from mlx_graphs.nn import HANConv

        class HAN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = HANConv(
                    {"movie": x.shape[1]}, 128, metadata, heads=8, dropout=0.6
                )
                self.lin = nn.Linear(128, 3)

            def __call__(self, x_dict, ei_dict):
                return self.lin(self.conv(x_dict, ei_dict)["movie"])

        model = HAN()
        x_dict = {"movie": mx.array(x)}
        ei_dict = {et: mx.array(ei.astype(np.int32)) for et, ei in zip(ets, eis)}
        return (lambda: model(x_dict, ei_dict)), model

    def pyg_build():
        import torch
        from torch_geometric.nn import HANConv

        class HAN(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = HANConv(
                    {"movie": x.shape[1]}, 128, metadata, heads=8, dropout=0.6
                )
                self.lin = torch.nn.Linear(128, 3)

            def forward(self, x_dict, ei_dict):
                return self.lin(self.conv(x_dict, ei_dict)["movie"])

        model = HAN()
        x_dict = {"movie": torch.from_numpy(x)}
        ei_dict = {et: torch.from_numpy(ei) for et, ei in zip(ets, eis)}
        return (lambda: model(x_dict, ei_dict)), model

    return _node_classification(
        args, mlx_build, pyg_build, 0.005, 0.001, data["y"], masks
    )


def run_hgt(args):
    """PyG's hgt_dblp.py: per-type Linear(-1, 64) + ReLU, HGTConv(64, 64,
    heads=2) x 2, Linear(64, 4) on the authors; Adam(lr=0.005,
    weight_decay=0.001)."""
    data = graph_data.dblp()
    x, ei = graph_data.dblp_hetero(data)
    metadata = (list(x), list(ei))
    masks = [np.nonzero(data[f"{s}_mask"])[0] for s in ("train", "val", "test")]
    H, layers = 64, 2

    def mlx_build():
        import mlx.core as mx
        import mlx.nn as nn

        from mlx_graphs.nn import HGTConv

        class HGT(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin_dict = {t: nn.Linear(v.shape[1], H) for t, v in x.items()}
                self.convs = [HGTConv(H, H, metadata, heads=2) for _ in range(layers)]
                self.lin = nn.Linear(H, 4)

            def __call__(self, x_dict, ei_dict):
                x_dict = {t: nn.relu(self.lin_dict[t](v)) for t, v in x_dict.items()}
                for conv in self.convs:
                    x_dict = conv(x_dict, ei_dict)
                return self.lin(x_dict["author"])

        model = HGT()
        x_dict = {t: mx.array(v) for t, v in x.items()}
        ei_dict = {et: mx.array(v.astype(np.int32)) for et, v in ei.items()}
        return (lambda: model(x_dict, ei_dict)), model

    def pyg_build():
        import torch
        from torch_geometric.nn import HGTConv

        class HGT(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin_dict = torch.nn.ModuleDict(
                    {t: torch.nn.Linear(v.shape[1], H) for t, v in x.items()}
                )
                self.convs = torch.nn.ModuleList(
                    HGTConv(H, H, metadata, heads=2) for _ in range(layers)
                )
                self.lin = torch.nn.Linear(H, 4)

            def forward(self, x_dict, ei_dict):
                x_dict = {t: self.lin_dict[t](v).relu() for t, v in x_dict.items()}
                for conv in self.convs:
                    x_dict = conv(x_dict, ei_dict)
                return self.lin(x_dict["author"])

        model = HGT()
        x_dict = {t: torch.from_numpy(v) for t, v in x.items()}
        ei_dict = {et: torch.from_numpy(v) for et, v in ei.items()}
        return (lambda: model(x_dict, ei_dict)), model

    return _node_classification(
        args, mlx_build, pyg_build, 0.005, 0.001, data["y"], masks
    )


# ---------------------------------------------------------------------------
# Relational layers on DBLP made homogeneous: RGATConv, FastRGCNConv
# ---------------------------------------------------------------------------


def _relational(args, layer):
    """2 relational layers (hidden 16) + Linear on DBLP converted with
    ``to_homogeneous()``, classifying the authors. RGAT follows PyG's
    rgat.py (Adam lr 0.01, weight decay 5e-4), FastRGCN rgcn.py (hidden 16,
    no bases, lr 0.01, weight decay 5e-4)."""
    data = graph_data.dblp_homogeneous()
    x, ei, et, y = data["x"], data["edge_index"], data["edge_type"], data["y"]
    R = int(et.max()) + 1
    idx = [data[f"{s}_idx"] for s in ("train", "val", "test")]
    labelled = np.concatenate(idx)
    pos = {i: k for k, i in enumerate(labelled)}
    idx_local = [np.array([pos[i] for i in s]) for s in idx]
    num_classes, H = int(y[labelled].max()) + 1, 16

    def mlx_build():
        import mlx.core as mx
        import mlx.nn as nn

        from mlx_graphs.nn import FastRGCNConv, RGATConv

        Conv = RGATConv if layer == "rgat" else FastRGCNConv

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = Conv(x.shape[1], H, R)
                self.conv2 = Conv(H, H, R)
                self.lin = nn.Linear(H, num_classes)

            def __call__(self, ei, x, et):
                h = nn.relu(self.conv1(ei, x, et))
                h = nn.relu(self.conv2(ei, h, et))
                return self.lin(h)

        model = Net()
        xm, eim, etm = (
            mx.array(x),
            mx.array(ei.astype(np.int32)),
            mx.array(et.astype(np.int32)),
        )
        lab = mx.array(labelled.astype(np.int32))
        return (lambda: model(eim, xm, etm)[lab]), model

    def pyg_build():
        import torch
        from torch_geometric.nn import FastRGCNConv, RGATConv

        Conv = RGATConv if layer == "rgat" else FastRGCNConv

        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = Conv(x.shape[1], H, R)
                self.conv2 = Conv(H, H, R)
                self.lin = torch.nn.Linear(H, num_classes)

            def forward(self, x, ei, et):
                h = self.conv1(x, ei, et).relu()
                h = self.conv2(h, ei, et).relu()
                return self.lin(h)

        model = Net()
        xt, eit, ett = (torch.from_numpy(a) for a in (x, ei, et))
        lab = torch.from_numpy(labelled)
        return (lambda: model(xt, eit, ett)[lab]), model

    return _node_classification(
        args, mlx_build, pyg_build, 0.01, 5e-4, y[labelled], idx_local
    )


def run_rgat(args):
    return _relational(args, "rgat")


def run_fastrgcn(args):
    return _relational(args, "fastrgcn")


# ---------------------------------------------------------------------------
# Graph auto-encoders (PyG's autoencoder.py)
# ---------------------------------------------------------------------------


def _auc_ap(pos, neg):
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    s = np.concatenate([pos, neg])
    return roc_auc_score(y, s), average_precision_score(y, s)


def run_gae(args):
    """GCN encoder (2 x 16 -> 16, or mu / logstd heads for VGAE), inner
    product decoder, Adam(lr=0.01), 400 epochs; trained on the positive
    training edges with one random negative per positive. Validation AUC uses
    the training edges for message passing, test AUC the training and
    validation edges (as ``RandomLinkSplit``'s test data)."""
    variational = args.model == "vgae"
    method = (
        "train_test_split_edges" if args.model.endswith("ttse") else "random_link_split"
    )
    data = graph_data.planetoid(args.dataset)
    split = graph_data.link_split(args.dataset, args.seed, method)
    x = data["x"]
    N, out = x.shape[0], 16
    mp_val = split["train_edge_index"]
    vp = split["val_pos"]
    mp_test = np.concatenate([mp_val, vp, vp[::-1]], axis=1)

    if args.framework == "mlx":
        import mlx.core as mx
        import mlx.nn as nn

        from mlx_graphs.nn import GAE, VGAE, GCNConv

        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = GCNConv(x.shape[1], 2 * out, add_self_loops=True)
                if variational:
                    self.conv_mu = GCNConv(2 * out, out, add_self_loops=True)
                    self.conv_logstd = GCNConv(2 * out, out, add_self_loops=True)
                else:
                    self.conv2 = GCNConv(2 * out, out, add_self_loops=True)

            def __call__(self, ei, x):
                h = nn.relu(self.conv1(ei, x))
                if variational:
                    return self.conv_mu(ei, h), self.conv_logstd(ei, h)
                return self.conv2(ei, h)

        model = (VGAE if variational else GAE)(Encoder())
        optimizer, _ = mlx_adam(0.01)
        xm = mx.array(x)
        e_train = mx.array(split["train_edge_index"].astype(np.int32))
        pos_train = mx.array(split["train_pos"].astype(np.int32))
        mps = {
            k: mx.array(v.astype(np.int32))
            for k, v in (("val", mp_val), ("test", mp_test))
        }

        def loss_fn(model):
            z = model.encode(e_train, xm)
            loss = model.recon_loss(z, pos_train)
            if variational:
                loss = loss + (1 / N) * model.kl_loss()
            return loss

        loss_and_grad = nn.value_and_grad(model, loss_fn)

        def step():
            model.train()
            loss, grads = loss_and_grad(model)
            optimizer.update(model, grads)
            mx.eval(loss, model.parameters(), optimizer.state)
            return loss.item()

        def scores(which):
            model.eval()
            z = np.array(model.encode(mps[which], xm))
            return z

        mlx_metric = {}

        def library_test(z, pos, neg):
            # the library's own AUC / AP, cross-checked against scikit-learn
            mz = mx.array(z)
            mlx_metric["auc_ap"] = model.test(
                mz, mx.array(pos.astype(np.int32)), mx.array(neg.astype(np.int32))
            )

    else:
        import torch
        from torch_geometric.nn import GAE, VGAE, GCNConv

        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = GCNConv(x.shape[1], 2 * out)
                if variational:
                    self.conv_mu = GCNConv(2 * out, out)
                    self.conv_logstd = GCNConv(2 * out, out)
                else:
                    self.conv2 = GCNConv(2 * out, out)

            def forward(self, x, ei):
                h = self.conv1(x, ei).relu()
                if variational:
                    return self.conv_mu(h, ei), self.conv_logstd(h, ei)
                return self.conv2(h, ei)

        model = (VGAE if variational else GAE)(Encoder())
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        xt = torch.from_numpy(x)
        e_train = torch.from_numpy(split["train_edge_index"])
        pos_train = torch.from_numpy(split["train_pos"])
        mps = {"val": torch.from_numpy(mp_val), "test": torch.from_numpy(mp_test)}

        def step():
            model.train()
            optimizer.zero_grad()
            z = model.encode(xt, e_train)
            loss = model.recon_loss(z, pos_train)
            if variational:
                loss = loss + (1 / N) * model.kl_loss()
            loss.backward()
            optimizer.step()
            return loss.item()

        @torch.no_grad()
        def scores(which):
            model.eval()
            return model.encode(xt, mps[which]).numpy()

        library_test = None

    def edge_scores(z, edges):
        logits = (z[edges[0]] * z[edges[1]]).sum(-1)
        return 1 / (1 + np.exp(-logits))

    selector, losses = Selector(), []
    for epoch in range(1, args.epochs + 1):
        losses.append(step())
        if epoch % 10 == 0 or epoch == args.epochs:
            zv, zt = scores("val"), scores("test")
            val_auc, val_ap = _auc_ap(
                edge_scores(zv, vp), edge_scores(zv, split["val_neg"])
            )
            auc, ap = _auc_ap(
                edge_scores(zt, split["test_pos"]), edge_scores(zt, split["test_neg"])
            )
            selector.add(epoch, val_auc, auc, 0.0, ap=ap)
    result = {"metric": "auc", "loss_curve": losses, **selector.summary()}
    if library_test is not None:  # the library's AUC / AP on the last model
        library_test(zt, split["test_pos"], split["test_neg"])
        result["library_auc_ap"] = list(mlx_metric["auc_ap"])
        result["sklearn_auc_ap"] = [auc, ap]
    return result


# ---------------------------------------------------------------------------
# LightGCN on MovieLens100K (BPR loss, recall@20)
# ---------------------------------------------------------------------------


def run_lightgcn(args):
    """PyG's lightgcn.py on MovieLens100K (u1.base / u1.test, ratings >= 4):
    LightGCN(64, 2 layers), BPR loss with lambda 1e-4 and one random negative
    movie per positive pair, batches of 1024, Adam(lr=0.001); recall@20 and
    precision@20 on the test ratings, excluding the training ones."""
    data = graph_data.movielens()
    U, M = int(data["num_users"]), int(data["num_movies"])
    train, test = data["train"], data["test"]
    train_label = np.stack([train[0], train[1] + U])  # movies after users
    edge_index = np.concatenate([train_label, train_label[::-1]], axis=1)
    k, batch_size = 20, 1024
    test_items = [set() for _ in range(U)]
    for u, m in test.T:
        test_items[u].add(m)
    train_mask = np.zeros((U, M), dtype=bool)
    train_mask[train[0], train[1]] = True
    rng = np.random.default_rng(args.seed)

    def recall_precision(emb):
        logits = emb[:U] @ emb[U:].T
        logits[train_mask] = -np.inf
        top = np.argpartition(-logits, k, axis=1)[:, :k]
        recalls, precisions = [], []
        for u in range(U):
            if not test_items[u]:
                continue
            hits = len(test_items[u].intersection(top[u].tolist()))
            recalls.append(hits / len(test_items[u]))
            precisions.append(hits / k)
        return float(np.mean(recalls)), float(np.mean(precisions))

    if args.framework == "mlx":
        import mlx.core as mx
        import mlx.nn as nn

        from mlx_graphs.nn import LightGCN

        model = LightGCN(U + M, embedding_dim=64, num_layers=2)
        optimizer, _ = mlx_adam(0.001)
        ei = mx.array(edge_index.astype(np.int32))

        def loss_fn(model, label_index):
            pos, neg = mx.split(model(ei, label_index), 2)
            node_id = mx.array(np.unique(np.array(label_index)).astype(np.int32))
            return model.recommendation_loss(pos, neg, node_id=node_id)

        loss_and_grad = nn.value_and_grad(model, loss_fn)

        def step(pos, neg_items):
            label = np.concatenate([pos, np.stack([pos[0], neg_items])], axis=1)
            loss, grads = loss_and_grad(model, mx.array(label.astype(np.int32)))
            optimizer.update(model, grads)
            mx.eval(loss, model.parameters(), optimizer.state)
            return loss.item()

        def embedding():
            return np.array(model.get_embedding(ei))

    else:
        import torch
        from torch_geometric.nn import LightGCN

        model = LightGCN(U + M, embedding_dim=64, num_layers=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        ei = torch.from_numpy(edge_index)

        def step(pos, neg_items):
            label = torch.from_numpy(
                np.concatenate([pos, np.stack([pos[0], neg_items])], axis=1)
            )
            optimizer.zero_grad()
            pos_rank, neg_rank = model(ei, label).chunk(2)
            loss = model.recommendation_loss(pos_rank, neg_rank, node_id=label.unique())
            loss.backward()
            optimizer.step()
            return loss.item()

        @torch.no_grad()
        def embedding():
            return model.get_embedding(ei).numpy()

    selector, losses = Selector(), []
    n = train_label.shape[1]
    for epoch in range(1, args.epochs + 1):
        perm = rng.permutation(n)
        total = 0.0
        for start in range(0, n, batch_size):
            index = perm[start : start + batch_size]
            neg = rng.integers(U, U + M, size=index.size)
            total += step(train_label[:, index], neg) * index.size
        losses.append(total / n)
        if epoch % 5 == 0 or epoch == args.epochs:
            recall, precision = recall_precision(embedding())
            # no validation split: the last evaluation is reported
            selector.add(epoch, epoch, recall, 0.0, precision=precision)
    return {"metric": "recall@20", "loss_curve": losses, **selector.summary()}


# ---------------------------------------------------------------------------
# Deep Graph Infomax on Cora (PyG's infomax_transductive.py)
# ---------------------------------------------------------------------------


def run_dgi(args):
    """GCNConv(1433, 512) + PReLU encoder, sigmoid(mean) summary, row
    permutation corruption, Adam(lr=0.001). The embeddings are evaluated by
    the same scikit-learn logistic regression (lbfgs, max_iter 150, as PyG's
    ``DeepGraphInfomax.test``) in both frameworks."""
    from sklearn.linear_model import LogisticRegression

    data = graph_data.planetoid(args.dataset)
    x, ei, y = data["x"], data["edge_index"], data["y"]
    H = 512
    masks = [data[f"{s}_mask"] for s in ("train", "val", "test")]

    def probe(z):
        clf = LogisticRegression(solver="lbfgs", max_iter=150)
        clf.fit(z[masks[0]], y[masks[0]])
        return [clf.score(z[m], y[m]) for m in masks[1:]]

    if args.framework == "mlx":
        import mlx.core as mx
        import mlx.nn as nn

        from mlx_graphs.nn import DeepGraphInfomax, GCNConv

        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = GCNConv(x.shape[1], H, add_self_loops=True)
                self.prelu = nn.PReLU(H)

            def __call__(self, ei, x):
                return self.prelu(self.conv(ei, x))

        def corruption(ei, x):
            return ei, x[mx.random.permutation(x.shape[0])]

        model = DeepGraphInfomax(
            H,
            Encoder(),
            summary=lambda z, *a, **k: mx.sigmoid(z.mean(axis=0)),
            corruption=corruption,
        )
        optimizer, _ = mlx_adam(0.001)
        xm, eim = mx.array(x), mx.array(ei.astype(np.int32))

        def loss_fn(model):
            return model.loss(*model(eim, xm))

        loss_and_grad = nn.value_and_grad(model, loss_fn)

        def step():
            model.train()
            loss, grads = loss_and_grad(model)
            optimizer.update(model, grads)
            mx.eval(loss, model.parameters(), optimizer.state)
            return loss.item()

        def embed():
            model.eval()
            return np.array(model(eim, xm)[0])

    else:
        import torch
        from torch_geometric.nn import DeepGraphInfomax, GCNConv

        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = GCNConv(x.shape[1], H)
                self.prelu = torch.nn.PReLU(H)

            def forward(self, x, ei):
                return self.prelu(self.conv(x, ei))

        def corruption(x, ei):
            return x[torch.randperm(x.size(0))], ei

        model = DeepGraphInfomax(
            H,
            Encoder(),
            summary=lambda z, *a, **k: z.mean(dim=0).sigmoid(),
            corruption=corruption,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        xt, eit = torch.from_numpy(x), torch.from_numpy(ei)

        def step():
            model.train()
            optimizer.zero_grad()
            loss = model.loss(*model(xt, eit))
            loss.backward()
            optimizer.step()
            return loss.item()

        @torch.no_grad()
        def embed():
            model.eval()
            return model(xt, eit)[0].numpy()

    selector, losses = Selector(), []
    for epoch in range(1, args.epochs + 1):
        losses.append(step())
        if epoch % 25 == 0 or epoch == args.epochs:
            val, test = probe(embed())
            selector.add(epoch, val, test)
    return {"metric": "accuracy", "loss_curve": losses, **selector.summary()}


# ---------------------------------------------------------------------------
# Knowledge graph embeddings on FB15k-237 (PyG's kge_fb15k_237.py)
# ---------------------------------------------------------------------------

KGE = {
    # model: (class name, kwargs, optimizer, lr, weight decay)
    "transe": ("TransE", {}, "adam", 0.01, 0.0),
    "distmult": ("DistMult", {}, "adam", 0.0001, 1e-6),
    "complex": ("ComplEx", {}, "adagrad", 0.001, 1e-6),
    "rotate": ("RotatE", {"margin": 9.0}, "adam", 0.001, 0.0),
}
KGE_EVAL = 2000  # validation / test triplets ranked at every evaluation


def run_kge(args):
    """50-dimensional embeddings, batches of 1000 triplets, the optimizers
    of kge_fb15k_237.py. Every 5 epochs the raw MRR and Hits@10 of the true
    tail among all the entities are computed by the same ranking code (rank =
    number of entities scored strictly higher) on fixed random subsets of
    the validation and test triplets."""
    data = graph_data.fb15k_237()
    N, R = int(data["num_nodes"]), int(data["num_relations"])
    cls_name, kwargs, opt, lr, wd = KGE[args.model]
    rng = np.random.default_rng(0)  # same evaluation triplets for every run
    evals = {}
    for s in ("val", "test"):
        sel = rng.choice(len(data[f"{s}_head"]), KGE_EVAL, replace=False)
        evals[s] = [data[f"{s}_{p}"][sel] for p in ("head", "rel", "tail")]
    train = [data[f"train_{p}"] for p in ("head", "rel", "tail")]
    chunk = 50

    if args.framework == "mlx":
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        from mlx.utils import tree_map

        import mlx_graphs.nn as gnn

        model = getattr(gnn, cls_name)(N, R, hidden_features_dim=50, **kwargs)
        if opt == "adam":
            optimizer = optim.Adam(learning_rate=lr, bias_correction=True)
        else:  # torch.optim.Adagrad: eps 1e-10, no initial accumulator
            optimizer = optim.Adagrad(learning_rate=lr, eps=1e-10)
        loader = model.loader(
            *(mx.array(a.astype(np.int32)) for a in train),
            batch_size=1000,
            shuffle=True,
        )
        loss_and_grad = nn.value_and_grad(model, lambda m, h, r, t: m.loss(h, r, t))

        def train_epoch():
            total, n = 0.0, 0
            for h, r, t in loader:
                loss, grads = loss_and_grad(model, h, r, t)
                if wd:
                    grads = tree_map(
                        lambda g, p: g + wd * p, grads, model.trainable_parameters()
                    )
                optimizer.update(model, grads)
                mx.eval(loss, model.parameters(), optimizer.state)
                total += loss.item() * h.size
                n += h.size
            return total / n

        def ranks(h, r, t):
            out = []
            cand = mx.arange(N)[None, :]
            for i in range(0, len(h), chunk):
                hh, rr, tt = (
                    mx.array(a[i : i + chunk, None].astype(np.int32)) for a in (h, r, t)
                )
                scores = model(hh, rr, cand)
                true = mx.take_along_axis(scores, tt, axis=1)
                out.append(np.array((scores > true).sum(axis=1)))
            return np.concatenate(out)

    else:
        import torch
        import torch_geometric.nn as pnn

        model = getattr(pnn, cls_name)(N, R, hidden_channels=50, **kwargs)
        if opt == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        else:
            optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=wd)
        loader = model.loader(
            *(torch.from_numpy(a) for a in train), batch_size=1000, shuffle=True
        )

        def train_epoch():
            model.train()
            total, n = 0.0, 0
            for h, r, t in loader:
                optimizer.zero_grad()
                loss = model.loss(h, r, t)
                loss.backward()
                optimizer.step()
                total += loss.item() * h.numel()
                n += h.numel()
            return total / n

        @torch.no_grad()
        def ranks(h, r, t):
            model.eval()
            out = []
            for i in range(0, len(h), chunk):
                # PyG's models score 1-D index arrays (as in KGEModel.test)
                hh, rr, tt = (torch.from_numpy(a[i : i + chunk]) for a in (h, r, t))
                c = hh.numel()
                scores = model(
                    hh.repeat_interleave(N),
                    rr.repeat_interleave(N),
                    torch.arange(N).repeat(c),
                ).view(c, N)
                true = scores.gather(1, tt[:, None])
                out.append((scores > true).sum(1).numpy())
            return np.concatenate(out)

    def metrics(split):
        rk = ranks(*evals[split]).astype(np.float64)
        return float(np.mean(1 / (rk + 1))), float(np.mean(rk < 10))

    selector, losses = Selector(), []
    for epoch in range(1, args.epochs + 1):
        losses.append(train_epoch())
        if epoch % 5 == 0 or epoch == args.epochs:
            val_mrr, _ = metrics("val")
            mrr, hits = metrics("test")
            selector.add(epoch, val_mrr, mrr, 0.0, hits10=hits)
    return {"metric": "mrr", "loss_curve": losses, **selector.summary()}


# ---------------------------------------------------------------------------
# EdgeCNN / DynamicEdgeConv on synthetic point clouds
# ---------------------------------------------------------------------------


def _torch_knn_edges(x, batch, k):
    """Edges (neighbor -> point) from the k nearest points of each point in
    its own cloud, the point itself included: PyG's ``DynamicEdgeConv``
    graph (``knn(x, x, k, batch, batch).flip(0)``), which needs ``pyg-lib``
    or ``torch-cluster``, neither of which installs on this machine."""
    import torch

    d = torch.cdist(x, x)
    d = d.masked_fill(batch[:, None] != batch[None, :], float("inf"))
    nbr = d.topk(k, dim=1, largest=False).indices  # [N, k]
    dst = torch.arange(x.size(0)).repeat_interleave(k)
    return torch.stack([nbr.reshape(-1), dst])


def run_pointcloud(args):
    """4-class shape classification of 64-point clouds (``point_clouds``):
    ``edgecnn`` = EdgeCNN(3, 32, 3 layers) on the fixed 8-NN graph,
    ``dynedge`` = 2 x DynamicEdgeConv(MLP, k=8) recomputing the graph in
    feature space; global max pooling + Linear; Adam(lr=0.001), batches of
    32, 80/10/10 split."""
    data = graph_data.point_clouds()
    pos, y, ei = data["pos"], data["y"], data["edge_index"]
    splits = graph_data.split_indices(args.seed, len(y))
    n_pts = pos.shape[1]
    per_cloud = ei.shape[1] // len(y)
    H, k = 32, 8

    def cloud(i):
        return pos[i], ei[:, i * per_cloud : (i + 1) * per_cloud], y[i : i + 1]

    if args.framework == "mlx":
        import mlx.core as mx
        import mlx.nn as nn

        import mlx_graphs.nn as gnn
        from mlx_graphs.data import GraphData
        from mlx_graphs.loaders import Dataloader

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                if args.model == "edgecnn":
                    self.gnn = gnn.EdgeCNN(3, H, num_layers=3, out_features_dim=H)
                else:
                    self.convs = [
                        gnn.DynamicEdgeConv(gnn.MLP([6, H, H]), k=k),
                        gnn.DynamicEdgeConv(gnn.MLP([2 * H, H, H]), k=k),
                    ]
                self.lin = nn.Linear(H, 4)

            def __call__(self, b):
                if args.model == "edgecnn":
                    h = self.gnn(b.edge_index, b.node_features)
                else:
                    h = b.node_features
                    for i, conv in enumerate(self.convs):
                        h = conv(h, b.batch_indices)
                        h = nn.relu(h) if i == 0 else h
                pooled = gnn.global_max_pool(h, b.batch_indices, b.num_graphs)
                return self.lin(pooled)

        def graph(i):
            p, e, label = cloud(i)
            return GraphData(
                edge_index=mx.array(e.astype(np.int32)),
                node_features=mx.array(p),
                graph_labels=mx.array(label.astype(np.int32)),
            )

        sets = [[graph(i) for i in s] for s in splits]
        train_loader = Dataloader(sets[0], batch_size=32, shuffle=True)
        eval_loaders = [Dataloader(s, batch_size=100) for s in sets[1:]]
        model = Net()
        optimizer, _ = mlx_adam(0.001)

        def loss_fn(model, b):
            return nn.losses.cross_entropy(
                model(b), b.graph_labels.reshape(-1), reduction="mean"
            )

        loss_and_grad = nn.value_and_grad(model, loss_fn)

        def train_epoch():
            model.train()
            total = 0.0
            for b in train_loader:
                loss, grads = loss_and_grad(model, b)
                optimizer.update(model, grads)
                mx.eval(loss, model.parameters(), optimizer.state)
                total += loss.item() * b.num_graphs
            return total / len(sets[0])

        def evaluate(loader):
            model.eval()
            correct, loss, n = 0, 0.0, 0
            for b in loader:
                logits = model(b)
                t = b.graph_labels.reshape(-1)
                correct += (mx.argmax(logits, axis=1) == t).sum().item()
                loss += nn.losses.cross_entropy(logits, t, reduction="sum").item()
                n += b.num_graphs
            return correct / n, loss / n

    else:
        import torch
        import torch.nn.functional as F
        import torch_geometric.nn as pnn
        from torch_geometric.data import Data
        from torch_geometric.loader import DataLoader

        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                if args.model == "edgecnn":
                    self.gnn = pnn.models.EdgeCNN(3, H, num_layers=3, out_channels=H)
                else:
                    self.convs = torch.nn.ModuleList(
                        [
                            pnn.EdgeConv(pnn.MLP([6, H, H], norm=None)),
                            pnn.EdgeConv(pnn.MLP([2 * H, H, H], norm=None)),
                        ]
                    )
                self.lin = torch.nn.Linear(H, 4)

            def forward(self, b):
                if args.model == "edgecnn":
                    h = self.gnn(b.x, b.edge_index)
                else:
                    h = b.x
                    for i, conv in enumerate(self.convs):
                        h = conv(h, _torch_knn_edges(h.detach(), b.batch, k))
                        h = h.relu() if i == 0 else h
                return self.lin(pnn.global_max_pool(h, b.batch, b.num_graphs))

        def graph(i):
            p, e, label = cloud(i)
            return Data(
                x=torch.from_numpy(p),
                edge_index=torch.from_numpy(e),
                y=torch.from_numpy(label),
                num_nodes=n_pts,
            )

        sets = [[graph(i) for i in s] for s in splits]
        train_loader = DataLoader(sets[0], batch_size=32, shuffle=True)
        eval_loaders = [DataLoader(s, batch_size=100) for s in sets[1:]]
        model = Net()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        def train_epoch():
            model.train()
            total = 0.0
            for b in train_loader:
                optimizer.zero_grad()
                loss = F.cross_entropy(model(b), b.y)
                loss.backward()
                optimizer.step()
                total += loss.item() * b.num_graphs
            return total / len(sets[0])

        @torch.no_grad()
        def evaluate(loader):
            model.eval()
            correct, loss, n = 0, 0.0, 0
            for b in loader:
                logits = model(b)
                correct += (logits.argmax(1) == b.y).sum().item()
                loss += F.cross_entropy(logits, b.y, reduction="sum").item()
                n += b.num_graphs
            return correct / n, loss / n

    selector, losses = Selector(), []
    for epoch in range(1, args.epochs + 1):
        losses.append(train_epoch())
        val_acc, val_loss = evaluate(eval_loaders[0])
        test_acc, _ = evaluate(eval_loaders[1])
        selector.add(epoch, val_acc, test_acc, val_loss)
    return {"metric": "accuracy", "loss_curve": losses, **selector.summary()}
