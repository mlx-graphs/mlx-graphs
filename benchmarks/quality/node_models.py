"""Node classification models, built the same way in mlx-graphs and in PyG.

Every configuration of ``node_quality.py`` names a pair of builders here:
``MLX[name](info, hp)`` and ``PYG[name](info, hp)``, which return a model
called as ``model(g)`` where ``g`` is a dict of the graph's arrays in the
framework (``x``, ``ei``, ``ea``, ``y``, ``visible``, ...; see
``node_worker.py``). Both models of a pair have the same architecture, the
same hyperparameters and, when a parameter group has its own weight decay,
the same top-level attribute names (e.g. ``convs`` and ``lins``).

Each framework initializes its layers with its own defaults: that is part
of what the comparison checks.
"""

from types import SimpleNamespace

MLX = {}
PYG = {}


def register(name, framework):
    table = MLX if framework == "mlx" else PYG

    def wrap(fn):
        table[name] = fn
        return fn

    return wrap


PNA_AGGREGATORS = ["mean", "min", "max", "std"]
PNA_SCALERS = ["identity", "amplification", "attenuation"]


# ---------------------------------------------------------------------------
# MLX
# ---------------------------------------------------------------------------


def _mlx():
    import mlx.core as mx
    import mlx.nn as nn

    import mlx_graphs.nn as gnn

    return mx, nn, gnn


def mlx_stack(convs, dropout, pre=None, post=None, norms=None, edge_kw=None):
    """dropout -> [pre -> relu -> dropout] -> conv (-> norm -> relu ->
    dropout -> conv)* [-> relu -> dropout -> post].

    ``edge_kw`` maps a conv keyword to a key of ``g`` (e.g. edge features).
    """
    mx, nn, gnn = _mlx()

    class Stack(nn.Module):
        def __init__(self):
            super().__init__()
            self.convs = list(convs)
            if pre is not None:
                self.pre = pre
            if post is not None:
                self.post = post
            if norms is not None:
                self.norms = list(norms)
            self.drop = nn.Dropout(dropout)

        def __call__(self, g):
            kw = {k: g[v] for k, v in (edge_kw or {}).items()}
            x = self.drop(g["x"])
            if "pre" in self:
                x = self.drop(nn.relu(self.pre(x)))
            n = len(self.convs)
            for i, conv in enumerate(self.convs):
                x = conv(g["ei"], x, **kw)
                if i < n - 1 or "post" in self:
                    if "norms" in self and i < len(self.norms):
                        x = self.norms[i](x)
                    x = self.drop(nn.relu(x))
            if "post" in self:
                x = self.post(x)
            return x

    return Stack()


def _two(build, info, hp, **kw):
    """Two convolutions: in -> hidden -> classes."""
    return mlx_stack(
        [build(info.in_dim, hp.hidden), build(hp.hidden, info.num_classes)],
        hp.dropout,
        **kw,
    )


@register("gcn", "mlx")
def _(info, hp):
    _, _, gnn = _mlx()
    return _two(lambda a, b: gnn.GCNConv(a, b, add_self_loops=True), info, hp)


@register("gcn2", "mlx")
def _(info, hp):
    mx, nn, gnn = _mlx()

    class GCN2(nn.Module):
        def __init__(self):
            super().__init__()
            self.lins = [
                nn.Linear(info.in_dim, hp.hidden),
                nn.Linear(hp.hidden, info.num_classes),
            ]
            self.convs = [
                gnn.GCN2Conv(
                    hp.hidden, alpha=hp.alpha, theta=hp.theta, layer=i + 1,
                    shared_weights=True, cached=True,
                )
                for i in range(hp.layers)
            ]  # fmt: skip
            self.drop = nn.Dropout(hp.dropout)

        def __call__(self, g):
            x = self.drop(g["x"])
            x = x0 = nn.relu(self.lins[0](x))
            for conv in self.convs:
                x = self.drop(x)
                x = nn.relu(conv(g["ei"], x, x0))
            return self.lins[1](self.drop(x))

    return GCN2()


@register("ssg", "mlx")
def _(info, hp):
    _, _, gnn = _mlx()
    conv = gnn.SSGConv(info.in_dim, info.num_classes, alpha=hp.alpha, K=hp.K)
    return mlx_stack([conv], hp.dropout)


@register("lg", "mlx")
def _(info, hp):
    mx, nn, gnn = _mlx()

    class LG(nn.Module):
        def __init__(self):
            super().__init__()
            self.props = [gnn.LGConv() for _ in range(hp.K)]
            self.lins = [
                nn.Linear(info.in_dim, hp.hidden),
                nn.Linear(hp.hidden, info.num_classes),
            ]
            self.drop = nn.Dropout(hp.dropout)

        def __call__(self, g):
            x = g["x"]
            for prop in self.props:
                x = prop(g["ei"], x)
            x = self.drop(nn.relu(self.lins[0](self.drop(x))))
            return self.lins[1](x)

    return LG()


@register("arma", "mlx")
def _(info, hp):
    _, _, gnn = _mlx()
    kw = dict(num_stacks=3, num_layers=2, shared_weights=True, dropout=0.25)
    return mlx_stack(
        [
            gnn.ARMAConv(info.in_dim, hp.hidden, **kw),
            gnn.ARMAConv(hp.hidden, info.num_classes, act=None, **kw),
        ],
        hp.dropout,
    )


@register("gated", "mlx")
def _(info, hp):
    mx, nn, gnn = _mlx()
    return mlx_stack(
        [gnn.GatedGraphConv(hp.hidden, num_layers=2)],
        hp.dropout,
        pre=nn.Linear(info.in_dim, hp.hidden),
        post=nn.Linear(hp.hidden, info.num_classes),
    )


@register("resgated", "mlx")
def _(info, hp):
    _, _, gnn = _mlx()
    return _two(gnn.ResGatedGraphConv, info, hp)


@register("nn", "mlx")
def _(info, hp):
    mx, nn, gnn = _mlx()
    h = hp.hidden

    def conv():
        mlp = nn.Sequential(
            nn.Linear(info.edge_dim, 32), nn.ReLU(), nn.Linear(32, h * h)
        )
        return gnn.NNConv(h, h, mlp, aggr="mean")

    return mlx_stack(
        [conv()],
        hp.dropout,
        pre=nn.Linear(info.in_dim, h),
        post=nn.Linear(h, info.num_classes),
        edge_kw={"edge_features": "ea"},
    )


@register("cg", "mlx")
def _(info, hp):
    mx, nn, gnn = _mlx()
    return mlx_stack(
        [gnn.CGConv(hp.hidden, edge_features_dim=info.edge_dim, aggr="mean")],
        hp.dropout,
        pre=nn.Linear(info.in_dim, hp.hidden),
        post=nn.Linear(hp.hidden, info.num_classes),
        edge_kw={"edge_features": "ea"},
    )


@register("mf", "mlx")
def _(info, hp):
    _, _, gnn = _mlx()
    return _two(gnn.MFConv, info, hp)


@register("le", "mlx")
def _(info, hp):
    _, _, gnn = _mlx()
    return _two(gnn.LEConv, info, hp)


@register("film", "mlx")
def _(info, hp):
    _, _, gnn = _mlx()
    return _two(gnn.FiLMConv, info, hp)


@register("agnn", "mlx")
def _(info, hp):
    mx, nn, gnn = _mlx()

    class AGNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin1 = nn.Linear(info.in_dim, hp.hidden)
            self.prop1 = gnn.AGNNConv(requires_grad=False)
            self.prop2 = gnn.AGNNConv(requires_grad=True)
            self.lin2 = nn.Linear(hp.hidden, info.num_classes)
            self.drop = nn.Dropout(hp.dropout)

        def __call__(self, g):
            x = nn.relu(self.lin1(self.drop(g["x"])))
            x = self.prop2(g["ei"], self.prop1(g["ei"], x))
            return self.lin2(self.drop(x))

    return AGNN()


@register("clustergcn", "mlx")
def _(info, hp):
    _, _, gnn = _mlx()
    return _two(gnn.ClusterGCNConv, info, hp)


def _mlx_gen(hp, aggr="softmax", **kw):
    _, _, gnn = _mlx()
    if aggr == "softmax":
        kw.update(t=1.0, learn_t=True)
    return gnn.GENConv(
        hp.hidden, hp.hidden, aggr=aggr, num_layers=2, norm="layer", **kw
    )


@register("gen", "mlx")
def _(info, hp):
    mx, nn, gnn = _mlx()
    return mlx_stack(
        [_mlx_gen(hp), _mlx_gen(hp)],
        hp.dropout,
        pre=nn.Linear(info.in_dim, hp.hidden),
        post=nn.Linear(hp.hidden, info.num_classes),
    )


@register("gen_multi", "mlx")
def _(info, hp):
    mx, nn, gnn = _mlx()
    aggr = ["softmax", "mean", "max"]
    return mlx_stack(
        [_mlx_gen(hp, aggr=aggr), _mlx_gen(hp, aggr=aggr)],
        hp.dropout,
        pre=nn.Linear(info.in_dim, hp.hidden),
        post=nn.Linear(hp.hidden, info.num_classes),
    )


@register("gen_msgnorm", "mlx")
def _(info, hp):
    mx, nn, gnn = _mlx()
    kw = dict(msg_norm=True, learn_msg_scale=True)
    return mlx_stack(
        [_mlx_gen(hp, **kw), _mlx_gen(hp, **kw)],
        hp.dropout,
        pre=nn.Linear(info.in_dim, hp.hidden),
        post=nn.Linear(hp.hidden, info.num_classes),
    )


@register("mixhop", "mlx")
def _(info, hp):
    mx, nn, gnn = _mlx()
    p = [0, 1, 2]
    return mlx_stack(
        [
            gnn.MixHopConv(info.in_dim, hp.hidden, powers=p),
            gnn.MixHopConv(len(p) * hp.hidden, hp.hidden, powers=p),
        ],
        hp.dropout,
        post=nn.Linear(len(p) * hp.hidden, info.num_classes),
    )


@register("fa", "mlx")
def _(info, hp):
    mx, nn, gnn = _mlx()

    class FAGCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.lins = [
                nn.Linear(info.in_dim, hp.hidden),
                nn.Linear(hp.hidden, info.num_classes),
            ]
            self.convs = [
                gnn.FAConv(hp.hidden, eps=hp.eps, dropout=hp.att_dropout)
                for _ in range(hp.layers)
            ]
            self.drop = nn.Dropout(hp.dropout)

        def __call__(self, g):
            x = self.drop(g["x"])
            x = x0 = self.drop(nn.relu(self.lins[0](x)))
            for conv in self.convs:
                x = conv(g["ei"], x, x0)
            return self.lins[1](x)

    return FAGCN()


@register("general", "mlx")
def _(info, hp):
    _, _, gnn = _mlx()
    return _two(lambda a, b: gnn.GeneralConv(a, b, aggr="mean"), info, hp)


@register("general_att", "mlx")
def _(info, hp):
    _, _, gnn = _mlx()
    return _two(
        lambda a, b: gnn.GeneralConv(a, b, aggr="add", attention=True, heads=1),
        info,
        hp,
    )


@register("dir_gcn", "mlx")
def _(info, hp):
    _, _, gnn = _mlx()
    return _two(
        lambda a, b: gnn.DirGNNConv(lambda: gnn.GCNConv(a, b, add_self_loops=True)),
        info,
        hp,
    )


@register("dir_sage", "mlx")
def _(info, hp):
    _, _, gnn = _mlx()
    return _two(lambda a, b: gnn.DirGNNConv(lambda: gnn.SAGEConv(a, b)), info, hp)


@register("pna_conv", "mlx")
def _(info, hp):
    mx, nn, gnn = _mlx()

    def conv(a, b):
        return gnn.PNAConv(
            a, b, aggregators=PNA_AGGREGATORS, scalers=PNA_SCALERS, deg=info.deg
        )

    # A linear encoder first: PNAConv's pre-layer works on edge-level
    # [x_i || x_j], which on 3703 raw CiteSeer features takes ~4 GB.
    return mlx_stack(
        [conv(hp.hidden, hp.hidden), conv(hp.hidden, info.num_classes)],
        hp.dropout,
        pre=nn.Linear(info.in_dim, hp.hidden),
    )


@register("sage_multi", "mlx")
def _(info, hp):
    _, _, gnn = _mlx()
    return _two(lambda a, b: gnn.SAGEConv(a, b, aggr=["mean", "max"]), info, hp)


def _mlx_deep_gcn(info, hp, norm):
    """4 GCN layers with a normalization after each hidden one."""
    _, _, gnn = _mlx()
    dims = [info.in_dim] + [hp.hidden] * (hp.layers - 1) + [info.num_classes]
    convs = [
        gnn.GCNConv(a, b, add_self_loops=True) for a, b in zip(dims[:-1], dims[1:])
    ]
    return mlx_stack(convs, hp.dropout, norms=[norm() for _ in range(hp.layers - 1)])


@register("graphnorm", "mlx")
def _(info, hp):
    _, _, gnn = _mlx()
    return _mlx_deep_gcn(info, hp, lambda: gnn.GraphNorm(hp.hidden))


@register("pairnorm", "mlx")
def _(info, hp):
    _, _, gnn = _mlx()
    return _mlx_deep_gcn(info, hp, lambda: gnn.PairNorm())


@register("diffgroupnorm", "mlx")
def _(info, hp):
    _, _, gnn = _mlx()
    return _mlx_deep_gcn(
        info, hp, lambda: gnn.DiffGroupNorm(hp.hidden, groups=info.num_classes)
    )


@register("nonorm", "mlx")
def _(info, hp):
    mx, nn, gnn = _mlx()
    return _mlx_deep_gcn(info, hp, lambda: nn.Identity())


@register("mlp", "mlx")
def _(info, hp):
    mx, nn, gnn = _mlx()

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.lins = [
                nn.Linear(info.in_dim, hp.hidden),
                nn.Linear(hp.hidden, info.num_classes),
            ]
            self.drop = nn.Dropout(hp.dropout)

        def __call__(self, g):
            x = self.drop(nn.relu(self.lins[0](self.drop(g["x"]))))
            return self.lins[1](x)

    return MLP()


@register("linkx", "mlx")
def _(info, hp):
    mx, nn, gnn = _mlx()

    class Wrap(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = gnn.LINKX(
                info.num_nodes, info.in_dim, hp.hidden, info.num_classes,
                num_layers=2, dropout=hp.dropout,
            )  # fmt: skip

        def __call__(self, g):
            return self.model(g["ei"], g["x"])

    return Wrap()


@register("deepgcn", "mlx")
def _(info, hp):
    mx, nn, gnn = _mlx()

    class DeeperGCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(info.in_dim, hp.hidden)
            self.layers = [
                gnn.DeepGCNLayer(
                    _mlx_gen(hp), nn.LayerNorm(hp.hidden), nn.relu,
                    block="res+", dropout=hp.layer_dropout,
                )
                for _ in range(hp.layers)
            ]  # fmt: skip
            self.lin = nn.Linear(hp.hidden, info.num_classes)
            self.drop = nn.Dropout(hp.dropout)

        def __call__(self, g):
            x = self.encoder(self.drop(g["x"]))
            x = self.layers[0].conv(g["ei"], x)
            for layer in self.layers[1:]:
                x = layer(g["ei"], x)
            x = self.layers[0].act(self.layers[0].norm(x))
            return self.lin(self.drop(x))

    return DeeperGCN()


@register("masklabel", "mlx")
def _(info, hp):
    mx, nn, gnn = _mlx()

    class UniMPLike(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(info.in_dim, hp.hidden)
            self.label = gnn.MaskLabel(info.num_classes, hp.hidden)
            self.convs = [
                gnn.GCNConv(hp.hidden, hp.hidden, add_self_loops=True),
                gnn.GCNConv(hp.hidden, info.num_classes, add_self_loops=True),
            ]
            self.drop = nn.Dropout(hp.dropout)

        def __call__(self, g):
            x = self.lin(self.drop(g["x"]))
            x = self.label(x, g["y"], g["visible"])
            x = self.drop(nn.relu(self.convs[0](g["ei"], x)))
            return self.convs[1](g["ei"], x)

    return UniMPLike()


@register("pna", "mlx")
def _(info, hp):
    mx, nn, gnn = _mlx()

    class Wrap(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(info.in_dim, hp.hidden)
            self.model = gnn.PNA(
                hp.hidden, hp.hidden, num_layers=2,
                out_features_dim=info.num_classes, dropout=hp.dropout,
                aggregators=PNA_AGGREGATORS, scalers=PNA_SCALERS, deg=info.deg,
            )  # fmt: skip
            self.drop = nn.Dropout(hp.dropout)

        def __call__(self, g):
            x = self.drop(nn.relu(self.encoder(self.drop(g["x"]))))
            return self.model(g["ei"], x)

    return Wrap()


@register("sign_mlp", "mlx")
def _(info, hp):
    mx, nn, gnn = _mlx()

    class SIGN(nn.Module):
        def __init__(self):
            super().__init__()
            self.lins = [nn.Linear(info.in_dim, hp.hidden) for _ in range(hp.K + 1)]
            self.lin = nn.Linear((hp.K + 1) * hp.hidden, info.num_classes)
            self.drop = nn.Dropout(hp.dropout)

        def __call__(self, g):
            hs = [
                self.drop(nn.relu(lin(g["xs"][i]))) for i, lin in enumerate(self.lins)
            ]
            return self.lin(mx.concatenate(hs, axis=-1))

    return SIGN()


# ---------------------------------------------------------------------------
# PyG
# ---------------------------------------------------------------------------


def _pyg():
    import torch
    import torch.nn.functional as F
    import torch_geometric.nn as pnn

    return torch, F, pnn


def pyg_stack(convs, dropout, pre=None, post=None, norms=None, edge_kw=None):
    torch, F, pnn = _pyg()

    class Stack(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.convs = torch.nn.ModuleList(convs)
            self.pre, self.post = pre, post
            self.norms = None if norms is None else torch.nn.ModuleList(norms)
            self.p = dropout

        def forward(self, g):
            kw = {k: g[v] for k, v in (edge_kw or {}).items()}
            x = F.dropout(g["x"], self.p, self.training)
            if self.pre is not None:
                x = F.dropout(F.relu(self.pre(x)), self.p, self.training)
            n = len(self.convs)
            for i, conv in enumerate(self.convs):
                x = conv(x, g["ei"], **kw)
                if i < n - 1 or self.post is not None:
                    if self.norms is not None and i < len(self.norms):
                        x = self.norms[i](x)
                    x = F.dropout(F.relu(x), self.p, self.training)
            if self.post is not None:
                x = self.post(x)
            return x

    return Stack()


def _ptwo(build, info, hp, **kw):
    return pyg_stack(
        [build(info.in_dim, hp.hidden), build(hp.hidden, info.num_classes)],
        hp.dropout,
        **kw,
    )


@register("gcn", "pyg")
def _(info, hp):
    _, _, pnn = _pyg()
    return _ptwo(lambda a, b: pnn.GCNConv(a, b, cached=True), info, hp)


@register("gcn2", "pyg")
def _(info, hp):
    torch, F, pnn = _pyg()

    class GCN2(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lins = torch.nn.ModuleList(
                [
                    torch.nn.Linear(info.in_dim, hp.hidden),
                    torch.nn.Linear(hp.hidden, info.num_classes),
                ]
            )
            self.convs = torch.nn.ModuleList(
                [
                    pnn.GCN2Conv(
                        hp.hidden, alpha=hp.alpha, theta=hp.theta, layer=i + 1,
                        shared_weights=True, cached=True,
                    )
                    for i in range(hp.layers)
                ]
            )  # fmt: skip
            self.p = hp.dropout

        def forward(self, g):
            x = F.dropout(g["x"], self.p, self.training)
            x = x0 = F.relu(self.lins[0](x))
            for conv in self.convs:
                x = F.dropout(x, self.p, self.training)
                x = F.relu(conv(x, x0, g["ei"]))
            return self.lins[1](F.dropout(x, self.p, self.training))

    return GCN2()


@register("ssg", "pyg")
def _(info, hp):
    _, _, pnn = _pyg()
    conv = pnn.SSGConv(info.in_dim, info.num_classes, alpha=hp.alpha, K=hp.K)
    return pyg_stack([conv], hp.dropout)


@register("lg", "pyg")
def _(info, hp):
    torch, F, pnn = _pyg()

    class LG(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.props = torch.nn.ModuleList([pnn.LGConv() for _ in range(hp.K)])
            self.lins = torch.nn.ModuleList(
                [
                    torch.nn.Linear(info.in_dim, hp.hidden),
                    torch.nn.Linear(hp.hidden, info.num_classes),
                ]
            )
            self.p = hp.dropout

        def forward(self, g):
            x = g["x"]
            for prop in self.props:
                x = prop(x, g["ei"])
            x = F.dropout(x, self.p, self.training)
            x = F.dropout(F.relu(self.lins[0](x)), self.p, self.training)
            return self.lins[1](x)

    return LG()


@register("arma", "pyg")
def _(info, hp):
    _, _, pnn = _pyg()
    kw = dict(num_stacks=3, num_layers=2, shared_weights=True, dropout=0.25)
    return pyg_stack(
        [
            pnn.ARMAConv(info.in_dim, hp.hidden, **kw),
            pnn.ARMAConv(hp.hidden, info.num_classes, act=None, **kw),
        ],
        hp.dropout,
    )


@register("gated", "pyg")
def _(info, hp):
    torch, F, pnn = _pyg()
    return pyg_stack(
        [pnn.GatedGraphConv(hp.hidden, num_layers=2)],
        hp.dropout,
        pre=torch.nn.Linear(info.in_dim, hp.hidden),
        post=torch.nn.Linear(hp.hidden, info.num_classes),
    )


@register("resgated", "pyg")
def _(info, hp):
    _, _, pnn = _pyg()
    return _ptwo(pnn.ResGatedGraphConv, info, hp)


@register("nn", "pyg")
def _(info, hp):
    torch, F, pnn = _pyg()
    h = hp.hidden
    mlp = torch.nn.Sequential(
        torch.nn.Linear(info.edge_dim, 32), torch.nn.ReLU(), torch.nn.Linear(32, h * h)
    )
    return pyg_stack(
        [pnn.NNConv(h, h, mlp, aggr="mean")],
        hp.dropout,
        pre=torch.nn.Linear(info.in_dim, h),
        post=torch.nn.Linear(h, info.num_classes),
        edge_kw={"edge_attr": "ea"},
    )


@register("cg", "pyg")
def _(info, hp):
    torch, F, pnn = _pyg()
    return pyg_stack(
        [pnn.CGConv(hp.hidden, dim=info.edge_dim, aggr="mean")],
        hp.dropout,
        pre=torch.nn.Linear(info.in_dim, hp.hidden),
        post=torch.nn.Linear(hp.hidden, info.num_classes),
        edge_kw={"edge_attr": "ea"},
    )


@register("mf", "pyg")
def _(info, hp):
    _, _, pnn = _pyg()
    return _ptwo(pnn.MFConv, info, hp)


@register("le", "pyg")
def _(info, hp):
    _, _, pnn = _pyg()
    return _ptwo(pnn.LEConv, info, hp)


@register("film", "pyg")
def _(info, hp):
    _, _, pnn = _pyg()
    return _ptwo(pnn.FiLMConv, info, hp)


@register("agnn", "pyg")
def _(info, hp):
    torch, F, pnn = _pyg()

    class AGNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin1 = torch.nn.Linear(info.in_dim, hp.hidden)
            self.prop1 = pnn.AGNNConv(requires_grad=False)
            self.prop2 = pnn.AGNNConv(requires_grad=True)
            self.lin2 = torch.nn.Linear(hp.hidden, info.num_classes)
            self.p = hp.dropout

        def forward(self, g):
            x = F.dropout(g["x"], self.p, self.training)
            x = F.relu(self.lin1(x))
            x = self.prop2(self.prop1(x, g["ei"]), g["ei"])
            return self.lin2(F.dropout(x, self.p, self.training))

    return AGNN()


@register("clustergcn", "pyg")
def _(info, hp):
    _, _, pnn = _pyg()
    return _ptwo(pnn.ClusterGCNConv, info, hp)


def _pyg_gen(hp, aggr="softmax", **kw):
    _, _, pnn = _pyg()
    if aggr == "softmax":
        kw.update(t=1.0, learn_t=True)
    return pnn.GENConv(
        hp.hidden, hp.hidden, aggr=aggr, num_layers=2, norm="layer", **kw
    )


@register("gen", "pyg")
def _(info, hp):
    torch, F, pnn = _pyg()
    return pyg_stack(
        [_pyg_gen(hp), _pyg_gen(hp)],
        hp.dropout,
        pre=torch.nn.Linear(info.in_dim, hp.hidden),
        post=torch.nn.Linear(hp.hidden, info.num_classes),
    )


@register("gen_multi", "pyg")
def _(info, hp):
    torch, F, pnn = _pyg()
    aggr = ["softmax", "mean", "max"]
    return pyg_stack(
        [_pyg_gen(hp, aggr=aggr), _pyg_gen(hp, aggr=aggr)],
        hp.dropout,
        pre=torch.nn.Linear(info.in_dim, hp.hidden),
        post=torch.nn.Linear(hp.hidden, info.num_classes),
    )


@register("gen_msgnorm", "pyg")
def _(info, hp):
    torch, F, pnn = _pyg()
    kw = dict(msg_norm=True, learn_msg_scale=True)
    return pyg_stack(
        [_pyg_gen(hp, **kw), _pyg_gen(hp, **kw)],
        hp.dropout,
        pre=torch.nn.Linear(info.in_dim, hp.hidden),
        post=torch.nn.Linear(hp.hidden, info.num_classes),
    )


@register("mixhop", "pyg")
def _(info, hp):
    torch, F, pnn = _pyg()
    p = [0, 1, 2]
    return pyg_stack(
        [
            pnn.MixHopConv(info.in_dim, hp.hidden, powers=p),
            pnn.MixHopConv(len(p) * hp.hidden, hp.hidden, powers=p),
        ],
        hp.dropout,
        post=torch.nn.Linear(len(p) * hp.hidden, info.num_classes),
    )


@register("fa", "pyg")
def _(info, hp):
    torch, F, pnn = _pyg()

    class FAGCN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lins = torch.nn.ModuleList(
                [
                    torch.nn.Linear(info.in_dim, hp.hidden),
                    torch.nn.Linear(hp.hidden, info.num_classes),
                ]
            )
            self.convs = torch.nn.ModuleList(
                [
                    pnn.FAConv(hp.hidden, eps=hp.eps, dropout=hp.att_dropout)
                    for _ in range(hp.layers)
                ]
            )
            self.p = hp.dropout

        def forward(self, g):
            x = F.dropout(g["x"], self.p, self.training)
            x = x0 = F.dropout(F.relu(self.lins[0](x)), self.p, self.training)
            for conv in self.convs:
                x = conv(x, x0, g["ei"])
            return self.lins[1](x)

    return FAGCN()


@register("general", "pyg")
def _(info, hp):
    _, _, pnn = _pyg()
    return _ptwo(lambda a, b: pnn.GeneralConv(a, b, aggr="mean"), info, hp)


@register("general_att", "pyg")
def _(info, hp):
    _, _, pnn = _pyg()
    return _ptwo(
        lambda a, b: pnn.GeneralConv(a, b, aggr="add", attention=True, heads=1),
        info,
        hp,
    )


@register("dir_gcn", "pyg")
def _(info, hp):
    _, _, pnn = _pyg()
    return _ptwo(lambda a, b: pnn.DirGNNConv(pnn.GCNConv(a, b)), info, hp)


@register("dir_sage", "pyg")
def _(info, hp):
    _, _, pnn = _pyg()
    return _ptwo(lambda a, b: pnn.DirGNNConv(pnn.SAGEConv(a, b)), info, hp)


@register("pna_conv", "pyg")
def _(info, hp):
    torch, _, pnn = _pyg()
    deg = torch.as_tensor(info.deg)

    def conv(a, b):
        return pnn.PNAConv(
            a, b, aggregators=PNA_AGGREGATORS, scalers=PNA_SCALERS, deg=deg
        )

    return pyg_stack(
        [conv(hp.hidden, hp.hidden), conv(hp.hidden, info.num_classes)],
        hp.dropout,
        pre=torch.nn.Linear(info.in_dim, hp.hidden),
    )


@register("sage_multi", "pyg")
def _(info, hp):
    _, _, pnn = _pyg()
    return _ptwo(lambda a, b: pnn.SAGEConv(a, b, aggr=["mean", "max"]), info, hp)


def _pyg_deep_gcn(info, hp, norm):
    _, _, pnn = _pyg()
    dims = [info.in_dim] + [hp.hidden] * (hp.layers - 1) + [info.num_classes]
    convs = [pnn.GCNConv(a, b, cached=True) for a, b in zip(dims[:-1], dims[1:])]
    return pyg_stack(convs, hp.dropout, norms=[norm() for _ in range(hp.layers - 1)])


@register("graphnorm", "pyg")
def _(info, hp):
    _, _, pnn = _pyg()
    return _pyg_deep_gcn(info, hp, lambda: pnn.GraphNorm(hp.hidden))


@register("pairnorm", "pyg")
def _(info, hp):
    _, _, pnn = _pyg()
    return _pyg_deep_gcn(info, hp, lambda: pnn.PairNorm())


@register("diffgroupnorm", "pyg")
def _(info, hp):
    _, _, pnn = _pyg()
    return _pyg_deep_gcn(
        info, hp, lambda: pnn.DiffGroupNorm(hp.hidden, groups=info.num_classes)
    )


@register("nonorm", "pyg")
def _(info, hp):
    torch, _, _ = _pyg()
    return _pyg_deep_gcn(info, hp, lambda: torch.nn.Identity())


@register("mlp", "pyg")
def _(info, hp):
    torch, F, pnn = _pyg()

    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lins = torch.nn.ModuleList(
                [
                    torch.nn.Linear(info.in_dim, hp.hidden),
                    torch.nn.Linear(hp.hidden, info.num_classes),
                ]
            )
            self.p = hp.dropout

        def forward(self, g):
            x = F.dropout(g["x"], self.p, self.training)
            x = F.dropout(F.relu(self.lins[0](x)), self.p, self.training)
            return self.lins[1](x)

    return MLP()


@register("linkx", "pyg")
def _(info, hp):
    torch, F, pnn = _pyg()

    class Wrap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = pnn.LINKX(
                info.num_nodes, info.in_dim, hp.hidden, info.num_classes,
                num_layers=2, dropout=hp.dropout,
            )  # fmt: skip

        def forward(self, g):
            return self.model(g["x"], g["ei"])

    return Wrap()


@register("deepgcn", "pyg")
def _(info, hp):
    torch, F, pnn = _pyg()

    class DeeperGCN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(info.in_dim, hp.hidden)
            self.layers = torch.nn.ModuleList(
                [
                    pnn.DeepGCNLayer(
                        _pyg_gen(hp), torch.nn.LayerNorm(hp.hidden),
                        torch.nn.ReLU(), block="res+", dropout=hp.layer_dropout,
                    )
                    for _ in range(hp.layers)
                ]
            )  # fmt: skip
            self.lin = torch.nn.Linear(hp.hidden, info.num_classes)
            self.p = hp.dropout

        def forward(self, g):
            x = self.encoder(F.dropout(g["x"], self.p, self.training))
            x = self.layers[0].conv(x, g["ei"])
            for layer in self.layers[1:]:
                x = layer(x, g["ei"])
            x = self.layers[0].act(self.layers[0].norm(x))
            return self.lin(F.dropout(x, self.p, self.training))

    return DeeperGCN()


@register("masklabel", "pyg")
def _(info, hp):
    torch, F, pnn = _pyg()

    class UniMPLike(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(info.in_dim, hp.hidden)
            self.label = pnn.MaskLabel(info.num_classes, hp.hidden)
            self.convs = torch.nn.ModuleList(
                [
                    pnn.GCNConv(hp.hidden, hp.hidden, cached=True),
                    pnn.GCNConv(hp.hidden, info.num_classes, cached=True),
                ]
            )
            self.p = hp.dropout

        def forward(self, g):
            x = self.lin(F.dropout(g["x"], self.p, self.training))
            x = self.label(x, g["y"], g["visible"])
            x = F.dropout(F.relu(self.convs[0](x, g["ei"])), self.p, self.training)
            return self.convs[1](x, g["ei"])

    return UniMPLike()


@register("pna", "pyg")
def _(info, hp):
    torch, F, pnn = _pyg()

    class Wrap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(info.in_dim, hp.hidden)
            self.model = pnn.PNA(
                hp.hidden, hp.hidden, num_layers=2,
                out_channels=info.num_classes, dropout=hp.dropout,
                aggregators=PNA_AGGREGATORS, scalers=PNA_SCALERS,
                deg=torch.as_tensor(info.deg),
            )  # fmt: skip
            self.p = hp.dropout

        def forward(self, g):
            x = F.dropout(g["x"], self.p, self.training)
            x = F.dropout(F.relu(self.encoder(x)), self.p, self.training)
            return self.model(x, g["ei"])

    return Wrap()


@register("sign_mlp", "pyg")
def _(info, hp):
    torch, F, pnn = _pyg()

    class SIGN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lins = torch.nn.ModuleList(
                [torch.nn.Linear(info.in_dim, hp.hidden) for _ in range(hp.K + 1)]
            )
            self.lin = torch.nn.Linear((hp.K + 1) * hp.hidden, info.num_classes)
            self.p = hp.dropout

        def forward(self, g):
            hs = [
                F.dropout(F.relu(lin(g["xs"][i])), self.p, self.training)
                for i, lin in enumerate(self.lins)
            ]
            return self.lin(torch.cat(hs, dim=-1))

    return SIGN()


def namespace(**kw):
    return SimpleNamespace(**kw)
