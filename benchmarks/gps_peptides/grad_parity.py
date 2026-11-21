"""Train-mode loss/gradient parity of the full GPS model (example vs PyG twin)
on a few Peptides graphs, with the weights copied from PyG."""

import os
import sys

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
from mlx.utils import tree_flatten

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(__file__))
from examples.graph_gps_peptides import GPS, loss_fn  # noqa
from gps_pyg import GPS as PyGGPS  # noqa
from gps_pyg import load_split  # noqa
from mlx_graphs.data import GraphData  # noqa
from mlx_graphs.data.batch import GraphDataBatch  # noqa
from tests.pyg_copy import load_torch, torch_state  # noqa
from torch_geometric.loader import DataLoader  # noqa

data, av, bv = load_split(f"{sys.argv[1]}/val.npz")
data = data[:16]
torch.manual_seed(0)
pyg = PyGGPS(av.tolist(), bv.tolist(), 32, 8, 2, 4, 0.0, 0.0)
model = GPS(av.tolist(), bv.tolist(), 32, 8, 2, 10, heads=4, attn_dropout=0.0)
state = torch_state(
    pyg,
    renames=[
        (r"^(node_emb|edge_emb)\.embeddings\.(\d+)\.", r"\1.embeddings.\2."),
        (r"^pe_lin\.", "pe_encoder.lin."),
        (r"^pe_norm\.", "pe_encoder.norm."),
        (r"^convs\.(\d+)\.mlp\.(\d)\.", r"convs.\1.mlp.layers.\2."),
        (r"^convs\.(\d+)\.conv\.nn\.(\d)\.", r"convs.\1.conv.mlp.layers.\2."),
        (r"^mlp\.(\d)\.", r"mlp.layers.\1."),
    ],
    drop=[r"conv\.eps$"],
)
load_torch(model, state)

batch_t = next(iter(DataLoader(data, 16)))
graphs = [
    GraphData(
        edge_index=mx.array(d.edge_index.numpy().astype(np.int32)),
        node_features=mx.array(d.x.numpy().astype(np.int32)),
        edge_features=mx.array(d.edge_attr.numpy().astype(np.int32)),
        graph_labels=mx.array(d.y.numpy()),
        random_walk_pe=mx.array(d.pe.numpy()),
    )
    for d in data
]
batch = GraphDataBatch(graphs)

pyg.train()
model.train()
out = pyg(batch_t.x, batch_t.pe, batch_t.edge_index, batch_t.edge_attr, batch_t.batch)
loss_t = torch.nn.BCEWithLogitsLoss()(out, batch_t.y)
loss_t.backward()
loss, grads = nn.value_and_grad(model, loss_fn)(model, batch)
print("loss", loss.item(), loss_t.item())
named = dict(pyg.named_parameters())
mg = dict(tree_flatten(grads))
# compare a few key gradients
pairs = [
    ("pe_lin.weight", "pe_encoder.lin.weight"),
    ("convs.0.conv.nn.0.weight", "convs.0.conv.mlp.layers.0.weight"),
    ("convs.1.mlp.0.weight", "convs.1.mlp.layers.0.weight"),
    ("convs.0.norm1.module.weight", "convs.0.norm1.module.weight"),
    ("mlp.0.weight", "mlp.layers.0.weight"),
    ("node_emb.embeddings.0.weight", "node_emb.embeddings.0.weight"),
    ("convs.1.attn.out_proj.weight", "convs.1.attn.out_proj.weight"),
]
for t, m in pairs:
    a, b = named[t].grad.numpy(), np.array(mg[m])
    print(f"{m:40s} rel err {np.abs(a - b).max() / (np.abs(a).max() + 1e-12):.2e}")
qkv = named["convs.0.attn.in_proj_weight"].grad.numpy()[:32]
q_grad = np.array(mg["convs.0.attn.query_proj.weight"])
print("q_proj rel err", np.abs(qkv - q_grad).max() / np.abs(qkv).max())
