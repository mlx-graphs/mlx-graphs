"""py-tgb's TGN example for dynamic node property prediction
(``examples/nodeproppred/tgbn-trade/tgn.py`` of github.com/shenyangHuang/TGB),
with PyG on the CPU and --epochs/--seed/--json, for the comparison with
``examples/tgn_tgbn.py``.

As in ``pyg_tgbl.py``, PyG's ``scatter_argmax`` gives empty groups an
out-of-range index as torch-scatter does, the ``NodePredictor`` and
``GraphAttentionEmbedding`` of the TGB repository are copied, and the dataset
and evaluator are py-tgb's (``--data`` holds mlx-graphs' download, e.g.
``tgbn_trade/raw``).
"""

import json
import os
import sys
import time
from argparse import ArgumentParser

import torch
import torch_geometric.nn.models.tgn as tgn_module
from torch.nn import Linear
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader,
)

parser = ArgumentParser()
parser.add_argument("--data", required=True)
parser.add_argument("--dataset", default="tgbn-trade")
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--threads", type=int, default=4)
parser.add_argument("--json", action="store_true")
args = parser.parse_args()


def scatter_argmax(src, index, dim=0, dim_size=None):
    # torch-scatter semantics: empty groups get an out-of-range index
    res = src.new_empty(dim_size)
    res.scatter_reduce_(0, index, src, reduce="amax", include_self=False)
    out = index.new_full((dim_size,), fill_value=src.size(0))
    nonzero = (src == res[index]).nonzero().view(-1)
    out[index[nonzero]] = nonzero
    return out


tgn_module.scatter_argmax = scatter_argmax


def py_tgb_root(data_dir: str, name: str) -> str:
    from tgb.utils.info import PROJ_DIR

    dir_name = name.replace("-", "_")
    raw = os.path.join(data_dir, dir_name, "raw")
    target = os.path.join(data_dir, "py_tgb", dir_name)
    os.makedirs(target, exist_ok=True)
    for f in os.listdir(raw):
        link = os.path.join(target, f)
        if not os.path.exists(link):
            os.symlink(os.path.join(raw, f), link)
    return os.path.relpath(os.path.join(data_dir, "py_tgb"), PROJ_DIR)


from tgb.nodeproppred.dataset_pyg import PyGNodePropPredDataset  # noqa: E402
from tgb.nodeproppred.evaluate import Evaluator  # noqa: E402
from tgb.utils.utils import set_random_seed  # noqa: E402

torch.set_num_threads(args.threads)
torch.manual_seed(args.seed)
set_random_seed(args.seed)
device = torch.device("cpu")
lr, batch_size = 0.0001, 200

name = args.dataset
dataset = PyGNodePropPredDataset(name=name, root=py_tgb_root(args.data, name))
eval_metric = dataset.eval_metric
num_classes = dataset.num_classes
data = dataset.get_TemporalData().to(device)
evaluator = Evaluator(name=name)
train_loader = TemporalDataLoader(data[dataset.train_mask], batch_size=batch_size)
val_loader = TemporalDataLoader(data[dataset.val_mask], batch_size=batch_size)
test_loader = TemporalDataLoader(data[dataset.test_mask], batch_size=batch_size)
neighbor_loader = LastNeighborLoader(data.num_nodes, size=10, device=device)
memory_dim = time_dim = embedding_dim = 100


class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(
            in_channels, out_channels // 2, heads=2, dropout=0.1, edge_dim=edge_dim
        )

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)


class NodePredictor(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin_node = Linear(in_dim, in_dim)
        self.out = Linear(in_dim, out_dim)

    def forward(self, node_embed):
        return self.out(self.lin_node(node_embed).relu())


memory = TGNMemory(
    data.num_nodes, data.msg.size(-1), memory_dim, time_dim,
    message_module=IdentityMessage(data.msg.size(-1), memory_dim, time_dim),
    aggregator_module=LastAggregator(),
).to(device)  # fmt: skip
gnn = GraphAttentionEmbedding(
    in_channels=memory_dim, out_channels=embedding_dim,
    msg_dim=data.msg.size(-1), time_enc=memory.time_enc,
).to(device).float()  # fmt: skip
node_pred = NodePredictor(in_dim=embedding_dim, out_dim=num_classes).to(device)
optimizer = torch.optim.Adam(
    set(memory.parameters()) | set(gnn.parameters()) | set(node_pred.parameters()),
    lr=lr,
)
criterion = torch.nn.CrossEntropyLoss()
assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)


def process_edges(src, dst, t, msg):
    if src.nelement() > 0:
        memory.update_state(src, dst, t, msg)
        neighbor_loader.insert(src, dst)


def predict(n_id):
    n_id_neighbors, mem_edge_index, e_id = neighbor_loader(n_id)
    assoc[n_id_neighbors] = torch.arange(n_id_neighbors.size(0), device=device)
    z, last_update = memory(n_id_neighbors)
    z = gnn(z, last_update, mem_edge_index, data.t[e_id], data.msg[e_id])
    return node_pred(z[assoc[n_id]])


def run(loader, training):
    with torch.set_grad_enabled(training):  # test() is under no_grad in TGB
        return _run(loader, training)


def _run(loader, training):
    for m in (memory, gnn, node_pred):
        m.train(training)
    if training:
        memory.reset_state()
        neighbor_loader.reset_state()
    total_loss, total_score, num_label_ts = 0, 0, 0
    label_t = dataset.get_label_time()
    for batch in loader:
        if training:
            optimizer.zero_grad()
        src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
        query_t = batch.t[-1]
        if query_t > label_t:
            label_tuple = dataset.get_node_label(query_t)
            if label_tuple is None:
                break
            _, label_srcs, labels = label_tuple
            label_t = dataset.get_label_time()
            previous_day_mask = batch.t < label_t
            process_edges(
                src[previous_day_mask], dst[previous_day_mask],
                t[previous_day_mask], msg[previous_day_mask],
            )  # fmt: skip
            src, dst, t, msg = (
                src[~previous_day_mask], dst[~previous_day_mask],
                t[~previous_day_mask], msg[~previous_day_mask],
            )  # fmt: skip
            pred = predict(label_srcs)
            if training:
                loss = criterion(pred, labels)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach())
            score = evaluator.eval(
                {"y_true": labels.numpy(), "y_pred": pred.detach().numpy(),
                 "eval_metric": [eval_metric]}
            )[eval_metric]  # fmt: skip
            total_score += score
            num_label_ts += 1
        process_edges(src, dst, t, msg)
        if training:
            memory.detach()
    out = {"ndcg": total_score / max(num_label_ts, 1)}
    if training:
        out["ce"] = total_loss / max(num_label_ts, 1)
    return out


history = []
for epoch in range(1, args.epochs + 1):
    start = time.perf_counter()
    train = run(train_loader, True)
    train_seconds = time.perf_counter() - start
    start = time.perf_counter()
    val = run(val_loader, False)
    val_seconds = time.perf_counter() - start
    start = time.perf_counter()
    test = run(test_loader, False)
    test_seconds = time.perf_counter() - start
    dataset.reset_label_time()
    print(
        f"Epoch: {epoch:02d}, Loss: {train['ce']:.4f}, "
        f"Train NDCG: {train['ndcg']:.4f}, Val NDCG: {val['ndcg']:.4f}, "
        f"Test NDCG: {test['ndcg']:.4f} (train {train_seconds:.1f} s, "
        f"val {val_seconds:.1f} s, test {test_seconds:.1f} s)",
        file=sys.stderr,
        flush=True,
    )
    history.append(dict(
        epoch=epoch, loss=train["ce"], train_ndcg=train["ndcg"],
        val_ndcg=val["ndcg"], test_ndcg=test["ndcg"],
        train_seconds=train_seconds, val_seconds=val_seconds,
        test_seconds=test_seconds,
    ))  # fmt: skip

best = max(history, key=lambda h: h["val_ndcg"])
print(f"Best epoch: {best['epoch']:02d}, Val NDCG: {best['val_ndcg']:.4f}, "
      f"Test NDCG: {best['test_ndcg']:.4f}", file=sys.stderr)  # fmt: skip
if args.json:
    print(json.dumps(dict(history=history, best_epoch=best["epoch"])))
