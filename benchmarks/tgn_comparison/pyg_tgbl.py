"""py-tgb's TGN example for dynamic link property prediction
(``examples/linkproppred/tgbl-wiki/tgn.py`` of github.com/shenyangHuang/TGB),
with PyG on the CPU and --epochs/--seed/--json, for the comparison with
``examples/tgn_tgbl.py``.

The TGB example imports its modules from the TGB repository (``modules/``),
which are copies of PyG's TGN classes: they are taken from PyG here, with
``scatter_argmax`` giving empty groups an out-of-range index as torch-scatter
does (TGB's ``LastAggregator`` uses torch-scatter), and TGB's
``LinkPredictor`` (with a final sigmoid) and ``EarlyStopMonitor`` are copied.

The dataset, negative samples and evaluator are py-tgb's. ``--data`` is a
directory with mlx-graphs' download of the dataset (``tgbl_wiki/raw``), which
is linked for py-tgb.

``--eval per_edge`` (default) scores the positive edges one at a time as the
TGB example; ``--eval batched`` scores the edges of a mini-batch together,
which gives the same scores (see ``examples/tgn_tgbl.py``).
"""

import json
import os
import sys
import time
from argparse import ArgumentParser

import numpy as np
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
parser.add_argument("--dataset", default="tgbl-wiki")
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--threads", type=int, default=4)
parser.add_argument("--eval", choices=["per_edge", "batched"], default="per_edge")
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
    """py-tgb reads ``<its package dir>/<root>/<name>``: link our raw files
    there and return ``root``."""
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


from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset  # noqa: E402
from tgb.linkproppred.evaluate import Evaluator  # noqa: E402
from tgb.utils.utils import set_random_seed  # noqa: E402

torch.set_num_threads(args.threads)
device = torch.device("cpu")
torch.manual_seed(args.seed)
set_random_seed(args.seed)

LR, BATCH_SIZE, MEM_DIM, TIME_DIM, EMB_DIM = 1e-4, 200, 100, 100, 100
TOLERANCE, PATIENCE, NUM_NEIGHBORS = 1e-6, 5, 10

dataset = PyGLinkPropPredDataset(
    name=args.dataset, root=py_tgb_root(args.data, args.dataset)
)
data = dataset.get_TemporalData().to(device)
metric = dataset.eval_metric
train_data = data[dataset.train_mask]
val_data = data[dataset.val_mask]
test_data = data[dataset.test_mask]
train_loader = TemporalDataLoader(train_data, batch_size=BATCH_SIZE)
val_loader = TemporalDataLoader(val_data, batch_size=BATCH_SIZE)
test_loader = TemporalDataLoader(test_data, batch_size=BATCH_SIZE)
min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())
evaluator = Evaluator(name=args.dataset)
neg_sampler = dataset.negative_sampler


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


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lin_src = Linear(in_channels, in_channels)
        self.lin_dst = Linear(in_channels, in_channels)
        self.lin_final = Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        return self.lin_final(h).sigmoid()


class EarlyStopMonitor:
    def __init__(self, tolerance, patience):
        self.tolerance, self.patience = tolerance, patience
        self.best, self.best_epoch, self.counter, self.epoch = None, 0, 0, 1
        self.state = None

    def step_check(self, curr, models):
        if self.best is None or (curr - self.best) / np.abs(self.best) > self.tolerance:
            self.best, self.best_epoch, self.counter = curr, self.epoch, 0
            self.state = {
                k: {n: v.clone() for n, v in m.state_dict().items()}
                for k, m in models.items()
            }
        else:
            self.counter += 1
        self.epoch += 1
        return self.counter >= self.patience

    def load_checkpoint(self, models):
        for k, m in models.items():
            m.load_state_dict(self.state[k])


def embed(n_id):
    n_id, edge_index, e_id = neighbor_loader(n_id)
    assoc[n_id] = torch.arange(n_id.size(0), device=device)
    z, last_update = model["memory"](n_id)
    return model["gnn"](z, last_update, edge_index, data.t[e_id], data.msg[e_id])


def train():
    for m in model.values():
        m.train()
    model["memory"].reset_state()
    neighbor_loader.reset_state()

    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
        neg_dst = torch.randint(
            min_dst_idx, max_dst_idx + 1, (src.size(0),), dtype=torch.long
        )
        z = embed(torch.cat([src, pos_dst, neg_dst]).unique())
        pos_out = model["link_pred"](z[assoc[src]], z[assoc[pos_dst]])
        neg_out = model["link_pred"](z[assoc[src]], z[assoc[neg_dst]])
        loss = criterion(pos_out, torch.ones_like(pos_out))
        loss += criterion(neg_out, torch.zeros_like(neg_out))

        model["memory"].update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)
        loss.backward()
        optimizer.step()
        model["memory"].detach()
        total_loss += float(loss.detach()) * batch.num_events
    return total_loss / train_data.num_events


@torch.no_grad()
def test(loader, split_mode):
    for m in model.values():
        m.eval()
    perf_list = []
    for pos_batch in loader:
        pos_src, pos_dst, pos_t, pos_msg = (
            pos_batch.src, pos_batch.dst, pos_batch.t, pos_batch.msg,
        )  # fmt: skip
        neg_batch_list = neg_sampler.query_batch(
            pos_src, pos_dst, pos_t, split_mode=split_mode
        )
        if args.eval == "per_edge":  # the TGB example
            for idx, neg_batch in enumerate(neg_batch_list):
                src = torch.full((1 + len(neg_batch),), pos_src[idx], device=device)
                dst = torch.tensor(
                    np.concatenate(
                        ([np.array([pos_dst.cpu().numpy()[idx]]), np.array(neg_batch)]),
                        axis=0,
                    ),
                    device=device,
                )
                z = embed(torch.cat([src, dst]).unique())
                y_pred = model["link_pred"](z[assoc[src]], z[assoc[dst]])
                input_dict = {
                    "y_pred_pos": np.array([y_pred[0, :].squeeze(dim=-1).cpu()]),
                    "y_pred_neg": np.array(y_pred[1:, :].squeeze(dim=-1).cpu()),
                    "eval_metric": [metric],
                }
                perf_list.append(evaluator.eval(input_dict)[metric])
        else:
            width = max(len(n) for n in neg_batch_list)
            neg = torch.tensor(
                [n + [-1] * (width - len(n)) for n in neg_batch_list], dtype=torch.long
            )
            valid = neg >= 0
            dst = torch.cat(
                [pos_dst[:, None], torch.where(valid, neg, pos_dst[:, None])], 1
            )
            z = embed(torch.cat([pos_src, dst.view(-1)]).unique())
            y_pred = model["link_pred"](
                z[assoc[pos_src]][:, None, :].expand(-1, dst.size(1), -1),
                z[assoc[dst]],
            ).squeeze(-1)
            y_neg = torch.where(valid, y_pred[:, 1:], -torch.inf).numpy()
            for i in range(y_pred.size(0)):
                input_dict = {
                    "y_pred_pos": y_pred[i, :1].numpy(),
                    "y_pred_neg": y_neg[i],
                    "eval_metric": [metric],
                }
                perf_list.append(evaluator.eval(input_dict)[metric])

        model["memory"].update_state(pos_src, pos_dst, pos_t, pos_msg)
        neighbor_loader.insert(pos_src, pos_dst)
    return float(torch.tensor(perf_list).mean())


neighbor_loader = LastNeighborLoader(data.num_nodes, size=NUM_NEIGHBORS, device=device)
memory = TGNMemory(
    data.num_nodes, data.msg.size(-1), MEM_DIM, TIME_DIM,
    message_module=IdentityMessage(data.msg.size(-1), MEM_DIM, TIME_DIM),
    aggregator_module=LastAggregator(),
).to(device)  # fmt: skip
gnn = GraphAttentionEmbedding(
    in_channels=MEM_DIM, out_channels=EMB_DIM, msg_dim=data.msg.size(-1),
    time_enc=memory.time_enc,
).to(device)  # fmt: skip
link_pred = LinkPredictor(in_channels=EMB_DIM).to(device)
model = {"memory": memory, "gnn": gnn, "link_pred": link_pred}
optimizer = torch.optim.Adam(
    set(memory.parameters()) | set(gnn.parameters()) | set(link_pred.parameters()),
    lr=LR,
)
criterion = torch.nn.BCEWithLogitsLoss()
assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)
early_stopper = EarlyStopMonitor(TOLERANCE, PATIENCE)

dataset.load_val_ns()
history = []
for epoch in range(1, args.epochs + 1):
    start = time.perf_counter()
    loss = train()
    train_seconds = time.perf_counter() - start
    start = time.perf_counter()
    val_mrr = test(val_loader, split_mode="val")
    val_seconds = time.perf_counter() - start
    print(
        f"Epoch: {epoch:02d}, Loss: {loss:.4f}, Val MRR: {val_mrr:.4f} "
        f"(train {train_seconds:.1f} s, val {val_seconds:.1f} s)",
        file=sys.stderr,
        flush=True,
    )
    history.append(
        dict(epoch=epoch, loss=loss, val_mrr=val_mrr, train_seconds=train_seconds,
             val_seconds=val_seconds)
    )  # fmt: skip
    if early_stopper.step_check(val_mrr, model):
        break

early_stopper.load_checkpoint(model)
dataset.load_test_ns()
start = time.perf_counter()
test_mrr = test(test_loader, split_mode="test")
test_seconds = time.perf_counter() - start
print(f"Best epoch: {early_stopper.best_epoch:02d}, Test MRR: {test_mrr:.4f} "
      f"({test_seconds:.1f} s)", file=sys.stderr)  # fmt: skip
if args.json:
    print(json.dumps(dict(history=history, best_epoch=early_stopper.best_epoch,
                          test_mrr=test_mrr, test_seconds=test_seconds)))  # fmt: skip
