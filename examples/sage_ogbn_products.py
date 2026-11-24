"""GraphSAGE on the official ogbn-products split using on-disk CSC and features.

    curl -fL -o products.zip https://snap.stanford.edu/ogb/data/nodeproppred/products.zip
    python examples/sage_ogbn_products.py prepare products.zip .mlx_graphs_data/products
    python examples/sage_ogbn_products.py train .mlx_graphs_data/products

Preparation requires pandas. Run each command under ``/usr/bin/time -l`` on
macOS to measure its peak physical footprint. Evaluation uses sampled neighbors
on the complete official validation and test splits.
"""

import argparse
import json
import tempfile
import time
from pathlib import Path
from zipfile import ZipFile

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from mlx_graphs.data import MemmapFeatureStore, MemmapGraphStore
from mlx_graphs.loaders import NeighborLoader
from mlx_graphs.nn import GraphSAGE


def csv_chunks(archive, name, dtype, chunk_size):
    import pandas as pd

    with archive.open(f"products/{name}.csv.gz") as file:
        with pd.read_csv(
            file, compression="gzip", dtype=dtype, header=None, chunksize=chunk_size
        ) as reader:
            for frame in reader:
                yield frame.to_numpy()


def write_features(archive, store, name, field, num_nodes, dtype):
    start = 0
    for chunk in csv_chunks(archive, f"raw/{name}", dtype, 32_768):
        if start == 0:
            shape = (num_nodes,) if field == "y" else (num_nodes, chunk.shape[1])
            array = store.create_tensor(field, shape, dtype)
        end = start + len(chunk)
        array[start:end] = chunk.reshape(-1) if field == "y" else chunk
        start = end
    if start != num_nodes:
        raise ValueError(f"Expected {num_nodes} rows in {name}, found {start}")
    array.flush()


def write_graph(archive, root, num_nodes):
    def edges():
        return csv_chunks(archive, "raw/edge", np.int64, 500_000)

    # Count in-edges, including the inverse of every raw edge as OGB does.
    colptr = np.zeros(num_nodes + 1, dtype=np.int64)
    for chunk in edges():
        np.add.at(colptr[1:], chunk.ravel(), 1)
    np.cumsum(colptr, out=colptr)

    scratch = root / "rows.npy"
    rows = np.lib.format.open_memmap(
        scratch, mode="w+", dtype=np.int64, shape=(int(colptr[-1]),)
    )
    cursor = colptr[:-1].copy()
    for chunk in edges():
        # Stable destination sort preserves the order of the directed edges
        # and their interleaved inverses in the official OGB loader.
        src = chunk.ravel()
        dst = chunk[:, ::-1].ravel()
        order = np.argsort(dst, kind="stable")
        nodes, starts, counts = np.unique(
            dst[order], return_index=True, return_counts=True
        )
        positions = np.arange(len(src)) + np.repeat(cursor[nodes] - starts, counts)
        rows[positions] = src[order]
        cursor[nodes] += counts
    rows.flush()

    graph = MemmapGraphStore(str(root / "graph"))
    graph.put_edge_index((rows, colptr), layout="csc", size=(num_nodes, num_nodes))
    del rows
    scratch.unlink()
    return int(colptr[-1])


def prepare(archive_path, output):
    start = time.perf_counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Dataset directory already exists: {output}")
    with (
        ZipFile(archive_path) as archive,
        tempfile.TemporaryDirectory(dir=output.parent) as staging,
    ):
        root = Path(staging) / "dataset"
        num_nodes = int(
            next(csv_chunks(archive, "raw/num-node-list", np.int64, 1)).item()
        )
        store = MemmapFeatureStore(str(root / "features"))
        write_features(archive, store, "node-feat", "x", num_nodes, np.float32)
        write_features(archive, store, "node-label", "y", num_nodes, np.int32)
        num_edges = write_graph(archive, root, num_nodes)
        for split in ("train", "valid", "test"):
            chunks = csv_chunks(
                archive, f"split/sales_ranking/{split}", np.int32, 500_000
            )
            np.save(root / f"{split}.npy", np.concatenate(list(chunks)).reshape(-1))
        root.rename(output)
    print(json.dumps({"prepare_s": time.perf_counter() - start, "edges": num_edges}))


def accuracy(model, loader):
    model.eval()
    correct = total = 0
    for batch in loader:
        n = batch.batch_size
        logits = model(batch.edge_index, batch.node_features)[:n]
        correct += (logits.argmax(axis=-1) == batch.node_labels[:n]).sum().item()
        total += n
    return correct / total


def train(args):
    mx.set_cache_limit(512 << 20)
    mx.random.seed(args.seed)
    np.random.seed(args.seed)
    features = MemmapFeatureStore(str(args.data / "features"))
    graph = MemmapGraphStore(str(args.data / "graph"))
    train_ids = np.load(args.data / "train.npy")
    valid_ids = np.load(args.data / "valid.npy")
    test_ids = np.load(args.data / "test.npy")
    start = time.perf_counter()
    allocated = mx.get_active_memory()
    train_loader = NeighborLoader(
        (features, graph),
        args.fanouts,
        input_nodes=train_ids,
        batch_size=args.batch_size,
        shuffle=True,
    )
    print(
        json.dumps(
            {
                "loader_init_s": time.perf_counter() - start,
                "loader_mlx_bytes": mx.get_active_memory() - allocated,
                "edges": graph.num_edges(),
            }
        ),
        flush=True,
    )

    def eval_loader(ids):
        return NeighborLoader(
            (features, graph),
            args.fanouts,
            input_nodes=ids,
            batch_size=args.eval_batch_size,
        )

    feature_dim = features.get_tensor_size(attr_name="x")[1]
    model = GraphSAGE(feature_dim, args.hidden, 2, out_features_dim=47, dropout=0.5)
    opt = optim.Adam(learning_rate=0.003, bias_correction=True)

    def loss_fn(model, batch):
        n = batch.batch_size
        logits = model(batch.edge_index, batch.node_features)[:n]
        return nn.losses.cross_entropy(logits, batch.node_labels[:n], reduction="mean")

    step = nn.value_and_grad(model, loss_fn)
    valid_loader = eval_loader(valid_ids)
    best_accuracy = -1
    best_parameters = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.perf_counter()
        loss_sum = 0.0
        for batch in train_loader:
            loss, grads = step(model, batch)
            opt.update(model, grads)
            mx.eval(loss, model.parameters(), opt.state)
            loss_sum += loss.item() * batch.batch_size
        train_s = time.perf_counter() - start
        start = time.perf_counter()
        val_accuracy = accuracy(model, valid_loader)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_parameters = model.parameters()
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_s": train_s,
                    "train_loss": loss_sum / len(train_ids),
                    "validation_s": time.perf_counter() - start,
                    "val_accuracy": val_accuracy,
                }
            ),
            flush=True,
        )

    model.update(best_parameters)
    start = time.perf_counter()
    test_accuracy = accuracy(model, eval_loader(test_ids))
    print(
        json.dumps(
            {
                "test_accuracy": test_accuracy,
                "test_s": time.perf_counter() - start,
                "mlx_peak_bytes": mx.get_peak_memory(),
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("archive", type=Path)
    prep.add_argument("data", type=Path)
    run = commands.add_parser("train")
    run.add_argument("data", type=Path)
    run.add_argument("--epochs", type=int, default=3)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--hidden", type=int, default=128)
    run.add_argument("--fanouts", type=int, nargs=2, default=[10, 5])
    run.add_argument("--batch-size", type=int, default=1024)
    run.add_argument("--eval-batch-size", type=int, default=4096)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.archive, args.data)
    else:
        if args.epochs < 1:
            parser.error("--epochs must be positive")
        train(args)


if __name__ == "__main__":
    main()
