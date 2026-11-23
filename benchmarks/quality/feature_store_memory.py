"""Does NeighborLoader over a MemmapFeatureStore keep memory bounded when the
node features are larger than what it reads per mini-batch?

    python benchmarks/quality/feature_store_memory.py --dir /tmp/fs_bench \\
        --num-nodes 1500000 --num-features 1024     # 6.1 GB of features

1. Writes a synthetic graph to disk: random edges (``--degree`` in-edges per
   node, in a :class:`MemmapGraphStore`) and ``num_nodes x num_features``
   float32 features in a :class:`MemmapFeatureStore`, filled in chunks
   (``x[i, j] = ((31 * i + j) % 1000) / 1000``, so any row can be checked).
2. In a fresh subprocess, iterates a :class:`NeighborLoader` over the
   ``(feature_store, graph_store)`` pair (``--batches`` mini-batches of
   ``--batch-size`` seeds, fan-out ``[10, 5]``), checks every gathered row,
   (spot-checks 256 rows per batch), and reports the peak resident memory
   (``ru_maxrss``) and the physical footprint (``top``, sampled every 10
   batches), compared with the size of the features file. MLX's buffer
   cache is capped (``--cache-gb``), since by default it keeps freed
   buffers up to the memory limit.
3. Deletes the files (unless ``--keep``).
"""

import argparse
import os
import resource
import shutil
import subprocess
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)


def expected_rows(index: np.ndarray, num_features: int) -> np.ndarray:
    return (((31 * index[:, None] + np.arange(num_features)) % 1000) / 1000).astype(
        np.float32
    )


def write(root: str, num_nodes: int, num_features: int, degree: int, seed: int):
    from mlx_graphs.data import MemmapFeatureStore, MemmapGraphStore

    rng = np.random.default_rng(seed)
    fs = MemmapFeatureStore(os.path.join(root, "features"))
    x = fs.create_tensor("x", (num_nodes, num_features), "float32")
    chunk = max(1, (1 << 28) // (4 * num_features))  # ~256 MB per chunk
    for start in range(0, num_nodes, chunk):
        end = min(start + chunk, num_nodes)
        x[start:end] = expected_rows(np.arange(start, end), num_features)
    x.flush()
    del x
    fs["y"] = rng.integers(0, 10, num_nodes).astype(np.int32)

    # Every node gets `degree` random in-neighbors: CSC built directly.
    gs = MemmapGraphStore(os.path.join(root, "graph"))
    colptr = np.arange(num_nodes + 1, dtype=np.int64) * degree
    row = rng.integers(0, num_nodes, num_nodes * degree, dtype=np.int64)
    gs.put_edge_index((row, colptr), layout="csc", size=(num_nodes, num_nodes))


def footprint_gb(pid: int) -> float:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from run_capped import footprint_gb as fp

    return fp(pid)


def iterate(root: str, batches: int, batch_size: int, seed: int, cache_gb: float):
    """Runs in a fresh process: sample and gather, then report memory."""
    import mlx.core as mx

    from mlx_graphs.data import MemmapFeatureStore, MemmapGraphStore
    from mlx_graphs.loaders import NeighborLoader

    np.random.seed(seed)
    # MLX keeps freed buffers in a cache (up to the memory limit by default),
    # which would hide the footprint of the loader itself.
    mx.set_cache_limit(int(cache_gb * 1e9))
    fs = MemmapFeatureStore(os.path.join(root, "features"))
    gs = MemmapGraphStore(os.path.join(root, "graph"))
    num_nodes, num_features = fs.get_tensor_size(attr_name="x")
    seeds = np.random.permutation(num_nodes)[: batches * batch_size]
    loader = NeighborLoader(
        (fs, gs),
        num_neighbors=[10, 5],
        input_nodes=mx.array(seeds),
        batch_size=batch_size,
    )
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.perf_counter()
    rows = 0
    max_fp = 0.0
    check = np.random.default_rng(seed)
    for i, batch in enumerate(loader):
        n_id = np.asarray(batch.n_id)
        x = np.asarray(batch.node_features)
        some = check.integers(0, n_id.size, 256)  # spot-check rows
        assert np.array_equal(x[some], expected_rows(n_id[some], num_features))
        rows += n_id.size
        del batch, x
        if i % 10 == 0:
            max_fp = max(max_fp, footprint_gb(os.getpid()))
    elapsed = time.perf_counter() - t0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1 if sys.platform == "darwin" else 1024  # bytes on macOS, KB on Linux
    print(
        f"RESULT batches={len(loader)} rows={rows} "
        f"gathered_gb={rows * num_features * 4 / 1e9:.2f} "
        f"rss_before_gb={rss0 * scale / 1e9:.2f} peak_rss_gb={peak * scale / 1e9:.2f} "
        f"max_footprint_gb={max_fp:.2f} mlx_peak_gb={mx.get_peak_memory() / 1e9:.2f} "
        f"seconds={elapsed:.1f} "
        f"batch_ms={1000 * elapsed / len(loader):.1f}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--num-nodes", type=int, default=1_500_000)
    parser.add_argument("--num-features", type=int, default=1024)
    parser.add_argument("--degree", type=int, default=10)
    parser.add_argument("--batches", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--cache-gb", type=float, default=1.0, help="MLX cache limit")
    parser.add_argument("--skip-write", action="store_true", help="reuse --dir")
    parser.add_argument("--iterate-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.iterate_only:
        iterate(args.dir, args.batches, args.batch_size, args.seed, args.cache_gb)
        return

    size_gb = args.num_nodes * args.num_features * 4 / 1e9
    try:
        t0 = time.perf_counter()
        if not args.skip_write:
            write(args.dir, args.num_nodes, args.num_features, args.degree, args.seed)
        print(
            f"wrote {size_gb:.2f} GB of features ({args.num_nodes:,} x "
            f"{args.num_features}) and {args.num_nodes * args.degree:,} edges in "
            f"{time.perf_counter() - t0:.0f} s",
            flush=True,
        )
        out = subprocess.run(
            [
                sys.executable,
                __file__,
                "--dir",
                args.dir,
                "--batches",
                str(args.batches),
                "--batch-size",
                str(args.batch_size),
                "--seed",
                str(args.seed),
                "--cache-gb",
                str(args.cache_gb),
                "--iterate-only",
            ],
            capture_output=True,
            text=True,
        )
        print(out.stdout.strip())
        if out.returncode != 0:
            print(out.stderr[-3000:])
    finally:
        if not args.keep:
            shutil.rmtree(args.dir, ignore_errors=True)


if __name__ == "__main__":
    main()
