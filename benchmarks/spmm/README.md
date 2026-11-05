# Fused (SpMM) message passing benchmark

Compares mlx-graphs' gather-scatter message passing with the fused
sparse x dense matmul kernels from mlx-cluster, on the same layers and
parameters.

| mode     | what runs |
|----------|-----------|
| `gather` | today's path: `x[src]` gathered per edge, then scattered to `dst` |
| `coo`    | `mlx_cluster.spmm_coo` on `edge_index` as-is (atomic adds) |
| `csr`    | `mlx_cluster.spmm_csr` on a CSR copy, built once per graph and cached (no atomics) |

Layers: `aggr-add` / `aggr-mean` (`SimpleConv`, aggregation only, the
cleanest view of the kernels), `gcn` (`GCNConv`) and `sage` (`SAGEConv`).

## 1. Setup (on an Apple Silicon Mac)

```bash
python -m venv .venv && source .venv/bin/activate

# mlx-cluster with the SpMM kernels (spmm-kernel branch)
cd /path/to/mlx_cluster
pip install -e . --verbose

# mlx-graphs from this branch. --no-deps keeps mlx >= 0.32 from mlx-cluster:
# mlx-graphs' pyproject still pins mlx < 0.31.
cd /path/to/mlx-graphs
pip install -e . --no-deps
pip install numpy requests fsspec tqdm snakeviz

python -c "import mlx_graphs.nn as n; print('fused kernels:', n.is_fused_message_passing_enabled())"
```

The last line must print `fused kernels: True`.

## 2. Run the benchmark

```bash
cd benchmarks/spmm
python bench_fused_mp.py
```

Default sweep: Cora, PubMed, 1M-edge uniform and power-law graphs;
`D = 16, 64, 256`; layers `aggr-add gcn sage`; all three modes. Planetoid
datasets download on first use.

Each row prints as it finishes and everything goes to `results.csv`.

| column | meaning |
|---|---|
| `fwd ms` | median forward time |
| `f+b ms` | median time of `mx.grad` w.r.t. the node features (backward through the layer, including the SpMM/SDDMM VJP) |
| `fwd MB` / `f+b MB` | peak memory allocated by that step, beyond what was already live |
| `speedup` | gather `f+b ms` / this mode's `f+b ms` |
| `mem x` | gather `f+b MB` / this mode's `f+b MB` |
| `err` | max absolute difference from the gather output (should be ~1e-5 or less) |
| `OOM` | the run ran out of memory |

`results.csv` also has `csr_build_ms` (the one-time COO -> CSR sort) and the
mlx / mlx-graphs / mlx-cluster versions and chip.

Useful variations:

```bash
# large graphs: gather-scatter should run out of memory first
python bench_fused_mp.py --graphs uniform-10m powerlaw-10m --dims 64 256

# real medium graph (downloads ogbn-arxiv, ~100 MB)
python bench_fused_mp.py --graphs arxiv

# just the kernels, more repeats for stable numbers
python bench_fused_mp.py --layers aggr-add aggr-mean --repeats 30

# everything
python bench_fused_mp.py --help
```

Close other GPU-heavy apps while benchmarking; run on power, not battery.

## 3. Profile

### a) GPU: Metal trace in Xcode (time per kernel)

This records exactly what ran on the GPU for one configuration, so you can
see how long `spmm_coo`, `spmm_csr`, `sddmm`, MLX's gather/scatter, the
matmuls etc. each take.

```bash
MTL_CAPTURE_ENABLED=1 python bench_fused_mp.py \
    --graphs powerlaw-1m --dims 64 --layers gcn --modes coo \
    --capture gcn_coo.gputrace
open gcn_coo.gputrace          # opens in Xcode (Xcode must be installed)
```

`--capture` traces the **first** value of `--graphs`, `--dims`, `--layers`
and `--modes`, for `--capture-iters` (default 2) forward+backward steps, after
a warm-up. Record one trace per mode to compare them, e.g. also
`--modes gather --capture gcn_gather.gputrace`.

In Xcode:

1. Click **Replay** (top of the window) if Xcode asks. The trace is replayed
   on your GPU to collect timings.
2. In the left **Debug Navigator**, switch the view to **"View Frame By Call"**
   (or "By Pipeline State"). You get one entry per kernel dispatch. Our
   kernels show up under their names: `spmm_coo`, `spmm_coo_val`,
   `spmm_csr`, `spmm_csr_val`, `spmm_zero`, `sddmm`.
3. Open the **Performance** / **Counters** view (the "Profile" button or the
   "Performance" item in the navigator) to see GPU time per dispatch, plus
   counters like memory bandwidth and ALU utilisation.
4. Compare with the `gather` trace: there you'll see MLX's `gather` and
   `scatter` kernels instead, which move the large per-edge message arrays.

Notes: the file must end in `.gputrace`, and Metal won't overwrite an
existing trace (the script stops if it exists). Traces can be large; keep
`--capture-iters` small.

### b) GPU: whole-run timeline with Instruments

For a timeline of the whole run (GPU busy/idle, CPU-GPU sync, memory),
without Xcode's per-dispatch replay:

```bash
xcrun xctrace record --template 'Metal System Trace' \
    --output fused.trace --launch -- \
    python bench_fused_mp.py --graphs powerlaw-1m --dims 64 --layers gcn --modes coo
open fused.trace
```

### c) CPU / Python side: cProfile + snakeviz

Shows Python overhead: graph building, the COO -> CSR conversion, the
decorator's checks, MLX graph construction.

```bash
python bench_fused_mp.py --graphs cora pubmed --cprofile fused.prof
snakeviz fused.prof            # opens an interactive view in the browser
```

The script also prints the 15 most expensive functions at the end.

## 4. Correctness

Before trusting numbers, run the tests (both use the GPU when available):

```bash
cd /path/to/mlx_cluster && pytest tests/test_spmm.py -v
cd /path/to/mlx-graphs  && pytest tests/nn/test_fused_message_passing.py -v
```
