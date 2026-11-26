# Kernel benchmarks

## CSR softmax SpMM

`csr_softmax_spmm` normalizes edge logits within each destination row and
applies them to source values. It accepts logits from any scoring function.
GAT and GATv2 can opt in with
`fused_message_passing_mode(True, format="csr", softmax_spmm=True)`.
Active attention dropout and unsupported inputs use the existing path.

The benchmark compares the two paths through complete GAT and GATv2 layers.
Both runs use the same initialized parameters and prepare CSR during warm-up.

```bash
uv run --python 3.13 python -m benchmarks.kernels.csr_softmax_spmm --large
```

## GATv2 edge scores

On an Apple Silicon Mac, run the 100k-edge layer comparison from the repo root.

```bash
uv run --python 3.13 python -m benchmarks.kernels.gatv2_scores
```

It compares three score paths through complete GATv2 layers: the gather
reference, the in-tree Metal kernel (whose backward pass adds into the
gradients with atomics) and `mlx_cluster.gatv2_scores` (whose backward pass
sums each node's edges on the CSR, without atomics, with chunking for hub
nodes). The mlx-cluster path needs an mlx-cluster with `gatv2_scores` and is
skipped otherwise; `--paths reference metal` skips it too. `--paths` also runs
the paths in the order given, so rotating it across runs evens out order
effects. Add `auto` to `--paths` to time GATv2Conv's default choice (mlx-cluster
for graphs with 1M+ edges, the in-tree kernel otherwise); its summary line says
which kernel it chose.

`--datasets` runs on real graphs instead of Coauthor Physics: node
classification on each, and with the layer task, a layer on each graph's edges
(random 128-dimensional features, so the dense layers cost the same on every
graph and the score paths make the difference). For example:

```bash
python -m benchmarks.kernels.gatv2_scores --tasks layer node --datasets ogbn-arxiv amazon-computers
```

The full comparison adds a 2M-edge layer run, Coauthor Physics node
classification, and DD graph classification.

```bash
uv run --python 3.13 --extra test python -m benchmarks.kernels.gatv2_scores \
  --large --tasks layer node graph --out .cache/gatv2-results.json
```

Each pair uses the same initialization seed and fused SpMM aggregation.
Training pairs share data splits and batch order. Each run has its own process
and excludes warm-up from timing. The script checks that the requested score
path ran.

The layer task reports forward and forward-plus-backward time. The training
tasks report time per epoch and test accuracy at the best validation epoch
for seeds 0, 1, and 2. Both report peak MLX memory. The JSON includes per-seed
metrics and system information. Training datasets download on first use.
Use `--seeds 0` for a shorter training comparison.
