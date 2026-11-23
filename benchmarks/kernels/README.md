# Kernel benchmarks

## GATv2 edge scores

On an Apple Silicon Mac, run the 100k-edge layer comparison from the repo root.

```bash
uv run --python 3.13 python -m benchmarks.kernels.gatv2_scores
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
