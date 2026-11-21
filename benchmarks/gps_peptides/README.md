# GPS on Peptides-func: mlx-graphs vs PyG

Accuracy sanity check of `GPSConv` (`examples/graph_gps_peptides.py`) against
a PyG twin of the same model (`gps_pyg.py`), with the same data, architecture
and hyperparameters.

- `export_npz.py <data_dir>`: exports the processed splits (with RWSE,
  computed once by `AddRandomWalkPE(20)`) to npz, so both frameworks train on
  identical data and the PyG run does not import MLX.
- `gps_pyg.py --data_dir <data_dir>`: PyG 2.8.1 twin (CPU, 2 threads).
- `grad_parity.py <data_dir>`: copies the PyG weights into the MLX model and
  compares the train-mode loss and gradients on 16 graphs.

Setup: 4 GPS layers (GINE + 4-head attention, attention dropout 0.5, batch
norm), 64 channels (8 for RWSE), mean pooling, Adam (lr 1e-3, L2 1e-5),
batch size 32, 10 epochs, seeds 0 and 1. Metric: average precision over the
10 tasks; "test" is taken at the best validation epoch. Each run was under a
4 GB memory watchdog.

| run | final train loss | best val AP | test AP | epoch time |
|---|---|---|---|---|
| PyG, seed 0 | 0.2522 | 0.4044 | 0.3881 | ~110 s (CPU) |
| PyG, seed 1 | 0.2613 | 0.4194 | 0.4079 | ~100 s (CPU) |
| mlx-graphs, seed 0 | 0.2588 | 0.4447 | 0.4288 | ~30 s (GPU) |
| mlx-graphs, seed 1 | 0.2553 | 0.4437 | 0.4268 | ~30 s (GPU) |

The full-model gradient check gives the same loss (0.70363 vs 0.70362) and
gradients within ~1% (MLX GPU matmul precision).

Note: `mlx.optimizers.Adam` has no bias correction by default, unlike
`torch.optim.Adam`. Without it (first runs), mlx-graphs reached best val AP
0.389 / 0.415 but a clearly higher training loss (0.276 / 0.274), so the
example uses `bias_correction=True`.
