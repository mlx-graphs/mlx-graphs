# mlx-graphs vs PyTorch Geometric on Apple Silicon

Compares mlx-graphs (MLX, GPU) with PyTorch Geometric on the same Mac, on
both of PyTorch's backends: MPS (the Apple GPU) and CPU.

```bash
pip install -e '.[benchmarks]'   # or: pip install torch torch_geometric ogb matplotlib
python -m benchmarks.pyg_comparison.run      # ~1.5 hours -> results/<chip>/<version>/
python benchmarks/pyg_comparison/plot.py      # figures/, results.md and the docs page of that version
python benchmarks/pyg_comparison/history.py   # results/<chip>/history.md: all versions
```

`run.py --quick` runs the smallest configuration of every task as a smoke
test. Datasets download into `.mlx_graphs_data/` on first use.

## Tracking performance across versions

Every run is stored under the version of mlx-graphs it measured:
`results/<chip>/<version>-<commit>/`, where the commit is the last one that
changed `mlx_graphs/` (see `versions.py`; `--label` overrides the name).
Each version directory has its own `results.md` and figures, and
`history.py` compares all the versions measured on a chip in
`results/<chip>/history.md`: mlx-graphs' time, memory and accuracy in each
version, and the speedup of the newest over the oldest.

The harness is independent of the library version: it loads datasets through
PyG / OGB and only uses layers that exist in every version, and workers import
`mlx_graphs` from the checkout `run.py` is in. To measure another branch or
release, run the harness from a worktree of it:

```bash
git worktree add ../mlx-graphs-v0.0.11 v0.0.11
cp benchmarks/__init__.py benchmarks/common.py benchmarks/versions.py \
    benchmarks/workloads.py ../mlx-graphs-v0.0.11/benchmarks/
cp -r benchmarks/pyg_comparison ../mlx-graphs-v0.0.11/benchmarks/
output="$PWD/benchmarks/pyg_comparison/results/CHIP/LABEL"
(cd ../mlx-graphs-v0.0.11 && python -m benchmarks.pyg_comparison.run --out "$output")
python benchmarks/pyg_comparison/history.py
```

Run one benchmark at a time: concurrent runs share the GPU and slow each
other down.

## What is measured

| task | setup | reported |
|---|---|---|
| `layer` | one convolution layer (GCN, GraphSAGE, GIN, GAT, GATv2) on uniform random graphs of 100k, 500k and 2M edges; 128 features in and out, 4 heads for the attention layers | median time of a forward pass and of a forward + backward pass; peak memory |
| `node` | full-batch node classification, 2-layer GCN / GraphSAGE / GAT, 64 hidden units, dropout 0.5, Adam (lr 0.01), 200 epochs (100 on ogbn-arxiv), on Cora, CiteSeer, PubMed, Amazon Photo / Computers, Coauthor CS / Physics and ogbn-arxiv; 3 seeds | median time per training step; training loss at every epoch; train / validation / test accuracy every 10 epochs; test accuracy at the best validation accuracy |
| `graph` | graph classification on TU datasets (MUTAG, PROTEINS, NCI1, DD), 3 layers + mean pooling, batches of 64 graphs, 100 epochs, random 80/10/10 train / validation / test split; 3 seeds | median time per epoch, *including loading and batching the graphs*; mean training loss of every epoch; accuracies every 5 epochs; test accuracy at the best validation accuracy |

## Do the models train the same?

Speed only matters if the models learn what they should, so the node and
graph tasks also record how training goes, in every framework:

- the training loss at every epoch, plotted for the three frameworks on the
  same axes (`node_loss` / `graph_loss` figures);
- the test accuracy reached, selected at the epoch with the best validation
  accuracy, as mean and range over 3 seeds (`node_accuracy` /
  `graph_accuracy` figures), and the final training loss (mean of the last
  10 epochs), in the "training quality" tables of `results.md`.

Losses and accuracies are read *after* each timed step has finished, so
measuring them does not change the timings. `--seeds` sets the seeds
(default `0 1 2`); times are medians over seeds.

## Making the comparison fair

- Both frameworks train the same architecture with the same hyperparameters
  on the same data (node classification loads the graphs through mlx-graphs
  and hands identical arrays to PyG).
- MLX's `Adam` is created with `bias_correction=True`, which is the update
  rule of `torch.optim.Adam`. MLX's default (no bias correction) takes much
  larger first steps and lowered GAT's accuracy on Cora by ~4 points.
- PyG's `GCNConv` gets `cached=True` in full-batch training, PyG's own way to
  normalize the graph once; mlx-graphs caches the normalization
  automatically. It is off for graph classification, where every batch is a
  new graph.
- GAT / GATv2 run on a graph whose self-loops were added once beforehand, in
  both frameworks, and are built with `add_self_loops=False` in PyG.
- Every timed step ends with a synchronization (`mx.eval` /
  `torch.mps.synchronize`), and the first steps are excluded as warm-up.
- Each configuration runs in a fresh process under a memory cap (16 GB by
  default, `--memory-gb`): a run over the cap is killed and reported as out
  of memory. Peak memory is the process' peak *footprint*, as in Activity
  Monitor, which includes GPU (Metal) memory, unlike RSS. Note that PyTorch
  MPS alone takes ~1.4 GB.
- mlx-graphs runs with its defaults: fused message passing in the `"auto"`
  format, eager execution (no `mx.compile`).

## Not covered

- Neighbor-sampled mini-batch training: PyG's `NeighborLoader` needs
  `pyg-lib` or `torch-sparse`, which have no wheels for recent PyTorch on
  macOS, and building `torch-sparse` from source failed on the machine used
  for the published numbers. `benchmarks/framework_comparison/neighbor_loader`
  holds an older comparison.
- `torch.compile` for PyG and `mx.compile` for mlx-graphs.
