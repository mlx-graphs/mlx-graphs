# mlx-graphs vs PyTorch Geometric on Apple Silicon

Compares mlx-graphs (MLX, GPU) with PyTorch Geometric on the same Mac, on
both of PyTorch's backends: MPS (the Apple GPU) and CPU.

```bash
pip install -e '.[benchmarks]'
python benchmarks/pyg_comparison/run.py          # ~1 hour; results/<chip>/results.jsonl
python benchmarks/pyg_comparison/plot.py         # figures/ and results.md
```

`run.py --quick` runs the smallest configuration of every task as a smoke
test. Datasets download into `.mlx_graphs_data/` on first use.

## What is measured

| task | setup | reported |
|---|---|---|
| `layer` | one convolution layer (GCN, GraphSAGE, GIN, GAT, GATv2) on uniform random graphs of 100k, 500k and 2M edges; 128 features in and out, 4 heads for the attention layers | median time of a forward pass and of a forward + backward pass; peak memory |
| `node` | full-batch node classification, 2-layer GCN / GraphSAGE / GAT, 64 hidden units, dropout 0.5, Adam (lr 0.01), 200 epochs (100 on ogbn-arxiv), on Cora, CiteSeer, PubMed, Amazon Photo / Computers, Coauthor CS / Physics and ogbn-arxiv | median time per training step; test accuracy at the best validation accuracy |
| `graph` | graph classification on TU datasets (MUTAG, PROTEINS, NCI1, DD), 3 layers + mean pooling, batches of 64 graphs, 30 epochs | median time per epoch, *including loading and batching the graphs*; test accuracy on a random 80/20 split |

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
