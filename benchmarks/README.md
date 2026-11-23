# Benchmarks

This folder contains benchmarks for mlx-graphs.

They're divided in
- pyg_comparison: the current end-to-end comparison with PyTorch Geometric
  (layers, node classification, graph classification), with figures. Start here.
- architectural: used to make architectural decisions
- framework comparison: older comparisons with other frameworks. The DGL and
  torch_cluster scripts need `pip install dgl torch_cluster`, which are not
  part of the `benchmarks` extra anymore.
- spmm: fused (SpMM) message passing against gather-scatter
- profiling: useful to profile operations

`run_capped.py` runs any command under a hard memory cap
(`python benchmarks/run_capped.py --gb 16 -- python my_script.py`), so that a
benchmark that grows out of control is killed instead of exhausting the
machine's memory.
