# Average benchmark

Averaged runtime benchmark of `mlx-graphs` layers, measured in `milliseconds`.

* `mlx_gpu`: mlx framework with gpu backend
* `mlx_cpu`: mlx framework with cpu backend
* `pyg_cpu`: torch_geometric framework with cpu backend
* `pyg_mps`: torch_geometric framework with mps (gpu) backend
* `mlx_gpu/pyg_cpu speedup`: runtime speedup of mlx_gpu compared to pyg_cpu
* `mlx_gpu/pyg_mps speedup`: runtime speedup of mlx_gpu compared to pyg_mps

## Apple Silicon

**M1 Pro (2E+8P+16GPU)**

| Operation              | mlx_gpu | mlx_cpu | pyg_mps | pyg_cpu | mlx_gpu/pyg_cpu speedup | mlx_gpu/pyg_mps speedup |
|------------------------|-------|-------|-------|-------|-----------------------|-----------------------|
| benchmark_GCNConv  |   0.46 |   0.98 |   4.07 |   0.22 |    -52% |   +789% |
| benchmark_GATConv  |   0.47 |   1.24 | nan |   0.24 |    -48% | nan |
