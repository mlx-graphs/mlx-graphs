# Detailed benchmark

Averaged runtime benchmark of `mlx-graphs` layers, measured in `milliseconds`.

* `mlx_gpu`: mlx framework with gpu backend
* `mlx_cpu`: mlx framework with cpu backend
* `pyg_cpu`: torch_geometric framework with cpu backend
* `pyg_mps`: torch_geometric framework with mps (gpu) backend
* `mlx_gpu/pyg_cpu speedup`: runtime speedup of mlx_gpu compared to pyg_cpu
* `mlx_gpu/pyg_mps speedup`: runtime speedup of mlx_gpu compared to pyg_mps

## Apple Silicon

**M1 Pro (2E+8P+16GPU)**

| Operation                                                             | mlx_gpu | mlx_cpu | pyg_mps | pyg_cpu | mlx_gpu/pyg_cpu speedup | mlx_gpu/pyg_mps speedup |
|-----------------------------------------------------------------------|-------|-------|-------|-------|-----------------------|-----------------------|
| benchmark_GCNConv / in_=64 out=128 edg=(2, 100000) nod=(100, 64)  |   0.14 |   0.32 |   1.16 |   0.08 |    -46% |   +704% |
| benchmark_GCNConv / in_=64 out=128 edg=(2, 1000000) nod=(100, 64) |   1.48 |   3.07 |  14.68 |   0.62 |    -57% |   +891% |
| benchmark_GCNConv / in_=8 out=16 edg=(2, 1000000) nod=(1000, 8)   |   0.19 |   0.52 |   0.32 |   0.16 |    -18% |    +66% |
| benchmark_GCNConv / in_=8 out=16 edg=(2, 1000) nod=(100, 8)       |   0.01 |   0.00 |   0.12 |   0.00 |    -89% |   +920% |
| benchmark_GATConv / in_=64 out=128 edg=(2, 100000) nod=(100, 64)  |   0.15 |   0.38 | nan |   0.08 |    -44% | nan |
| benchmark_GATConv / in_=64 out=128 edg=(2, 1000000) nod=(100, 64) |   1.51 |   3.87 | nan |   0.69 |    -54% | nan |
| benchmark_GATConv / in_=8 out=16 edg=(2, 1000000) nod=(1000, 8)   |   0.20 |   0.69 | nan |   0.20 |      0% | nan |
| benchmark_GATConv / in_=8 out=16 edg=(2, 1000) nod=(100, 8)       |   0.02 |   0.00 | nan |   0.00 |    -92% | nan |
