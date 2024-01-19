## Average benchmark

| Operation              | mlx_gpu | mlx_cpu | pyg_mps | pyg_cpu | mlx_gpu/pyg_cpu speedup | mlx_gpu/pyg_mps speedup |
|------------------------|-------|-------|-------|-------|-----------------------|-----------------------|
| benchmark_GCNConv  |   0.46 |   0.98 |   4.07 |   0.22 |    -52% |   +789% |
| benchmark_GATConv  |   0.47 |   1.24 | nan |   0.24 |    -48% | nan |
