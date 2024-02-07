**M1 Pro (2E+8P+16GPU)**

Detailed benchmark:
| Operation                                                    | mlx_gpu | mlx_cpu | pyg_cpu | mlx_gpu/pyg_cpu speedup |
|--------------------------------------------------------------|-------|-------|-------|-----------------------|
| benchmark_gather / edg=(2, 1000) nod=(10, 64)            |   0.44 |   0.07 |   0.23 |    -47% |
| benchmark_gather / edg=(2, 10000) nod=(10, 64)           |   2.72 |   0.25 |   0.74 |    -72% |
| benchmark_gather / edg=(2, 100000) nod=(10, 64)          |   5.00 |   5.36 |   3.00 |    -39% |
| benchmark_gather / edg=(2, 1000000) nod=(10, 64)         |  17.24 |  57.80 |  38.03 |   +120% |
| benchmark_gather / edg=(2, 1000) nod=(100, 64)           |   5.62 |   0.07 |   0.21 |    -96% |
| benchmark_gather / edg=(2, 10000) nod=(100, 64)          |   8.06 |   0.29 |   0.73 |    -90% |
| benchmark_gather / edg=(2, 100000) nod=(100, 64)         |  12.98 |   4.83 |   2.67 |    -79% |
| benchmark_gather / edg=(2, 1000000) nod=(100, 64)        |  25.82 |  55.91 |  37.07 |    +43% |
| benchmark_gather / edg=(2, 1000) nod=(1000, 64)          |   0.46 |   0.08 |   0.20 |    -56% |
| benchmark_gather / edg=(2, 10000) nod=(1000, 64)         |   2.74 |   0.28 |   0.71 |    -74% |
| benchmark_gather / edg=(2, 100000) nod=(1000, 64)        |   8.12 |   5.77 |   2.88 |    -64% |
| benchmark_gather / edg=(2, 1000000) nod=(1000, 64)       |  20.40 |  62.45 |  37.53 |    +83% |
| benchmark_gather / edg=(2, 1000) nod=(10000, 64)         |   0.44 |   0.07 |   0.20 |    -53% |
| benchmark_gather / edg=(2, 10000) nod=(10000, 64)        |   0.80 |   0.45 |   0.73 |     -8% |
| benchmark_gather / edg=(2, 100000) nod=(10000, 64)       |   4.03 |   9.83 |   3.49 |    -13% |
| benchmark_gather / edg=(2, 1000000) nod=(10000, 64)      |  13.95 | 123.89 |  38.62 |   +176% |
| benchmark_fast_gather / edg=[2, 1000] nod=[10, 64]       |   0.40 |   0.07 |   0.24 |    -40% |
| benchmark_fast_gather / edg=[2, 10000] nod=[10, 64]      |   1.25 |   0.25 |   0.69 |    -44% |
| benchmark_fast_gather / edg=[2, 100000] nod=[10, 64]     |   3.70 |   3.65 |   2.63 |    -28% |
| benchmark_fast_gather / edg=[2, 1000000] nod=[10, 64]    |  14.73 |  12.11 |  35.80 |   +142% |
| benchmark_fast_gather / edg=[2, 1000] nod=[100, 64]      |   0.32 |   0.06 |   0.18 |    -43% |
| benchmark_fast_gather / edg=[2, 10000] nod=[100, 64]     |   0.88 |   0.24 |   0.77 |    -12% |
| benchmark_fast_gather / edg=[2, 100000] nod=[100, 64]    |  12.50 |   4.21 |   2.65 |    -78% |
| benchmark_fast_gather / edg=[2, 1000000] nod=[100, 64]   |  20.00 |  11.97 |  36.12 |    +80% |
| benchmark_fast_gather / edg=[2, 1000] nod=[1000, 64]     |   0.41 |   0.07 |   0.18 |    -55% |
| benchmark_fast_gather / edg=[2, 10000] nod=[1000, 64]    |   0.65 |   0.26 |   0.74 |    +13% |
| benchmark_fast_gather / edg=[2, 100000] nod=[1000, 64]   |   7.56 |   3.47 |   2.67 |    -64% |
| benchmark_fast_gather / edg=[2, 1000000] nod=[1000, 64]  |  15.29 |  11.94 |  35.87 |   +134% |
| benchmark_fast_gather / edg=[2, 1000] nod=[10000, 64]    |   0.44 |   0.08 |   0.21 |    -52% |
| benchmark_fast_gather / edg=[2, 10000] nod=[10000, 64]   |   1.48 |   0.35 |   0.77 |    -48% |
| benchmark_fast_gather / edg=[2, 100000] nod=[10000, 64]  |   3.74 |   3.57 |   3.12 |    -16% |
| benchmark_fast_gather / edg=[2, 1000000] nod=[10000, 64] |  12.48 |  12.31 |  36.47 |   +192% |

 Average benchmark:
| Operation                  | mlx_gpu | mlx_cpu | pyg_cpu | mlx_gpu/pyg_cpu speedup |
|----------------------------|-------|-------|-------|-----------------------|
| benchmark_gather       |   8.05 |  20.46 |  10.44 |    +29% |
| benchmark_fast_gather  |   5.99 |   4.04 |   9.94 |    +66% |
