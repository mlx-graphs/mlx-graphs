**M1 Pro (2E+8P+16GPU)**

Detailed benchmark:
| Operation                                                              | mlx_gpu | mlx_cpu | pyg_cpu | mlx_gpu/pyg_cpu speedup |
|------------------------------------------------------------------------|-------|-------|-------|-----------------------|
| benchmark_gather / edg=(2, 1000) nod=(10, 64)                      |   0.33 |   0.04 |   0.15 |    -53% |
| benchmark_gather / edg=(2, 10000) nod=(10, 64)                     |   0.73 |   0.22 |   0.37 |    -49% |
| benchmark_gather / edg=(2, 100000) nod=(10, 64)                    |   1.89 |   2.04 |   2.77 |    +46% |
| benchmark_gather / edg=(2, 1000000) nod=(10, 64)                   |  11.71 |  19.00 |  35.64 |   +204% |
| benchmark_gather / edg=(2, 1000) nod=(100, 64)                     |   0.33 |   0.04 |   0.14 |    -57% |
| benchmark_gather / edg=(2, 10000) nod=(100, 64)                    |   0.71 |   0.22 |   0.40 |    -44% |
| benchmark_gather / edg=(2, 100000) nod=(100, 64)                   |   1.45 |   1.96 |   2.57 |    +77% |
| benchmark_gather / edg=(2, 1000000) nod=(100, 64)                  |  11.73 |  19.06 |  36.66 |   +212% |
| benchmark_gather / edg=(2, 1000) nod=(1000, 64)                    |   0.35 |   0.04 |   0.14 |    -59% |
| benchmark_gather / edg=(2, 10000) nod=(1000, 64)                   |   0.70 |   0.24 |   0.38 |    -45% |
| benchmark_gather / edg=(2, 100000) nod=(1000, 64)                  |   1.44 |   2.13 |   2.54 |    +76% |
| benchmark_gather / edg=(2, 1000000) nod=(1000, 64)                 |  11.71 |  21.53 |  36.21 |   +209% |
| benchmark_gather / edg=(2, 1000) nod=(10000, 64)                   |   0.30 |   0.05 |   0.14 |    -52% |
| benchmark_gather / edg=(2, 10000) nod=(10000, 64)                  |   0.43 |   0.36 |   0.55 |    +26% |
| benchmark_gather / edg=(2, 100000) nod=(10000, 64)                 |   1.83 |   3.82 |   2.98 |    +62% |
| benchmark_gather / edg=(2, 1000000) nod=(10000, 64)                |  11.77 |  43.86 |  35.89 |   +204% |
| benchmark_fast_gather / edg=[2, 1000] nod=[10, 64]                 |   0.27 |   0.05 |   0.17 |    -37% |
| benchmark_fast_gather / edg=[2, 10000] nod=[10, 64]                |   0.46 |   0.22 |   0.33 |    -27% |
| benchmark_fast_gather / edg=[2, 100000] nod=[10, 64]               |   1.45 |   1.41 |   2.43 |    +66% |
| benchmark_fast_gather / edg=[2, 1000000] nod=[10, 64]              |  12.76 |  11.72 |  35.64 |   +179% |
| benchmark_fast_gather / edg=[2, 1000] nod=[100, 64]                |   0.28 |   0.05 |   0.13 |    -53% |
| benchmark_fast_gather / edg=[2, 10000] nod=[100, 64]               |   0.49 |   0.22 |   0.38 |    -22% |
| benchmark_fast_gather / edg=[2, 100000] nod=[100, 64]              |   1.47 |   1.43 |   2.40 |    +62% |
| benchmark_fast_gather / edg=[2, 1000000] nod=[100, 64]             |  11.74 |  11.75 |  35.81 |   +204% |
| benchmark_fast_gather / edg=[2, 1000] nod=[1000, 64]               |   0.29 |   0.06 |   0.15 |    -47% |
| benchmark_fast_gather / edg=[2, 10000] nod=[1000, 64]              |   0.52 |   0.24 |   0.50 |     -3% |
| benchmark_fast_gather / edg=[2, 100000] nod=[1000, 64]             |   1.46 |   1.41 |   2.54 |    +74% |
| benchmark_fast_gather / edg=[2, 1000000] nod=[1000, 64]            |  11.79 |  11.74 |  35.73 |   +203% |
| benchmark_fast_gather / edg=[2, 1000] nod=[10000, 64]              |   0.33 |   0.05 |   0.16 |    -52% |
| benchmark_fast_gather / edg=[2, 10000] nod=[10000, 64]             |   1.17 |   0.39 |   0.46 |    -60% |
| benchmark_fast_gather / edg=[2, 100000] nod=[10000, 64]            |   1.45 |   1.41 |   3.66 |   +152% |
| benchmark_fast_gather / edg=[2, 1000000] nod=[10000, 64]           |  11.70 |  11.69 |  34.54 |   +195% |

 Average benchmark:
| Operation                  | mlx_gpu | mlx_cpu | pyg_cpu | mlx_gpu/pyg_cpu speedup |
|----------------------------|-------|-------|-------|-----------------------|
| benchmark_gather       |   3.59 |   7.16 |   9.85 |   +174% |
| benchmark_fast_gather  |   3.60 |   3.36 |   9.69 |   +169% |

**M3 Pro (6E+5P+14GPU)**

Detailed benchmark:
| Operation                                                              | mlx_gpu | mlx_cpu | pyg_cpu | mlx_gpu/pyg_cpu speedup |
|------------------------------------------------------------------------|-------|-------|-------|-----------------------|
| benchmark_gather / edg=(2, 1000) nod=(10, 64)                      |   0.25 |   0.08 |   0.17 |    -31% |
| benchmark_gather / edg=(2, 10000) nod=(10, 64)                     |   0.95 |   0.48 |   0.93 |     -1% |
| benchmark_gather / edg=(2, 100000) nod=(10, 64)                    |   2.02 |   1.51 |   1.62 |    -19% |
| benchmark_gather / edg=(2, 1000000) nod=(10, 64)                   |   9.72 |  14.78 |  25.18 |   +159% |
| benchmark_gather / edg=(2, 1000) nod=(100, 64)                     |   0.25 |   0.07 |   0.20 |    -20% |
| benchmark_gather / edg=(2, 10000) nod=(100, 64)                    |   0.67 |   0.49 |   0.88 |    +31% |
| benchmark_gather / edg=(2, 100000) nod=(100, 64)                   |   1.66 |   1.53 |   1.63 |     -1% |
| benchmark_gather / edg=(2, 1000000) nod=(100, 64)                  |   9.80 |  14.80 |  25.53 |   +160% |
| benchmark_gather / edg=(2, 1000) nod=(1000, 64)                    |   0.26 |   0.08 |   0.20 |    -21% |
| benchmark_gather / edg=(2, 10000) nod=(1000, 64)                   |   0.35 |   0.57 |   0.89 |   +153% |
| benchmark_gather / edg=(2, 100000) nod=(1000, 64)                  |   1.73 |   1.84 |   1.70 |     -1% |
| benchmark_gather / edg=(2, 1000000) nod=(1000, 64)                 |   9.82 |  17.73 |  26.58 |   +170% |
| benchmark_gather / edg=(2, 1000) nod=(10000, 64)                   |   0.26 |   0.09 |   0.22 |    -18% |
| benchmark_gather / edg=(2, 10000) nod=(10000, 64)                  |   0.65 |   0.70 |   1.20 |    +84% |
| benchmark_gather / edg=(2, 100000) nod=(10000, 64)                 |   1.63 |   1.84 |   1.77 |     +8% |
| benchmark_gather / edg=(2, 1000000) nod=(10000, 64)                |   9.80 |  18.31 |  27.32 |   +178% |
| benchmark_fast_gather / edg=[2, 1000] nod=[10, 64]                 |   0.24 |   0.08 |   0.20 |    -17% |
| benchmark_fast_gather / edg=[2, 10000] nod=[10, 64]                |   0.68 |   0.48 |   0.65 |     -3% |
| benchmark_fast_gather / edg=[2, 100000] nod=[10, 64]               |   1.65 |   1.73 |   1.61 |     -2% |
| benchmark_fast_gather / edg=[2, 1000000] nod=[10, 64]              |   9.72 |   9.70 |  25.93 |   +166% |
| benchmark_fast_gather / edg=[2, 1000] nod=[100, 64]                |   0.24 |   0.08 |   0.20 |    -16% |
| benchmark_fast_gather / edg=[2, 10000] nod=[100, 64]               |   0.69 |   0.50 |   0.81 |    +18% |
| benchmark_fast_gather / edg=[2, 100000] nod=[100, 64]              |   1.67 |   1.64 |   1.66 |      0% |
| benchmark_fast_gather / edg=[2, 1000000] nod=[100, 64]             |   9.78 |   9.75 |  25.73 |   +163% |
| benchmark_fast_gather / edg=[2, 1000] nod=[1000, 64]               |   0.25 |   0.09 |   0.20 |    -18% |
| benchmark_fast_gather / edg=[2, 10000] nod=[1000, 64]              |   0.74 |   0.57 |   1.09 |    +47% |
| benchmark_fast_gather / edg=[2, 100000] nod=[1000, 64]             |   1.68 |   1.63 |   1.69 |     +0% |
| benchmark_fast_gather / edg=[2, 1000000] nod=[1000, 64]            |   9.88 |   9.78 |  26.53 |   +168% |
| benchmark_fast_gather / edg=[2, 1000] nod=[10000, 64]              |   0.25 |   0.09 |   0.12 |    -51% |
| benchmark_fast_gather / edg=[2, 10000] nod=[10000, 64]             |   0.94 |   0.71 |   1.16 |    +24% |
| benchmark_fast_gather / edg=[2, 100000] nod=[10000, 64]            |   1.67 |   1.64 |   1.76 |     +5% |
| benchmark_fast_gather / edg=[2, 1000000] nod=[10000, 64]           |   9.76 |   9.75 |  26.90 |   +175% |

 Average benchmark:
| Operation                  | mlx_gpu | mlx_cpu | pyg_cpu | mlx_gpu/pyg_cpu speedup |
|----------------------------|-------|-------|-------|-----------------------|
| benchmark_gather       |   3.11 |   4.68 |   7.25 |   +132% |
| benchmark_fast_gather  |   3.11 |   3.01 |   7.27 |   +133% |
