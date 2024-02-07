**M1 Pro (2E+8P+16GPU)**

Detailed benchmark:
| Operation                                                              | mlx_gpu | mlx_cpu | pyg_cpu | mlx_gpu/pyg_cpu speedup |
|------------------------------------------------------------------------|-------|-------|-------|-----------------------|
| benchmark_scatter / edg=(2, 1000) nod=(10, 64) sca=add             |   0.89 |   0.29 |   0.11 |    -87% |
| benchmark_scatter / edg=(2, 10000) nod=(10, 64) sca=add            |   2.33 |   2.33 |   0.64 |    -72% |
| benchmark_scatter / edg=(2, 100000) nod=(10, 64) sca=add           |   8.93 |  22.91 |   5.40 |    -39% |
| benchmark_scatter / edg=(2, 1000000) nod=(10, 64) sca=add          |  63.06 | 233.44 |  56.40 |    -10% |
| benchmark_scatter / edg=(2, 1000) nod=(100, 64) sca=add            |   0.79 |   0.28 |   0.10 |    -86% |
| benchmark_scatter / edg=(2, 10000) nod=(100, 64) sca=add           |   2.29 |   2.43 |   0.59 |    -74% |
| benchmark_scatter / edg=(2, 100000) nod=(100, 64) sca=add          |   6.99 |  22.92 |   5.52 |    -21% |
| benchmark_scatter / edg=(2, 1000000) nod=(100, 64) sca=add         |  61.74 | 227.87 |  49.05 |    -20% |
| benchmark_scatter / edg=(2, 1000) nod=(1000, 64) sca=add           |   0.83 |   0.29 |   0.16 |    -81% |
| benchmark_scatter / edg=(2, 10000) nod=(1000, 64) sca=add          |   2.81 |   2.39 |   0.66 |    -76% |
| benchmark_scatter / edg=(2, 100000) nod=(1000, 64) sca=add         |   7.09 |  23.34 |   5.21 |    -26% |
| benchmark_scatter / edg=(2, 1000000) nod=(1000, 64) sca=add        |  61.68 | 227.78 |  49.52 |    -19% |
| benchmark_scatter / edg=(2, 1000) nod=(10000, 64) sca=add          |   0.91 |   0.40 |   0.52 |    -43% |
| benchmark_scatter / edg=(2, 10000) nod=(10000, 64) sca=add         |   2.35 |   2.57 |   0.96 |    -59% |
| benchmark_scatter / edg=(2, 100000) nod=(10000, 64) sca=add        |  10.05 |  23.75 |   5.54 |    -44% |
| benchmark_scatter / edg=(2, 1000000) nod=(10000, 64) sca=add       |  61.66 | 235.02 |  48.65 |    -21% |
| benchmark_gather / edg=(2, 1000) nod=(10, 64)                      |   0.39 |   0.06 |   0.21 |    -46% |
| benchmark_gather / edg=(2, 10000) nod=(10, 64)                     |   2.69 |   0.24 |   0.68 |    -74% |
| benchmark_gather / edg=(2, 100000) nod=(10, 64)                    |   6.39 |   4.16 |   2.57 |    -59% |
| benchmark_gather / edg=(2, 1000000) nod=(10, 64)                   |  14.60 |  48.06 |  35.59 |   +143% |
| benchmark_gather / edg=(2, 1000) nod=(100, 64)                     |   4.08 |   0.06 |   0.19 |    -95% |
| benchmark_gather / edg=(2, 10000) nod=(100, 64)                    |   7.61 |   0.25 |   0.80 |    -89% |
| benchmark_gather / edg=(2, 100000) nod=(100, 64)                   |  12.55 |   4.08 |   2.59 |    -79% |
| benchmark_gather / edg=(2, 1000000) nod=(100, 64)                  |  20.84 |  48.50 |  35.54 |    +70% |
| benchmark_gather / edg=(2, 1000) nod=(1000, 64)                    |   0.43 |   0.07 |   0.18 |    -57% |
| benchmark_gather / edg=(2, 10000) nod=(1000, 64)                   |   2.58 |   0.27 |   0.72 |    -72% |
| benchmark_gather / edg=(2, 100000) nod=(1000, 64)                  |   7.30 |   4.90 |   2.73 |    -62% |
| benchmark_gather / edg=(2, 1000000) nod=(1000, 64)                 |  15.11 |  50.61 |  35.54 |   +135% |
| benchmark_gather / edg=(2, 1000) nod=(10000, 64)                   |   0.45 |   0.08 |   0.23 |    -50% |
| benchmark_gather / edg=(2, 10000) nod=(10000, 64)                  |   0.72 |   0.43 |   0.87 |    +20% |
| benchmark_gather / edg=(2, 100000) nod=(10000, 64)                 |   4.35 |   7.13 |   3.01 |    -30% |
| benchmark_gather / edg=(2, 1000000) nod=(10000, 64)                |  13.51 |  84.37 |  35.94 |   +166% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 1000) nod=(10, 64)       |   2.45 |   0.76 |   0.34 |    -86% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 10000) nod=(10, 64)      |   9.25 |   3.50 |   1.54 |    -83% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 100000) nod=(10, 64)     |  34.35 |  29.67 |   8.96 |    -73% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 1000000) nod=(10, 64)    | 416.31 | 288.62 |  94.37 |    -77% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 1000) nod=(100, 64)      |   2.08 |   0.73 |   0.42 |    -79% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 10000) nod=(100, 64)     |   4.20 |   3.53 |   1.86 |    -55% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 100000) nod=(100, 64)    |  12.81 |  30.15 |   9.12 |    -28% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 1000000) nod=(100, 64)   | 123.46 | 289.06 |  92.28 |    -25% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 1000) nod=(1000, 64)     |   2.07 |   0.82 |   0.71 |    -65% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 10000) nod=(1000, 64)    |   4.40 |   3.71 |   2.13 |    -51% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 100000) nod=(1000, 64)   |  14.09 |  30.65 |   9.25 |    -34% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 1000000) nod=(1000, 64)  |  84.17 | 301.61 |  98.40 |    +16% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 1000) nod=(10000, 64)    |   2.28 |   1.57 |   3.73 |    +63% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 10000) nod=(10000, 64)   |   4.88 |   4.74 |   4.61 |     -5% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 100000) nod=(10000, 64)  |  12.55 |  35.92 |  10.95 |    -12% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 1000000) nod=(10000, 64) |  74.32 | 347.70 |  99.29 |    +33% |

 Average benchmark:
| Operation              | mlx_gpu | mlx_cpu | pyg_cpu | mlx_gpu/pyg_cpu speedup |
|------------------------|-------|-------|-------|-----------------------|
| benchmark_scatter  |  18.40 |  64.25 |  14.31 |    -22% |
| benchmark_gather   |   7.10 |  15.83 |   9.84 |    +38% |
| benchmark_GCNConv  |  50.23 |  85.80 |  27.37 |    -45% |


**M3 Pro (6E+5P+14GPU)**

Detailed benchmark:
| Operation                                                              | mlx_gpu | mlx_cpu | pyg_cpu | mlx_gpu/pyg_cpu speedup |
|------------------------------------------------------------------------|-------|-------|-------|-----------------------|
| benchmark_scatter / edg=(2, 1000) nod=(10, 64) sca=add             |   0.52 |   0.48 |   0.12 |    -76% |
| benchmark_scatter / edg=(2, 10000) nod=(10, 64) sca=add            |   1.88 |   4.94 |   1.45 |    -23% |
| benchmark_scatter / edg=(2, 100000) nod=(10, 64) sca=add           |   7.69 |  23.29 |   3.62 |    -53% |
| benchmark_scatter / edg=(2, 1000000) nod=(10, 64) sca=add          |  44.64 | 194.25 |  36.46 |    -18% |
| benchmark_scatter / edg=(2, 1000) nod=(100, 64) sca=add            |   0.55 |   0.56 |   0.14 |    -74% |
| benchmark_scatter / edg=(2, 10000) nod=(100, 64) sca=add           |   1.80 |   5.07 |   1.20 |    -33% |
| benchmark_scatter / edg=(2, 100000) nod=(100, 64) sca=add          |   7.23 |  19.55 |   2.80 |    -61% |
| benchmark_scatter / edg=(2, 1000000) nod=(100, 64) sca=add         |  44.49 | 194.22 |  31.31 |    -29% |
| benchmark_scatter / edg=(2, 1000) nod=(1000, 64) sca=add           |   0.73 |   0.62 |   0.16 |    -77% |
| benchmark_scatter / edg=(2, 10000) nod=(1000, 64) sca=add          |   1.94 |   5.59 |   0.97 |    -49% |
| benchmark_scatter / edg=(2, 100000) nod=(1000, 64) sca=add         |   7.02 |  19.49 |   2.78 |    -60% |
| benchmark_scatter / edg=(2, 1000000) nod=(1000, 64) sca=add        |  44.51 | 193.88 |  30.65 |    -31% |
| benchmark_scatter / edg=(2, 1000) nod=(10000, 64) sca=add          |   0.83 |   0.80 |   0.45 |    -45% |
| benchmark_scatter / edg=(2, 10000) nod=(10000, 64) sca=add         |   1.91 |   5.37 |   1.20 |    -37% |
| benchmark_scatter / edg=(2, 100000) nod=(10000, 64) sca=add        |   6.65 |  20.44 |   3.09 |    -53% |
| benchmark_scatter / edg=(2, 1000000) nod=(10000, 64) sca=add       |  44.57 | 194.31 |  30.67 |    -31% |
| benchmark_gather / edg=(2, 1000) nod=(10, 64)                      |   0.34 |   0.11 |   0.24 |    -27% |
| benchmark_gather / edg=(2, 10000) nod=(10, 64)                     |   0.75 |   0.53 |   1.56 |   +106% |
| benchmark_gather / edg=(2, 100000) nod=(10, 64)                    |   4.37 |   6.87 |   5.66 |    +29% |
| benchmark_gather / edg=(2, 1000000) nod=(10, 64)                   |  12.99 |  36.48 |  26.86 |   +106% |
| benchmark_gather / edg=(2, 1000) nod=(100, 64)                     |   0.35 |   0.09 |   0.25 |    -28% |
| benchmark_gather / edg=(2, 10000) nod=(100, 64)                    |   1.23 |   0.55 |   1.30 |     +5% |
| benchmark_gather / edg=(2, 100000) nod=(100, 64)                   |   2.79 |   6.13 |   5.85 |   +109% |
| benchmark_gather / edg=(2, 1000000) nod=(100, 64)                  |  14.36 |  36.77 |  26.61 |    +85% |
| benchmark_gather / edg=(2, 1000) nod=(1000, 64)                    |   0.50 |   0.09 |   0.29 |    -42% |
| benchmark_gather / edg=(2, 10000) nod=(1000, 64)                   |   1.58 |   0.78 |   1.29 |    -18% |
| benchmark_gather / edg=(2, 100000) nod=(1000, 64)                  |   3.93 |   7.48 |   6.00 |    +52% |
| benchmark_gather / edg=(2, 1000000) nod=(1000, 64)                 |  14.71 |  38.87 |  27.23 |    +85% |
| benchmark_gather / edg=(2, 1000) nod=(10000, 64)                   |   0.41 |   0.84 |   0.24 |    -41% |
| benchmark_gather / edg=(2, 10000) nod=(10000, 64)                  |   0.63 |   0.71 |   1.52 |   +143% |
| benchmark_gather / edg=(2, 100000) nod=(10000, 64)                 |   2.44 |   9.47 |   2.87 |    +17% |
| benchmark_gather / edg=(2, 1000000) nod=(10000, 64)                |  14.50 |  39.93 |  27.74 |    +91% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 1000) nod=(10, 64)       |   0.96 |   0.74 |   0.39 |    -59% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 10000) nod=(10, 64)      |   2.31 |   6.13 |   2.52 |     +8% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 100000) nod=(10, 64)     |   7.36 |  24.58 |   5.48 |    -25% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 1000000) nod=(10, 64)    |  53.55 | 249.05 |  67.22 |    +25% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 1000) nod=(100, 64)      |   1.65 |   0.90 |   0.55 |    -66% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 10000) nod=(100, 64)     |   1.83 |   6.03 |   2.67 |    +45% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 100000) nod=(100, 64)    |   8.07 |  25.20 |   5.54 |    -31% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 1000000) nod=(100, 64)   |  53.44 | 249.53 |  66.36 |    +24% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 1000) nod=(1000, 64)     |   1.10 |   1.04 |   0.77 |    -30% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 10000) nod=(1000, 64)    |   2.34 |   6.36 |   3.19 |    +36% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 100000) nod=(1000, 64)   |   9.40 |  25.22 |   5.70 |    -39% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 1000000) nod=(1000, 64)  |  53.46 | 250.78 |  64.50 |    +20% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 1000) nod=(10000, 64)    |   1.28 |   3.11 |   3.95 |   +208% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 10000) nod=(10000, 64)   |   2.67 |   7.67 |   5.46 |   +104% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 100000) nod=(10000, 64)  |   7.42 |  25.65 |   7.00 |     -5% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 1000000) nod=(10000, 64) |  53.54 | 251.35 |  65.88 |    +23% |

 Average benchmark:
| Operation              | mlx_gpu | mlx_cpu | pyg_cpu | mlx_gpu/pyg_cpu speedup |
|------------------------|-------|-------|-------|-----------------------|
| benchmark_scatter  |  13.56 |  55.18 |   9.19 |    -32% |
| benchmark_gather   |   4.74 |  11.61 |   8.47 |    +78% |
| benchmark_GCNConv  |  16.27 |  70.83 |  19.20 |    +17% |
