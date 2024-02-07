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
| benchmark_scatter / edg=(2, 1000) nod=(10, 64) sca=add             |   0.47 |   0.54 |   0.13 |    -72% |
| benchmark_scatter / edg=(2, 10000) nod=(10, 64) sca=add            |   1.72 |   1.97 |   0.36 |    -78% |
| benchmark_scatter / edg=(2, 100000) nod=(10, 64) sca=add           |   5.92 |  19.40 |   3.00 |    -49% |
| benchmark_scatter / edg=(2, 1000000) nod=(10, 64) sca=add          |  44.36 | 193.72 |  35.53 |    -19% |
| benchmark_scatter / edg=(2, 1000) nod=(100, 64) sca=add            |   0.93 |   0.54 |   0.13 |    -86% |
| benchmark_scatter / edg=(2, 10000) nod=(100, 64) sca=add           |   1.72 |   1.99 |   0.29 |    -82% |
| benchmark_scatter / edg=(2, 100000) nod=(100, 64) sca=add          |   4.25 |  19.40 |   2.68 |    -37% |
| benchmark_scatter / edg=(2, 1000000) nod=(100, 64) sca=add         |  44.36 | 193.88 |  30.56 |    -31% |
| benchmark_scatter / edg=(2, 1000) nod=(1000, 64) sca=add           |   0.49 |   0.56 |   0.15 |    -70% |
| benchmark_scatter / edg=(2, 10000) nod=(1000, 64) sca=add          |   1.73 |   2.00 |   0.29 |    -83% |
| benchmark_scatter / edg=(2, 100000) nod=(1000, 64) sca=add         |   5.49 |  19.42 |   2.66 |    -51% |
| benchmark_scatter / edg=(2, 1000000) nod=(1000, 64) sca=add        |  44.37 | 193.51 |  29.20 |    -34% |
| benchmark_scatter / edg=(2, 1000) nod=(10000, 64) sca=add          |   0.59 |   0.73 |   0.40 |    -33% |
| benchmark_scatter / edg=(2, 10000) nod=(10000, 64) sca=add         |   1.37 |   2.06 |   0.38 |    -72% |
| benchmark_scatter / edg=(2, 100000) nod=(10000, 64) sca=add        |   4.23 |  19.52 |   3.03 |    -28% |
| benchmark_scatter / edg=(2, 1000000) nod=(10000, 64) sca=add       |  44.42 | 193.94 |  27.22 |    -38% |
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
| benchmark_GCNConv / in_=64 out=64 edg=(2, 1000) nod=(10, 64)       |   0.78 |   0.80 |   0.33 |    -57% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 10000) nod=(10, 64)      |   2.40 |   2.31 |   0.81 |    -66% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 100000) nod=(10, 64)     |   5.40 |  22.03 |   5.01 |     -7% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 1000000) nod=(10, 64)    |  53.37 | 219.14 |  61.34 |    +14% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 1000) nod=(100, 64)      |   0.72 |   0.80 |   0.39 |    -45% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 10000) nod=(100, 64)     |   2.39 |   2.39 |   0.76 |    -68% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 100000) nod=(100, 64)    |   5.23 |  21.98 |   5.20 |      0% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 1000000) nod=(100, 64)   |  53.27 | 218.87 |  63.85 |    +19% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 1000) nod=(1000, 64)     |   0.83 |   0.91 |   0.28 |    -66% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 10000) nod=(1000, 64)    |   2.40 |   2.36 |   1.06 |    -55% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 100000) nod=(1000, 64)   |   5.16 |  22.06 |   5.38 |     +4% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 1000000) nod=(1000, 64)  |  53.27 | 220.68 |  62.06 |    +16% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 1000) nod=(10000, 64)    |   1.17 |   0.59 |   1.34 |    +14% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 10000) nod=(10000, 64)   |   1.35 |   2.65 |   1.90 |    +40% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 100000) nod=(10000, 64)  |   7.46 |  22.61 |   6.24 |    -16% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 1000000) nod=(10000, 64) |  53.36 | 220.14 |  63.73 |    +19% |

 Average benchmark:
| Operation                  | mlx_gpu | mlx_cpu | pyg_cpu | mlx_gpu/pyg_cpu speedup |
|----------------------------|-------|-------|-------|-----------------------|
| benchmark_scatter      |  12.90 |  53.95 |   8.50 |    -34% |
| benchmark_gather       |   3.11 |   4.68 |   7.25 |   +132% |
| benchmark_GCNConv      |  15.54 |  61.27 |  17.48 |    +12% |
