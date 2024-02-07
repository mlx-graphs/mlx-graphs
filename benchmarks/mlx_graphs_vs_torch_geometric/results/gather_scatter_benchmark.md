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
| benchmark_scatter / edg=(2, 1000) nod=(10, 64) sca=add             |   0.66 |   0.56 |   0.16 |    -75% |
| benchmark_scatter / edg=(2, 10000) nod=(10, 64) sca=add            |   1.94 |   4.99 |   1.51 |    -22% |
| benchmark_scatter / edg=(2, 100000) nod=(10, 64) sca=add           |   6.58 |  19.39 |   3.28 |    -50% |
| benchmark_scatter / edg=(2, 1000000) nod=(10, 64) sca=add          |  44.51 | 194.10 |  37.94 |    -14% |
| benchmark_scatter / edg=(2, 1000) nod=(100, 64) sca=add            |   0.67 |   0.57 |   0.19 |    -72% |
| benchmark_scatter / edg=(2, 10000) nod=(100, 64) sca=add           |   1.78 |   4.98 |   1.22 |    -31% |
| benchmark_scatter / edg=(2, 100000) nod=(100, 64) sca=add          |   6.01 |  19.60 |   2.76 |    -54% |
| benchmark_scatter / edg=(2, 1000000) nod=(100, 64) sca=add         |  44.53 | 194.45 |  30.79 |    -30% |
| benchmark_scatter / edg=(2, 1000) nod=(1000, 64) sca=add           |   0.56 |   0.59 |   0.22 |    -60% |
| benchmark_scatter / edg=(2, 10000) nod=(1000, 64) sca=add          |   1.56 |   4.62 |   1.70 |     +9% |
| benchmark_scatter / edg=(2, 100000) nod=(1000, 64) sca=add         |   6.58 |  21.09 |   2.83 |    -56% |
| benchmark_scatter / edg=(2, 1000000) nod=(1000, 64) sca=add        |  44.60 | 194.19 |  31.02 |    -30% |
| benchmark_scatter / edg=(2, 1000) nod=(10000, 64) sca=add          |   0.68 |   0.78 |   0.55 |    -18% |
| benchmark_scatter / edg=(2, 10000) nod=(10000, 64) sca=add         |   2.12 |   5.40 |   2.53 |    +19% |
| benchmark_scatter / edg=(2, 100000) nod=(10000, 64) sca=add        |   5.54 |  19.66 |   3.13 |    -43% |
| benchmark_scatter / edg=(2, 1000000) nod=(10000, 64) sca=add       |  44.57 | 194.75 |  29.40 |    -34% |
| benchmark_gather / edg=(2, 1000) nod=(10, 64)                      |   0.61 |   0.09 |   0.21 |    -66% |
| benchmark_gather / edg=(2, 10000) nod=(10, 64)                     |   2.65 |   0.55 |   1.27 |    -51% |
| benchmark_gather / edg=(2, 100000) nod=(10, 64)                    |   8.91 |   4.90 |   5.28 |    -40% |
| benchmark_gather / edg=(2, 1000000) nod=(10, 64)                   |  79.78 |  35.42 |  24.93 |    -68% |
| benchmark_gather / edg=(2, 1000) nod=(100, 64)                     |   0.48 |   0.09 |   0.23 |    -52% |
| benchmark_gather / edg=(2, 10000) nod=(100, 64)                    |   2.49 |   0.56 |   1.04 |    -58% |
| benchmark_gather / edg=(2, 100000) nod=(100, 64)                   |   8.99 |   6.02 |   3.08 |    -65% |
| benchmark_gather / edg=(2, 1000000) nod=(100, 64)                  |  79.72 |  37.87 |  25.55 |    -67% |
| benchmark_gather / edg=(2, 1000) nod=(1000, 64)                    |   0.52 |   0.09 |   0.28 |    -45% |
| benchmark_gather / edg=(2, 10000) nod=(1000, 64)                   |   2.38 |   0.62 |   1.42 |    -40% |
| benchmark_gather / edg=(2, 100000) nod=(1000, 64)                  |   9.74 |   7.08 |   3.44 |    -64% |
| benchmark_gather / edg=(2, 1000000) nod=(1000, 64)                 |  79.69 |  40.16 |  26.55 |    -66% |
| benchmark_gather / edg=(2, 1000) nod=(10000, 64)                   |   0.46 |   0.09 |   0.26 |    -44% |
| benchmark_gather / edg=(2, 10000) nod=(10000, 64)                  |   1.69 |   0.72 |   1.98 |    +17% |
| benchmark_gather / edg=(2, 100000) nod=(10000, 64)                 |   7.99 |   8.61 |   2.36 |    -70% |
| benchmark_gather / edg=(2, 1000000) nod=(10000, 64)                |  79.68 |  39.46 |  26.73 |    -66% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 1000) nod=(10, 64)       |   1.11 |   0.85 |   0.51 |    -53% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 10000) nod=(10, 64)      |   2.37 |   5.25 |   2.00 |    -15% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 100000) nod=(10, 64)     |   9.21 |  24.68 |   5.49 |    -40% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 1000000) nod=(10, 64)    |  88.33 | 250.65 |  59.28 |    -32% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 1000) nod=(100, 64)      |   0.98 |   0.89 |   0.54 |    -44% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 10000) nod=(100, 64)     |   3.04 |   6.18 |   2.51 |    -17% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 100000) nod=(100, 64)    |   8.83 |  24.70 |   5.55 |    -37% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 1000000) nod=(100, 64)   |  88.33 | 251.43 |  61.18 |    -30% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 1000) nod=(1000, 64)     |   0.98 |   1.05 |   0.79 |    -19% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 10000) nod=(1000, 64)    |   2.34 |   6.41 |   2.95 |    +26% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 100000) nod=(1000, 64)   |  10.07 |  24.92 |   5.52 |    -45% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 1000000) nod=(1000, 64)  |  88.32 | 251.51 |  60.36 |    -31% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 1000) nod=(10000, 64)    |   2.22 |   2.13 |   3.62 |    +63% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 10000) nod=(10000, 64)   |   2.35 |   5.49 |   4.76 |   +102% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 100000) nod=(10000, 64)  |   9.46 |  25.73 |   6.81 |    -28% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 1000000) nod=(10000, 64) |  88.42 | 252.48 |  62.36 |    -29% |

 Average benchmark:
| Operation              | mlx_gpu | mlx_cpu | pyg_cpu | mlx_gpu/pyg_cpu speedup |
|------------------------|-------|-------|-------|-----------------------|
| benchmark_scatter  |  13.31 |  54.98 |   9.33 |    -29% |
| benchmark_gather   |  22.86 |  11.40 |   7.79 |    -65% |
| benchmark_GCNConv  |  25.40 |  70.90 |  17.76 |    -30% |

