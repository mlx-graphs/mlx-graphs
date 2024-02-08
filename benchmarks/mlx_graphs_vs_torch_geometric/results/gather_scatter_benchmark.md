**M1 Pro (2E+8P+16GPU)**

Detailed benchmark:
| Operation                                                              | mlx_gpu | mlx_cpu | pyg_cpu | mlx_gpu/pyg_cpu speedup |
|------------------------------------------------------------------------|-------|-------|-------|-----------------------|
| benchmark_scatter / edg=(2, 1000) nod=(10, 64) sca=add             |   0.69 |   0.28 |   0.08 |    -88% |
| benchmark_scatter / edg=(2, 10000) nod=(10, 64) sca=add            |   1.19 |   2.31 |   0.57 |    -51% |
| benchmark_scatter / edg=(2, 100000) nod=(10, 64) sca=add           |   6.53 |  22.77 |   5.32 |    -18% |
| benchmark_scatter / edg=(2, 1000000) nod=(10, 64) sca=add          |  62.17 | 226.47 |  53.20 |    -14% |
| benchmark_scatter / edg=(2, 1000) nod=(100, 64) sca=add            |   0.62 |   0.27 |   0.07 |    -88% |
| benchmark_scatter / edg=(2, 10000) nod=(100, 64) sca=add           |   2.17 |   2.30 |   0.55 |    -74% |
| benchmark_scatter / edg=(2, 100000) nod=(100, 64) sca=add          |   5.86 |  22.67 |   5.12 |    -12% |
| benchmark_scatter / edg=(2, 1000000) nod=(100, 64) sca=add         |  61.45 | 226.39 |  47.00 |    -23% |
| benchmark_scatter / edg=(2, 1000) nod=(1000, 64) sca=add           |   0.64 |   0.28 |   0.08 |    -87% |
| benchmark_scatter / edg=(2, 10000) nod=(1000, 64) sca=add          |   2.11 |   2.32 |   0.55 |    -73% |
| benchmark_scatter / edg=(2, 100000) nod=(1000, 64) sca=add         |   5.91 |  22.74 |   4.99 |    -15% |
| benchmark_scatter / edg=(2, 1000000) nod=(1000, 64) sca=add        |  61.44 | 226.33 |  46.91 |    -23% |
| benchmark_scatter / edg=(2, 1000) nod=(10000, 64) sca=add          |   0.68 |   0.35 |   0.27 |    -59% |
| benchmark_scatter / edg=(2, 10000) nod=(10000, 64) sca=add         |   2.21 |   2.40 |   0.84 |    -61% |
| benchmark_scatter / edg=(2, 100000) nod=(10000, 64) sca=add        |   5.91 |  23.04 |   5.47 |     -7% |
| benchmark_scatter / edg=(2, 1000000) nod=(10000, 64) sca=add       |  61.50 | 227.80 |  45.47 |    -26% |
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
| benchmark_GCNConv / in_=64 out=64 edg=(2, 1000) nod=(10, 64)       |   1.15 |   0.49 |   0.21 |    -81% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 10000) nod=(10, 64)      |   3.03 |   2.82 |   1.32 |    -56% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 100000) nod=(10, 64)     |  27.38 |  26.22 |   8.31 |    -69% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 1000000) nod=(10, 64)    | 378.54 | 259.39 |  91.67 |    -75% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 1000) nod=(100, 64)      |   0.89 |   0.50 |   0.26 |    -70% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 10000) nod=(100, 64)     |   1.42 |   2.85 |   1.37 |     -3% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 100000) nod=(100, 64)    |  10.56 |  26.23 |   8.65 |    -18% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 1000000) nod=(100, 64)   | 135.10 | 259.15 |  94.99 |    -29% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 1000) nod=(1000, 64)     |   0.90 |   0.54 |   0.34 |    -61% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 10000) nod=(1000, 64)    |   2.74 |   2.86 |   1.54 |    -43% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 100000) nod=(1000, 64)   |   7.27 |  26.70 |   8.67 |    +19% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 1000000) nod=(1000, 64)  |  76.29 | 260.64 |  93.75 |    +22% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 1000) nod=(10000, 64)    |   1.05 |   0.96 |   2.01 |    +91% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 10000) nod=(10000, 64)   |   1.26 |   3.40 |   3.47 |   +175% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 100000) nod=(10000, 64)  |   7.10 |  27.62 |  10.29 |    +44% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 1000000) nod=(10000, 64) |  71.34 | 267.05 |  92.59 |    +29% |

 Average benchmark:
| Operation                  | mlx_gpu | mlx_cpu | pyg_cpu | mlx_gpu/pyg_cpu speedup |
|----------------------------|-------|-------|-------|-----------------------|
| benchmark_scatter      |  17.57 |  63.04 |  13.53 |    -22% |
| benchmark_gather       |   3.59 |   7.16 |   9.85 |   +174% |
| benchmark_GCNConv      |  45.38 |  72.96 |  26.21 |    -42% |


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
