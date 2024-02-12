# Results before int64 speed improvement on scatter ([Pull request](https://github.com/ml-explore/mlx/pull/662))

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

# Results after int64 speed improvement on scatter

**M1 Pro (2E+8P+16GPU)**

Detailed benchmark:
| Operation                                                              | mlx_gpu | mlx_cpu | pyg_mps | pyg_cpu | mlx_gpu/pyg_cpu speedup | mlx_gpu/pyg_mps speedup |
|------------------------------------------------------------------------|-------|-------|-------|-------|-----------------------|-----------------------|
| benchmark_scatter / edg=(2, 1000) nod=(10, 64) sca=add             |   0.71 |   0.27 |   2.97 |   0.05 |    -92% |   +319% |
| benchmark_scatter / edg=(2, 10000) nod=(10, 64) sca=add            |   2.58 |   2.30 |  54.59 |   0.56 |    -78% |  +2013% |
| benchmark_scatter / edg=(2, 100000) nod=(10, 64) sca=add           |  11.35 |  22.67 | 1747.41 |   5.75 |    -49% | +15293% |
| benchmark_scatter / edg=(2, 1000000) nod=(10, 64) sca=add          | 116.20 | 226.83 | 16359.14 |  54.84 |    -52% | +13978% |
| benchmark_scatter / edg=(2, 1000) nod=(100, 64) sca=add            |   0.46 |   0.27 |   1.29 |   0.08 |    -83% |   +179% |
| benchmark_scatter / edg=(2, 10000) nod=(100, 64) sca=add           |   1.15 |   2.34 |   4.18 |   0.57 |    -50% |   +263% |
| benchmark_scatter / edg=(2, 100000) nod=(100, 64) sca=add          |   5.12 |  22.77 | 126.34 |   5.07 |      0% |  +2369% |
| benchmark_scatter / edg=(2, 1000000) nod=(100, 64) sca=add         |  13.37 | 226.61 | 1652.02 |  47.80 |   +257% | +12255% |
| benchmark_scatter / edg=(2, 1000) nod=(1000, 64) sca=add           |   0.45 |   0.28 |   0.87 |   0.07 |    -84% |    +94% |
| benchmark_scatter / edg=(2, 10000) nod=(1000, 64) sca=add          |   0.91 |   2.36 |   2.63 |   0.60 |    -33% |   +190% |
| benchmark_scatter / edg=(2, 100000) nod=(1000, 64) sca=add         |   1.77 |  23.52 |  15.41 |   5.04 |   +184% |   +769% |
| benchmark_scatter / edg=(2, 1000000) nod=(1000, 64) sca=add        |  13.29 | 226.84 | 167.11 |  48.08 |   +261% |  +1157% |
| benchmark_scatter / edg=(2, 1000) nod=(10000, 64) sca=add          |   0.50 |   0.38 |   0.84 |   0.20 |    -60% |    +69% |
| benchmark_scatter / edg=(2, 10000) nod=(10000, 64) sca=add         |   1.23 |   2.43 |   2.22 |   0.71 |    -41% |    +80% |
| benchmark_scatter / edg=(2, 100000) nod=(10000, 64) sca=add        |   2.65 |  23.12 |   7.23 |   5.20 |    +96% |   +172% |
| benchmark_scatter / edg=(2, 1000000) nod=(10000, 64) sca=add       |  13.25 | 230.33 |  65.46 |  45.09 |   +240% |   +393% |
| benchmark_gather / edg=(2, 1000) nod=(10, 64)                      |   0.43 |   0.12 |   0.33 |   0.35 |    -19% |    -22% |
| benchmark_gather / edg=(2, 10000) nod=(10, 64)                     |   0.70 |   0.22 |   0.80 |   0.37 |    -47% |    +14% |
| benchmark_gather / edg=(2, 100000) nod=(10, 64)                    |   4.24 |   1.92 |   5.14 |   2.65 |    -37% |    +21% |
| benchmark_gather / edg=(2, 1000000) nod=(10, 64)                   |  11.69 |  19.09 |  51.64 |  34.92 |   +198% |   +341% |
| benchmark_gather / edg=(2, 1000) nod=(100, 64)                     |   0.30 |   0.04 |   0.39 |   0.16 |    -46% |    +30% |
| benchmark_gather / edg=(2, 10000) nod=(100, 64)                    |   0.88 |   0.22 |   0.80 |   0.34 |    -61% |     -9% |
| benchmark_gather / edg=(2, 100000) nod=(100, 64)                   |   3.24 |   1.92 |   5.22 |   2.50 |    -22% |    +60% |
| benchmark_gather / edg=(2, 1000000) nod=(100, 64)                  |  11.82 |  19.17 |  50.55 |  35.30 |   +198% |   +327% |
| benchmark_gather / edg=(2, 1000) nod=(1000, 64)                    |   0.47 |   0.18 |   0.52 |   0.45 |     -3% |     +8% |
| benchmark_gather / edg=(2, 10000) nod=(1000, 64)                   |   1.04 |   0.61 |   0.81 |   0.35 |    -65% |    -21% |
| benchmark_gather / edg=(2, 100000) nod=(1000, 64)                  |   1.97 |   2.16 |   5.30 |   2.69 |    +36% |   +169% |
| benchmark_gather / edg=(2, 1000000) nod=(1000, 64)                 |  11.72 |  21.47 |  53.33 |  34.75 |   +196% |   +355% |
| benchmark_gather / edg=(2, 1000) nod=(10000, 64)                   |   0.43 |   0.27 |   0.32 |   0.50 |    +14% |    -25% |
| benchmark_gather / edg=(2, 10000) nod=(10000, 64)                  |   0.61 |   0.35 |   0.77 |   0.37 |    -38% |    +26% |
| benchmark_gather / edg=(2, 100000) nod=(10000, 64)                 |   2.20 |   4.28 |   5.36 |   3.09 |    +40% |   +143% |
| benchmark_gather / edg=(2, 1000000) nod=(10000, 64)                |  11.82 |  68.53 |  53.64 |  35.62 |   +201% |   +353% |
| benchmark_fast_gather / edg=[2, 1000] nod=[10, 64]                 |   0.48 |   0.14 |   0.36 |   0.44 |     -6% |    -25% |
| benchmark_fast_gather / edg=[2, 10000] nod=[10, 64]                |   1.12 |   0.64 |   0.80 |   1.57 |    +39% |    -28% |
| benchmark_fast_gather / edg=[2, 100000] nod=[10, 64]               |   2.03 |   1.77 |   5.13 |   2.48 |    +22% |   +152% |
| benchmark_fast_gather / edg=[2, 1000000] nod=[10, 64]              |  11.78 |  11.73 |  53.34 |  34.67 |   +194% |   +352% |
| benchmark_fast_gather / edg=[2, 1000] nod=[100, 64]                |   0.44 |   0.21 |   0.33 |   0.22 |    -49% |    -24% |
| benchmark_fast_gather / edg=[2, 10000] nod=[100, 64]               |   1.39 |   0.87 |   0.79 |   0.45 |    -67% |    -43% |
| benchmark_fast_gather / edg=[2, 100000] nod=[100, 64]              |   2.17 |   1.38 |   5.22 |   2.51 |    +16% |   +140% |
| benchmark_fast_gather / edg=[2, 1000000] nod=[100, 64]             |  11.81 |  11.82 |  52.68 |  34.70 |   +193% |   +345% |
| benchmark_fast_gather / edg=[2, 1000] nod=[1000, 64]               |   0.25 |   0.05 |   0.34 |   0.11 |    -57% |    +34% |
| benchmark_fast_gather / edg=[2, 10000] nod=[1000, 64]              |   1.93 |   0.27 |   0.84 |   0.46 |    -76% |    -56% |
| benchmark_fast_gather / edg=[2, 100000] nod=[1000, 64]             |   1.72 |   1.44 |   5.39 |   2.70 |    +57% |   +213% |
| benchmark_fast_gather / edg=[2, 1000000] nod=[1000, 64]            |  11.90 |  11.73 |  53.28 |  34.64 |   +191% |   +347% |
| benchmark_fast_gather / edg=[2, 1000] nod=[10000, 64]              |   0.60 |   0.29 |   0.33 |   0.16 |    -73% |    -44% |
| benchmark_fast_gather / edg=[2, 10000] nod=[10000, 64]             |   3.82 |   0.36 |   0.81 |   0.56 |    -85% |    -78% |
| benchmark_fast_gather / edg=[2, 100000] nod=[10000, 64]            |   4.16 |   1.82 |   5.21 |   2.91 |    -29% |    +25% |
| benchmark_fast_gather / edg=[2, 1000000] nod=[10000, 64]           |  11.67 |  11.70 |  51.28 |  35.50 |   +204% |   +339% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 1000) nod=(10, 64)       |   1.71 |   1.55 |   4.07 |   0.19 |    -89% |   +137% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 10000) nod=(10, 64)      |   3.37 |   2.90 |  52.48 |   1.19 |    -64% |  +1455% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 100000) nod=(10, 64)     |  36.56 |  26.20 | 888.87 |   8.52 |    -76% |  +2330% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 1000000) nod=(10, 64)    | 444.28 | 260.73 | 6634.62 |  91.62 |    -79% |  +1393% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 1000) nod=(100, 64)      |   0.75 |   0.47 |   2.34 |   0.25 |    -67% |   +212% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 10000) nod=(100, 64)     |   1.62 |   2.81 |   5.03 |   1.23 |    -23% |   +210% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 100000) nod=(100, 64)    |   9.20 |  26.12 | 136.77 |   8.54 |     -7% |  +1386% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 1000000) nod=(100, 64)   |  84.81 | 257.95 | 1194.25 |  96.79 |    +14% |  +1308% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 1000) nod=(1000, 64)     |   1.49 |   0.53 |   2.42 |   0.38 |    -74% |    +62% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 10000) nod=(1000, 64)    |   1.52 |   2.85 |   3.09 |   1.63 |     +6% |   +102% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 100000) nod=(1000, 64)   |   5.97 |  26.54 |  18.15 |   8.73 |    +46% |   +203% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 1000000) nod=(1000, 64)  |  26.39 | 261.37 | 175.17 |  94.59 |   +258% |   +563% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 1000) nod=(10000, 64)    |   1.59 |   0.92 |   3.90 |   1.86 |    +17% |   +145% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 10000) nod=(10000, 64)   |   1.71 |   3.39 |   4.29 |   3.01 |    +76% |   +151% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 100000) nod=(10000, 64)  |   4.10 |  29.01 |  10.14 |  10.29 |   +150% |   +147% |
| benchmark_GCNConv / in_=64 out=64 edg=(2, 1000000) nod=(10000, 64) |  23.24 | 271.31 |  74.44 |  94.20 |   +305% |   +220% |
| benchmark_GATConv / in_=64 out=64 edg=(2, 1000) nod=(10, 64)       |   1.16 |   0.64 | nan |   0.24 |    -79% | nan |
| benchmark_GATConv / in_=64 out=64 edg=(2, 10000) nod=(10, 64)      |   4.08 |   3.88 | nan |   1.68 |    -58% | nan |
| benchmark_GATConv / in_=64 out=64 edg=(2, 100000) nod=(10, 64)     |  37.35 |  37.07 | nan |   9.41 |    -74% | nan |
| benchmark_GATConv / in_=64 out=64 edg=(2, 1000000) nod=(10, 64)    | 467.45 | 355.52 | nan | 100.05 |    -78% | nan |
| benchmark_GATConv / in_=64 out=64 edg=(2, 1000) nod=(100, 64)      |   1.18 |   0.86 | nan |   0.49 |    -58% | nan |
| benchmark_GATConv / in_=64 out=64 edg=(2, 10000) nod=(100, 64)     |   3.14 |   3.82 | nan |   1.65 |    -47% | nan |
| benchmark_GATConv / in_=64 out=64 edg=(2, 100000) nod=(100, 64)    |   7.85 |  36.36 | nan |   9.95 |    +26% | nan |
| benchmark_GATConv / in_=64 out=64 edg=(2, 1000000) nod=(100, 64)   |  93.35 | 357.46 | nan | 101.30 |     +8% | nan |
| benchmark_GATConv / in_=64 out=64 edg=(2, 1000) nod=(1000, 64)     |   2.00 |   0.75 | nan |   0.61 |    -69% | nan |
| benchmark_GATConv / in_=64 out=64 edg=(2, 10000) nod=(1000, 64)    |   3.01 |   3.94 | nan |   1.91 |    -36% | nan |
| benchmark_GATConv / in_=64 out=64 edg=(2, 100000) nod=(1000, 64)   |   3.85 |  36.69 | nan |   9.98 |   +159% | nan |
| benchmark_GATConv / in_=64 out=64 edg=(2, 1000000) nod=(1000, 64)  |  32.33 | 361.82 | nan |  98.96 |   +206% | nan |
| benchmark_GATConv / in_=64 out=64 edg=(2, 1000) nod=(10000, 64)    |   2.44 |   1.68 | nan |   2.97 |    +21% | nan |
| benchmark_GATConv / in_=64 out=64 edg=(2, 10000) nod=(10000, 64)   |   3.82 |   4.92 | nan |   4.12 |     +7% | nan |
| benchmark_GATConv / in_=64 out=64 edg=(2, 100000) nod=(10000, 64)  |   5.14 |  38.05 | nan |  11.80 |   +129% | nan |
| benchmark_GATConv / in_=64 out=64 edg=(2, 1000000) nod=(10000, 64) |  29.43 | 404.52 | nan | 107.16 |   +264% | nan |

 Average benchmark:
| Operation                  | mlx_gpu | mlx_cpu | pyg_mps | pyg_cpu | mlx_gpu/pyg_cpu speedup | mlx_gpu/pyg_mps speedup |
|----------------------------|-------|-------|-------|-------|-----------------------|-----------------------|
| benchmark_scatter      |  11.56 |  63.33 | 1263.11 |  13.73 |    +18% | +10825% |
| benchmark_gather       |   3.97 |   8.78 |  14.68 |   9.65 |   +142% |   +269% |
| benchmark_fast_gather  |   4.21 |   3.51 |  14.76 |   9.63 |   +129% |   +250% |
| benchmark_GCNConv      |  40.52 |  73.41 | 575.63 |  26.44 |    -34% |  +1320% |
| benchmark_GATConv      |  43.60 | 103.00 | nan |  28.89 |    -33% | nan |
