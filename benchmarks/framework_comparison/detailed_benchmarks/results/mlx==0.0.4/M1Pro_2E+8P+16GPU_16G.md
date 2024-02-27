Platform macOS-14.2

mlx version: 0.4.0

mlx-graphs version: 0.0.2

torch version: 2.2.1

torch_geometric version: 2.5.0

Detailed benchmark:
| Operation                                                                  | mlx_gpu | mlx_cpu | pyg_cpu | mlx_gpu/pyg_cpu speedup |
|----------------------------------------------------------------------------|-------|-------|-------|-----------------------|
| benchmark_gather / edg=(2, 1000) nod=(10, 64)                          |   0.28 |   0.04 |   0.03 |    -89% |
| benchmark_gather / edg=(2, 10000) nod=(10, 64)                         |   0.63 |   0.23 |   0.35 |    -44% |
| benchmark_gather / edg=(2, 100000) nod=(10, 64)                        |   1.05 |   1.92 |   2.66 |   +151% |
| benchmark_gather / edg=(2, 1000000) nod=(10, 64)                       |   8.03 |  18.98 |  35.68 |   +344% |
| benchmark_gather / edg=(2, 1000) nod=(100, 64)                         |   0.36 |   0.12 |   0.56 |    +53% |
| benchmark_gather / edg=(2, 10000) nod=(100, 64)                        |   0.65 |   0.48 |   0.58 |    -11% |
| benchmark_gather / edg=(2, 100000) nod=(100, 64)                       |   1.37 |   1.92 |   2.68 |    +95% |
| benchmark_gather / edg=(2, 1000000) nod=(100, 64)                      |   8.23 |  19.04 |  34.91 |   +324% |
| benchmark_gather / edg=(2, 1000) nod=(1000, 64)                        |   0.45 |   0.14 |   0.05 |    -89% |
| benchmark_gather / edg=(2, 10000) nod=(1000, 64)                       |   0.69 |   1.10 |   0.42 |    -38% |
| benchmark_gather / edg=(2, 100000) nod=(1000, 64)                      |   1.54 |   2.14 |   2.76 |    +79% |
| benchmark_gather / edg=(2, 1000000) nod=(1000, 64)                     |   8.59 |  21.29 |  34.55 |   +302% |
| benchmark_gather / edg=(2, 1000) nod=(10000, 64)                       |   0.38 |   0.08 |   0.13 |    -65% |
| benchmark_gather / edg=(2, 10000) nod=(10000, 64)                      |   0.44 |   0.45 |   0.60 |    +36% |
| benchmark_gather / edg=(2, 100000) nod=(10000, 64)                     |   1.48 |   4.04 |   3.64 |   +145% |
| benchmark_gather / edg=(2, 1000000) nod=(10000, 64)                    |   8.72 |  54.44 |  35.76 |   +310% |
| benchmark_gather_batch / edg=(2, 1000) nod=(10, 64) bat=16             |   0.77 |   1.16 |   0.46 |    -40% |
| benchmark_gather_batch / edg=(2, 10000) nod=(10, 64) bat=16            |   1.95 |   3.06 |   4.18 |   +114% |
| benchmark_gather_batch / edg=(2, 100000) nod=(10, 64) bat=16           |  12.71 |  30.73 |  42.09 |   +231% |
| benchmark_gather_batch / edg=(2, 1000000) nod=(10, 64) bat=16          | 123.84 | 307.65 | 386.54 |   +212% |
| benchmark_gather_batch / edg=(2, 1000) nod=(100, 64) bat=16            |   0.74 |   0.33 |   0.49 |    -33% |
| benchmark_gather_batch / edg=(2, 10000) nod=(100, 64) bat=16           |   1.55 |   3.08 |   4.25 |   +174% |
| benchmark_gather_batch / edg=(2, 100000) nod=(100, 64) bat=16          |  13.03 |  31.09 |  43.34 |   +232% |
| benchmark_gather_batch / edg=(2, 1000) nod=(1000, 64) bat=16           |   0.81 |   0.37 |   0.65 |    -19% |
| benchmark_gather_batch / edg=(2, 10000) nod=(1000, 64) bat=16          |   2.90 |   3.40 |   4.34 |    +49% |
| benchmark_gather_batch / edg=(2, 1000) nod=(10000, 64) bat=16          |   0.38 |   1.50 |   0.60 |    +58% |
| benchmark_gather_batch / edg=(2, 1000) nod=(10, 64) bat=128            |   1.55 |   2.49 |   3.24 |   +108% |
| benchmark_gather_batch / edg=(2, 10000) nod=(10, 64) bat=128           |  10.24 |  24.40 |  37.44 |   +265% |
| benchmark_gather_batch / edg=(2, 100000) nod=(10, 64) bat=128          |  99.78 | 246.76 | 307.67 |   +208% |
| benchmark_gather_batch / edg=(2, 1000) nod=(100, 64) bat=128           |   1.59 |   2.46 |   3.32 |   +108% |
| benchmark_gather_batch / edg=(2, 10000) nod=(100, 64) bat=128          |  10.43 |  24.38 |  38.09 |   +265% |
| benchmark_gather_batch / edg=(2, 1000) nod=(1000, 64) bat=128          |   1.33 |   2.72 |   3.37 |   +153% |
| benchmark_gather_batch / edg=(2, 1000) nod=(10, 64) bat=1024           |   8.19 |  19.63 |  35.64 |   +335% |
| benchmark_gather_batch / edg=(2, 10000) nod=(10, 64) bat=1024          |  79.40 | 195.57 | 247.16 |   +211% |
| benchmark_gather_batch / edg=(2, 1000) nod=(100, 64) bat=1024          |   8.39 |  19.49 |  35.54 |   +323% |
| benchmark_scatter / edg=(2, 1000) nod=(10, 64) sca=add                 |   0.83 |   0.90 |   0.47 |    -43% |
| benchmark_scatter / edg=(2, 10000) nod=(10, 64) sca=add                |   2.83 |   2.31 |   0.57 |    -79% |
| benchmark_scatter / edg=(2, 100000) nod=(10, 64) sca=add               |  11.98 |  22.68 |   5.66 |    -52% |
| benchmark_scatter / edg=(2, 1000000) nod=(10, 64) sca=add              | 123.15 | 227.02 |  54.29 |    -55% |
| benchmark_scatter / edg=(2, 1000) nod=(100, 64) sca=add                |   0.68 |   0.93 |   0.20 |    -69% |
| benchmark_scatter / edg=(2, 10000) nod=(100, 64) sca=add               |   0.95 |   2.31 |   0.58 |    -38% |
| benchmark_scatter / edg=(2, 100000) nod=(100, 64) sca=add              |   3.03 |  22.66 |   4.91 |    +62% |
| benchmark_scatter / edg=(2, 1000000) nod=(100, 64) sca=add             |  11.86 | 226.49 |  47.73 |   +302% |
| benchmark_scatter / edg=(2, 1000) nod=(1000, 64) sca=add               |   0.68 |   0.94 |   0.24 |    -64% |
| benchmark_scatter / edg=(2, 10000) nod=(1000, 64) sca=add              |   0.82 |   2.31 |   0.62 |    -25% |
| benchmark_scatter / edg=(2, 100000) nod=(1000, 64) sca=add             |   1.66 |  22.72 |   4.95 |   +197% |
| benchmark_scatter / edg=(2, 1000000) nod=(1000, 64) sca=add            |   4.76 | 226.71 |  45.28 |   +850% |
| benchmark_scatter / edg=(2, 1000) nod=(10000, 64) sca=add              |   0.74 |   0.35 |   0.20 |    -73% |
| benchmark_scatter / edg=(2, 10000) nod=(10000, 64) sca=add             |   0.68 |   2.42 |   1.11 |    +63% |
| benchmark_scatter / edg=(2, 100000) nod=(10000, 64) sca=add            |   1.63 |  23.11 |   5.06 |   +210% |
| benchmark_scatter / edg=(2, 1000000) nod=(10000, 64) sca=add           |   4.07 | 228.18 |  44.90 |  +1002% |
| benchmark_scatter / edg=(2, 1000) nod=(10, 64) sca=max                 |   0.65 |   0.91 |   0.38 |    -40% |
| benchmark_scatter / edg=(2, 10000) nod=(10, 64) sca=max                |   0.58 |   2.31 |   1.23 |   +111% |
| benchmark_scatter / edg=(2, 100000) nod=(10, 64) sca=max               |   0.94 |  22.72 |  10.49 |  +1017% |
| benchmark_scatter / edg=(2, 1000000) nod=(10, 64) sca=max              |   2.39 | 227.59 |  88.55 |  +3607% |
| benchmark_scatter / edg=(2, 1000) nod=(100, 64) sca=max                |   0.62 |   0.96 |   0.50 |    -20% |
| benchmark_scatter / edg=(2, 10000) nod=(100, 64) sca=max               |   0.70 |   2.31 |   1.21 |    +73% |
| benchmark_scatter / edg=(2, 100000) nod=(100, 64) sca=max              |   1.00 |  22.73 |   9.56 |   +856% |
| benchmark_scatter / edg=(2, 1000000) nod=(100, 64) sca=max             |   2.11 | 227.95 |  85.83 |  +3972% |
| benchmark_scatter / edg=(2, 1000) nod=(1000, 64) sca=max               |   0.64 |   0.96 |   0.19 |    -70% |
| benchmark_scatter / edg=(2, 10000) nod=(1000, 64) sca=max              |   0.77 |   2.32 |   1.26 |    +63% |
| benchmark_scatter / edg=(2, 100000) nod=(1000, 64) sca=max             |   1.02 |  22.77 |   9.50 |   +827% |
| benchmark_scatter / edg=(2, 1000000) nod=(1000, 64) sca=max            |   2.77 | 227.98 |  82.39 |  +2873% |
| benchmark_scatter / edg=(2, 1000) nod=(10000, 64) sca=max              |   0.54 |   0.35 |   0.34 |    -37% |
| benchmark_scatter / edg=(2, 10000) nod=(10000, 64) sca=max             |   0.84 |   2.45 |   1.52 |    +79% |
| benchmark_scatter / edg=(2, 100000) nod=(10000, 64) sca=max            |   1.55 |  23.11 |   9.74 |   +529% |
| benchmark_scatter / edg=(2, 1000000) nod=(10000, 64) sca=max           |   2.59 | 228.43 |  79.84 |  +2987% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(10, 64) sca=add bat=16    |   2.37 |   3.66 |   1.02 |    -57% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(10, 64) sca=add bat=16   |  18.37 |  36.38 |   8.96 |    -51% |
| benchmark_scatter_batch / edg=(2, 100000) nod=(10, 64) sca=add bat=16  | 202.11 | 362.57 |  83.33 |    -58% |
| benchmark_scatter_batch / edg=(2, 1000000) nod=(10, 64) sca=add bat=16 | 2290.06 | 3635.57 | 844.00 |    -63% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(100, 64) sca=add bat=16   |   0.89 |   3.82 |   0.96 |     +6% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(100, 64) sca=add bat=16  |   3.73 |  36.41 |   8.02 |   +114% |
| benchmark_scatter_batch / edg=(2, 100000) nod=(100, 64) sca=add bat=16 |  18.59 | 366.92 |  74.99 |   +303% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(1000, 64) sca=add bat=16  |   0.65 |   3.68 |   1.09 |    +67% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(1000, 64) sca=add bat=16 |   1.57 |  36.42 |   7.71 |   +392% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(10000, 64) sca=add bat=16 |   0.55 |   3.93 |   1.22 |   +120% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(10, 64) sca=add bat=128   |  15.61 |  29.27 |   7.27 |    -53% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(10, 64) sca=add bat=128  | 163.39 | 295.76 |  68.26 |    -58% |
| benchmark_scatter_batch / edg=(2, 100000) nod=(10, 64) sca=add bat=128 | 1729.42 | 2975.00 | 705.27 |    -59% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(100, 64) sca=add bat=128  |   2.08 |  30.01 |   7.15 |   +243% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(100, 64) sca=add bat=128 |  14.90 | 300.02 |  66.01 |   +343% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(1000, 64) sca=add bat=128 |   1.08 |  30.04 |   6.78 |   +526% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(10, 64) sca=add bat=1024  | 126.54 | 240.66 |  58.97 |    -53% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(10, 64) sca=add bat=1024 | 1363.99 | 2361.76 | 539.40 |    -60% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(100, 64) sca=add bat=1024 |  12.10 | 237.32 |  47.82 |   +295% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(10, 64) sca=max bat=16    |   0.54 |   3.68 |   1.95 |   +260% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(10, 64) sca=max bat=16   |   1.09 |  36.93 |  15.81 |  +1345% |
| benchmark_scatter_batch / edg=(2, 100000) nod=(10, 64) sca=max bat=16  |   3.00 | 367.75 | 140.12 |  +4563% |
| benchmark_scatter_batch / edg=(2, 1000000) nod=(10, 64) sca=max bat=16 |  25.40 | 3668.15 | 1383.75 |  +5348% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(100, 64) sca=max bat=16   |   0.51 |   3.72 |   1.71 |   +238% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(100, 64) sca=max bat=16  |   1.26 |  37.60 |  14.73 |  +1067% |
| benchmark_scatter_batch / edg=(2, 100000) nod=(100, 64) sca=max bat=16 |   3.05 | 368.41 | 128.43 |  +4115% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(1000, 64) sca=max bat=16  |   0.82 |   3.75 |   1.79 |   +118% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(1000, 64) sca=max bat=16 |   1.48 |  36.95 |  14.78 |   +898% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(10000, 64) sca=max bat=16 |   0.55 |   3.96 |   2.14 |   +289% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(10, 64) sca=max bat=128   |   0.72 |  29.60 |  13.33 |  +1763% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(10, 64) sca=max bat=128  |   2.50 | 295.13 | 116.47 |  +4565% |
| benchmark_scatter_batch / edg=(2, 100000) nod=(10, 64) sca=max bat=128 |  20.96 | 2931.60 | 1121.77 |  +5250% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(100, 64) sca=max bat=128  |   0.71 |  29.59 |  12.12 |  +1606% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(100, 64) sca=max bat=128 |   2.56 | 294.86 | 103.29 |  +3937% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(1000, 64) sca=max bat=128 |   0.75 |  29.55 |  11.93 |  +1488% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(10, 64) sca=max bat=1024  |   2.04 | 236.74 |  92.57 |  +4429% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(10, 64) sca=max bat=1024 |  16.52 | 2351.67 | 905.48 |  +5379% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(100, 64) sca=max bat=1024 |   2.10 | 238.13 |  84.22 |  +3918% |
| benchmark_GATConv / edg=(2, 1000) nod=(10, 64) in_=64 out=64           |   1.94 |   0.89 |   0.24 |    -87% |
| benchmark_GATConv / edg=(2, 10000) nod=(10, 64) in_=64 out=64          |   3.41 |   4.43 |   1.33 |    -61% |
| benchmark_GATConv / edg=(2, 100000) nod=(10, 64) in_=64 out=64         |  41.56 |  36.91 |   9.93 |    -76% |
| benchmark_GATConv / edg=(2, 1000000) nod=(10, 64) in_=64 out=64        | 503.79 | 355.12 |  99.21 |    -80% |
| benchmark_GATConv / edg=(2, 1000) nod=(100, 64) in_=64 out=64          |   2.02 |   0.94 |   0.32 |    -84% |
| benchmark_GATConv / edg=(2, 10000) nod=(100, 64) in_=64 out=64         |   3.18 |   4.40 |   1.59 |    -49% |
| benchmark_GATConv / edg=(2, 100000) nod=(100, 64) in_=64 out=64        |   7.24 |  36.40 |   8.85 |    +22% |
| benchmark_GATConv / edg=(2, 1000000) nod=(100, 64) in_=64 out=64       |  62.87 | 357.17 | 104.17 |    +65% |
| benchmark_GATConv / edg=(2, 1000) nod=(1000, 64) in_=64 out=64         |   1.78 |   2.46 |   0.36 |    -79% |
| benchmark_GATConv / edg=(2, 10000) nod=(1000, 64) in_=64 out=64        |   1.08 |   4.56 |   1.95 |    +79% |
| benchmark_GATConv / edg=(2, 100000) nod=(1000, 64) in_=64 out=64       |   2.45 |  37.81 |  10.74 |   +337% |
| benchmark_GATConv / edg=(2, 1000000) nod=(1000, 64) in_=64 out=64      |  16.92 | 364.53 | 101.14 |   +497% |
| benchmark_GATConv / edg=(2, 1000) nod=(10000, 64) in_=64 out=64        |   1.81 |   2.36 |   2.52 |    +39% |
| benchmark_GATConv / edg=(2, 10000) nod=(10000, 64) in_=64 out=64       |   3.43 |   5.74 |   4.04 |    +18% |
| benchmark_GATConv / edg=(2, 100000) nod=(10000, 64) in_=64 out=64      |   2.94 |  38.63 |  12.25 |   +317% |
| benchmark_GATConv / edg=(2, 1000000) nod=(10000, 64) in_=64 out=64     |  13.38 | 379.89 | 102.69 |   +667% |
| benchmark_GCNConv / edg=(2, 1000) nod=(10, 64) in_=64 out=64           |   1.95 |   1.83 |   0.28 |    -85% |
| benchmark_GCNConv / edg=(2, 10000) nod=(10, 64) in_=64 out=64          |   4.26 |   2.87 |   1.38 |    -67% |
| benchmark_GCNConv / edg=(2, 100000) nod=(10, 64) in_=64 out=64         |  41.62 |  26.40 |   8.77 |    -78% |
| benchmark_GCNConv / edg=(2, 1000000) nod=(10, 64) in_=64 out=64        | 472.85 | 259.89 |  93.38 |    -80% |
| benchmark_GCNConv / edg=(2, 1000) nod=(100, 64) in_=64 out=64          |   1.72 |   0.58 |   0.23 |    -86% |
| benchmark_GCNConv / edg=(2, 10000) nod=(100, 64) in_=64 out=64         |   2.88 |   2.95 |   1.62 |    -43% |
| benchmark_GCNConv / edg=(2, 100000) nod=(100, 64) in_=64 out=64        |   6.77 |  26.41 |   8.62 |    +27% |
| benchmark_GCNConv / edg=(2, 1000000) nod=(100, 64) in_=64 out=64       |  85.55 | 258.19 |  92.68 |     +8% |
| benchmark_GCNConv / edg=(2, 1000) nod=(1000, 64) in_=64 out=64         |   0.79 |   0.61 |   0.22 |    -72% |
| benchmark_GCNConv / edg=(2, 10000) nod=(1000, 64) in_=64 out=64        |   1.19 |   2.94 |   1.72 |    +44% |
| benchmark_GCNConv / edg=(2, 100000) nod=(1000, 64) in_=64 out=64       |   2.67 |  26.32 |   8.53 |   +219% |
| benchmark_GCNConv / edg=(2, 1000000) nod=(1000, 64) in_=64 out=64      |  16.64 | 258.57 |  93.43 |   +461% |
| benchmark_GCNConv / edg=(2, 1000) nod=(10000, 64) in_=64 out=64        |   0.98 |   1.00 |   2.44 |   +148% |
| benchmark_GCNConv / edg=(2, 10000) nod=(10000, 64) in_=64 out=64       |   1.80 |   3.52 |   3.07 |    +70% |
| benchmark_GCNConv / edg=(2, 100000) nod=(10000, 64) in_=64 out=64      |   2.89 |  27.80 |   9.84 |   +240% |
| benchmark_GCNConv / edg=(2, 1000000) nod=(10000, 64) in_=64 out=64     |  12.53 | 267.43 |  94.23 |   +652% |

Average benchmark:
| Operation                    | mlx_gpu | mlx_cpu | pyg_cpu | mlx_gpu/pyg_cpu speedup |
|------------------------------|-------|-------|-------|-----------------------|
| benchmark_gather         |   2.68 |   7.90 |   9.71 |   +262% |
| benchmark_gather_batch   |  19.98 |  48.44 |  63.07 |   +215% |
| benchmark_scatter        |   5.94 |  63.37 |  18.73 |   +215% |
| benchmark_scatter_batch  | 159.33 | 577.82 | 176.44 |    +10% |
| benchmark_GATConv        |  41.86 | 102.02 |  28.83 |    -31% |
| benchmark_GCNConv        |  41.07 |  72.96 |  26.28 |    -36% |
