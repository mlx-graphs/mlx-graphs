Platform macOS-14.2.1

mlx version: 0.4.0

mlx-graphs version: 0.0.3

torch version: 2.2.0

torch_geometric version: 2.5.0

Detailed benchmark:
| Operation                                                                  | mlx_gpu | mlx_cpu | pyg_cpu | mlx_gpu/pyg_cpu speedup |
|----------------------------------------------------------------------------|-------|-------|-------|-----------------------|
| benchmark_gather / edg=(2, 1000) nod=(10, 64)                          |   0.37 |   0.07 |   0.18 |    -52% |
| benchmark_gather / edg=(2, 10000) nod=(10, 64)                         |   0.52 |   0.49 |   1.08 |   +109% |
| benchmark_gather / edg=(2, 100000) nod=(10, 64)                        |   3.22 |   1.54 |   1.63 |    -49% |
| benchmark_gather / edg=(2, 1000000) nod=(10, 64)                       |   6.96 |  14.66 |  25.44 |   +265% |
| benchmark_gather / edg=(2, 1000) nod=(100, 64)                         |   0.21 |   0.07 |   0.19 |    -12% |
| benchmark_gather / edg=(2, 10000) nod=(100, 64)                        |   0.81 |   0.49 |   0.97 |    +19% |
| benchmark_gather / edg=(2, 100000) nod=(100, 64)                       |   3.13 |   2.10 |   1.63 |    -47% |
| benchmark_gather / edg=(2, 1000000) nod=(100, 64)                      |   7.32 |  14.89 |  25.55 |   +248% |
| benchmark_gather / edg=(2, 1000) nod=(1000, 64)                        |   0.46 |   0.08 |   0.20 |    -55% |
| benchmark_gather / edg=(2, 10000) nod=(1000, 64)                       |   0.81 |   0.57 |   1.27 |    +56% |
| benchmark_gather / edg=(2, 100000) nod=(1000, 64)                      |   3.18 |   1.78 |   1.67 |    -47% |
| benchmark_gather / edg=(2, 1000000) nod=(1000, 64)                     |   7.04 |  17.33 |  26.16 |   +271% |
| benchmark_gather / edg=(2, 1000) nod=(10000, 64)                       |   0.45 |   0.08 |   0.19 |    -56% |
| benchmark_gather / edg=(2, 10000) nod=(10000, 64)                      |   0.61 |   0.71 |   1.11 |    +82% |
| benchmark_gather / edg=(2, 100000) nod=(10000, 64)                     |   1.53 |   1.79 |   1.75 |    +13% |
| benchmark_gather / edg=(2, 1000000) nod=(10000, 64)                    |   7.77 |  18.76 |  27.15 |   +249% |
| benchmark_gather_batch / edg=(2, 1000) nod=(10, 64) bat=16             |   0.91 |   0.76 |   0.41 |    -55% |
| benchmark_gather_batch / edg=(2, 10000) nod=(10, 64) bat=16            |   1.39 |   2.44 |   2.52 |    +81% |
| benchmark_gather_batch / edg=(2, 100000) nod=(10, 64) bat=16           |  11.98 |  24.27 |  34.67 |   +189% |
| benchmark_gather_batch / edg=(2, 1000000) nod=(10, 64) bat=16          | 113.89 | 239.73 | 311.96 |   +173% |
| benchmark_gather_batch / edg=(2, 1000) nod=(100, 64) bat=16            |   0.67 |   0.26 |   0.38 |    -43% |
| benchmark_gather_batch / edg=(2, 10000) nod=(100, 64) bat=16           |   1.60 |   2.44 |   2.49 |    +55% |
| benchmark_gather_batch / edg=(2, 100000) nod=(100, 64) bat=16          |  10.97 |  24.07 |  33.49 |   +205% |
| benchmark_gather_batch / edg=(2, 1000) nod=(1000, 64) bat=16           |   0.39 |   0.89 |   0.39 |      0% |
| benchmark_gather_batch / edg=(2, 10000) nod=(1000, 64) bat=16          |   1.94 |   2.96 |   2.63 |    +35% |
| benchmark_gather_batch / edg=(2, 1000) nod=(10000, 64) bat=16          |   0.28 |   0.32 |   0.41 |    +47% |
| benchmark_gather_batch / edg=(2, 1000) nod=(10, 64) bat=128            |   1.60 |   1.93 |   2.01 |    +25% |
| benchmark_gather_batch / edg=(2, 10000) nod=(10, 64) bat=128           |   8.75 |  19.08 |  28.35 |   +224% |
| benchmark_gather_batch / edg=(2, 100000) nod=(10, 64) bat=128          |  84.94 | 195.74 | 235.25 |   +176% |
| benchmark_gather_batch / edg=(2, 1000) nod=(100, 64) bat=128           |   1.51 |   1.93 |   2.05 |    +36% |
| benchmark_gather_batch / edg=(2, 10000) nod=(100, 64) bat=128          |   8.85 |  19.00 |  27.26 |   +207% |
| benchmark_gather_batch / edg=(2, 1000) nod=(1000, 64) bat=128          |   1.11 |   2.27 |   2.13 |    +91% |
| benchmark_gather_batch / edg=(2, 1000) nod=(10, 64) bat=1024           |   7.03 |  15.33 |  25.81 |   +267% |
| benchmark_gather_batch / edg=(2, 10000) nod=(10, 64) bat=1024          |  68.02 | 153.04 | 187.85 |   +176% |
| benchmark_gather_batch / edg=(2, 1000) nod=(100, 64) bat=1024          |   7.10 |  15.25 |  26.00 |   +266% |
| benchmark_scatter / edg=(2, 1000) nod=(10, 64) sca=add                 |   0.34 |   0.54 |   0.12 |    -64% |
| benchmark_scatter / edg=(2, 10000) nod=(10, 64) sca=add                |   0.84 |   3.57 |   1.00 |    +18% |
| benchmark_scatter / edg=(2, 100000) nod=(10, 64) sca=add               |   0.78 |  19.41 |   3.22 |   +311% |
| benchmark_scatter / edg=(2, 1000000) nod=(10, 64) sca=add              |   2.14 | 194.07 |  34.07 |  +1492% |
| benchmark_scatter / edg=(2, 1000) nod=(100, 64) sca=add                |   0.31 |   0.54 |   0.13 |    -57% |
| benchmark_scatter / edg=(2, 10000) nod=(100, 64) sca=add               |   0.40 |   4.92 |   1.19 |   +200% |
| benchmark_scatter / edg=(2, 100000) nod=(100, 64) sca=add              |   0.78 |  19.46 |   2.62 |   +234% |
| benchmark_scatter / edg=(2, 1000000) nod=(100, 64) sca=add             |   2.12 | 193.48 |  30.56 |  +1341% |
| benchmark_scatter / edg=(2, 1000) nod=(1000, 64) sca=add               |   0.34 |   0.56 |   0.14 |    -58% |
| benchmark_scatter / edg=(2, 10000) nod=(1000, 64) sca=add              |   0.39 |   5.01 |   1.22 |   +211% |
| benchmark_scatter / edg=(2, 100000) nod=(1000, 64) sca=add             |   0.92 |  19.43 |   2.68 |   +192% |
| benchmark_scatter / edg=(2, 1000000) nod=(1000, 64) sca=add            |   2.13 | 193.71 |  27.80 |  +1204% |
| benchmark_scatter / edg=(2, 1000) nod=(10000, 64) sca=add              |   0.46 |   0.74 |   0.32 |    -29% |
| benchmark_scatter / edg=(2, 10000) nod=(10000, 64) sca=add             |   0.39 |   5.25 |   0.40 |     +4% |
| benchmark_scatter / edg=(2, 100000) nod=(10000, 64) sca=add            |   0.89 |  19.55 |   2.84 |   +218% |
| benchmark_scatter / edg=(2, 1000000) nod=(10000, 64) sca=add           |   2.29 | 194.37 |  29.26 |  +1178% |
| benchmark_scatter / edg=(2, 1000) nod=(10, 64) sca=max                 |   0.35 |   0.56 |   0.29 |    -15% |
| benchmark_scatter / edg=(2, 10000) nod=(10, 64) sca=max                |   0.36 |   4.98 |   1.37 |   +281% |
| benchmark_scatter / edg=(2, 100000) nod=(10, 64) sca=max               |   0.77 |  19.38 |   6.71 |   +777% |
| benchmark_scatter / edg=(2, 1000000) nod=(10, 64) sca=max              |   2.15 | 193.87 |  65.95 |  +2970% |
| benchmark_scatter / edg=(2, 1000) nod=(100, 64) sca=max                |   0.34 |   0.54 |   0.37 |     +7% |
| benchmark_scatter / edg=(2, 10000) nod=(100, 64) sca=max               |   0.39 |   5.04 |   2.59 |   +560% |
| benchmark_scatter / edg=(2, 100000) nod=(100, 64) sca=max              |   1.18 |  19.36 |   6.35 |   +439% |
| benchmark_scatter / edg=(2, 1000000) nod=(100, 64) sca=max             |   2.16 | 193.77 |  64.68 |  +2900% |
| benchmark_scatter / edg=(2, 1000) nod=(1000, 64) sca=max               |   0.35 |   0.57 |   0.31 |    -10% |
| benchmark_scatter / edg=(2, 10000) nod=(1000, 64) sca=max              |   0.41 |   3.93 |   0.76 |    +82% |
| benchmark_scatter / edg=(2, 100000) nod=(1000, 64) sca=max             |   0.87 |  19.37 |   6.26 |   +616% |
| benchmark_scatter / edg=(2, 1000000) nod=(1000, 64) sca=max            |   2.44 | 193.35 |  63.87 |  +2518% |
| benchmark_scatter / edg=(2, 1000) nod=(10000, 64) sca=max              |   0.46 |   0.75 |   0.54 |    +17% |
| benchmark_scatter / edg=(2, 10000) nod=(10000, 64) sca=max             |   0.52 |   4.50 |   0.79 |    +52% |
| benchmark_scatter / edg=(2, 100000) nod=(10000, 64) sca=max            |   1.08 |  19.49 |   6.22 |   +474% |
| benchmark_scatter / edg=(2, 1000000) nod=(10000, 64) sca=max           |   2.34 | 193.87 |  57.90 |  +2369% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(10, 64) sca=add bat=16    |   0.41 |   3.22 |   0.57 |    +38% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(10, 64) sca=add bat=16   |   0.59 |  31.06 |   5.36 |   +811% |
| benchmark_scatter_batch / edg=(2, 100000) nod=(10, 64) sca=add bat=16  |   3.24 | 310.14 |  56.34 |  +1638% |
| benchmark_scatter_batch / edg=(2, 1000000) nod=(10, 64) sca=add bat=16 |  29.09 | 3105.28 | 630.64 |  +2068% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(100, 64) sca=add bat=16   |   0.40 |   3.17 |   0.45 |    +11% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(100, 64) sca=add bat=16  |   1.04 |  31.08 |   4.41 |   +324% |
| benchmark_scatter_batch / edg=(2, 100000) nod=(100, 64) sca=add bat=16 |   3.41 | 310.35 |  47.79 |  +1299% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(1000, 64) sca=add bat=16  |   0.43 |   3.16 |   0.44 |     +3% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(1000, 64) sca=add bat=16 |   0.60 |  31.08 |   4.33 |   +619% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(10000, 64) sca=add bat=16 |   0.37 |   3.30 |   0.52 |    +41% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(10, 64) sca=add bat=128   |   0.57 |  24.84 |   3.84 |   +577% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(10, 64) sca=add bat=128  |   2.67 | 248.33 |  45.37 |  +1599% |
| benchmark_scatter_batch / edg=(2, 100000) nod=(10, 64) sca=add bat=128 |  23.34 | 2483.90 | 501.49 |  +2048% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(100, 64) sca=add bat=128  |   0.91 |  24.82 |   3.67 |   +302% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(100, 64) sca=add bat=128 |   2.71 | 248.28 |  38.85 |  +1332% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(1000, 64) sca=add bat=128 |   0.54 |  24.84 |   3.62 |   +564% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(10, 64) sca=add bat=1024  |   2.14 | 198.49 |  31.35 |  +1367% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(10, 64) sca=add bat=1024 |  18.79 | 1984.97 | 400.29 |  +2030% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(100, 64) sca=add bat=1024 |   2.22 | 198.46 |  32.52 |  +1361% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(10, 64) sca=max bat=16    |   0.35 |   3.14 |   1.14 |   +230% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(10, 64) sca=max bat=16   |   1.09 |  31.05 |  10.79 |   +890% |
| benchmark_scatter_batch / edg=(2, 100000) nod=(10, 64) sca=max bat=16  |   3.31 | 310.10 | 108.29 |  +3176% |
| benchmark_scatter_batch / edg=(2, 1000000) nod=(10, 64) sca=max bat=16 |  29.16 | 3102.74 | 1178.11 |  +3939% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(100, 64) sca=max bat=16   |   0.35 |   3.13 |   1.11 |   +215% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(100, 64) sca=max bat=16  |   1.04 |  30.99 |  10.68 |   +928% |
| benchmark_scatter_batch / edg=(2, 100000) nod=(100, 64) sca=max bat=16 |   3.28 | 309.60 | 100.54 |  +2962% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(1000, 64) sca=max bat=16  |   0.47 |   3.14 |   1.10 |   +135% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(1000, 64) sca=max bat=16 |   1.14 |  30.93 |  10.48 |   +819% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(10000, 64) sca=max bat=16 |   0.42 |   3.27 |   1.11 |   +161% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(10, 64) sca=max bat=128   |   0.86 |  24.78 |   8.62 |   +901% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(10, 64) sca=max bat=128  |   2.67 | 247.96 |  89.14 |  +3234% |
| benchmark_scatter_batch / edg=(2, 100000) nod=(10, 64) sca=max bat=128 |  23.29 | 2479.58 | 945.39 |  +3958% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(100, 64) sca=max bat=128  |   0.89 |  24.82 |   8.12 |   +816% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(100, 64) sca=max bat=128 |   2.71 | 247.99 |  84.24 |  +3010% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(1000, 64) sca=max bat=128 |   0.55 |  24.78 |   8.25 |  +1393% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(10, 64) sca=max bat=1024  |   2.08 | 198.48 |  68.95 |  +3207% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(10, 64) sca=max bat=1024 |  18.55 | 1982.35 | 756.74 |  +3979% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(100, 64) sca=max bat=1024 |   2.21 | 198.41 |  64.67 |  +2830% |
| benchmark_GATConv / edg=(2, 1000) nod=(10, 64) in_=64 out=64           |   1.06 |   1.49 |   0.18 |    -82% |
| benchmark_GATConv / edg=(2, 10000) nod=(10, 64) in_=64 out=64          |   1.48 |   3.73 |   0.96 |    -35% |
| benchmark_GATConv / edg=(2, 100000) nod=(10, 64) in_=64 out=64         |   3.72 |  30.77 |   6.42 |    +72% |
| benchmark_GATConv / edg=(2, 1000000) nod=(10, 64) in_=64 out=64        |  10.89 | 304.33 |  74.45 |   +583% |
| benchmark_GATConv / edg=(2, 1000) nod=(100, 64) in_=64 out=64          |   0.84 |   1.36 |   0.54 |    -35% |
| benchmark_GATConv / edg=(2, 10000) nod=(100, 64) in_=64 out=64         |   1.20 |   3.68 |   1.02 |    -15% |
| benchmark_GATConv / edg=(2, 100000) nod=(100, 64) in_=64 out=64        |   3.87 |  30.98 |   6.57 |    +69% |
| benchmark_GATConv / edg=(2, 1000000) nod=(100, 64) in_=64 out=64       |  10.75 | 301.28 |  71.31 |   +563% |
| benchmark_GATConv / edg=(2, 1000) nod=(1000, 64) in_=64 out=64         |   0.88 |   1.92 |   0.38 |    -56% |
| benchmark_GATConv / edg=(2, 10000) nod=(1000, 64) in_=64 out=64        |   1.45 |   3.86 |   1.27 |    -12% |
| benchmark_GATConv / edg=(2, 100000) nod=(1000, 64) in_=64 out=64       |   4.25 |  31.02 |   6.59 |    +55% |
| benchmark_GATConv / edg=(2, 1000000) nod=(1000, 64) in_=64 out=64      |  10.43 | 309.30 |  72.01 |   +590% |
| benchmark_GATConv / edg=(2, 1000) nod=(10000, 64) in_=64 out=64        |   2.09 |   1.47 |   1.99 |     -4% |
| benchmark_GATConv / edg=(2, 10000) nod=(10000, 64) in_=64 out=64       |   2.21 |   4.56 |   2.52 |    +13% |
| benchmark_GATConv / edg=(2, 100000) nod=(10000, 64) in_=64 out=64      |   2.47 |  32.37 |   9.94 |   +302% |
| benchmark_GATConv / edg=(2, 1000000) nod=(10000, 64) in_=64 out=64     |  10.98 | 306.73 |  70.65 |   +543% |
| benchmark_GCNConv / edg=(2, 1000) nod=(10, 64) in_=64 out=64           |   0.70 |   0.86 |   0.36 |    -48% |
| benchmark_GCNConv / edg=(2, 10000) nod=(10, 64) in_=64 out=64          |   1.25 |   2.41 |   0.76 |    -39% |
| benchmark_GCNConv / edg=(2, 100000) nod=(10, 64) in_=64 out=64         |   1.80 |  22.03 |   4.82 |   +168% |
| benchmark_GCNConv / edg=(2, 1000000) nod=(10, 64) in_=64 out=64        |  10.03 | 218.59 |  63.20 |   +530% |
| benchmark_GCNConv / edg=(2, 1000) nod=(100, 64) in_=64 out=64          |   0.70 |   0.89 |   0.42 |    -39% |
| benchmark_GCNConv / edg=(2, 10000) nod=(100, 64) in_=64 out=64         |   0.98 |   5.57 |   0.80 |    -18% |
| benchmark_GCNConv / edg=(2, 100000) nod=(100, 64) in_=64 out=64        |   1.82 |  22.09 |   5.25 |   +188% |
| benchmark_GCNConv / edg=(2, 1000000) nod=(100, 64) in_=64 out=64       |   9.81 | 219.16 |  63.81 |   +550% |
| benchmark_GCNConv / edg=(2, 1000) nod=(1000, 64) in_=64 out=64         |   1.00 |   1.12 |   0.21 |    -78% |
| benchmark_GCNConv / edg=(2, 10000) nod=(1000, 64) in_=64 out=64        |   1.31 |   2.39 |   1.15 |    -11% |
| benchmark_GCNConv / edg=(2, 100000) nod=(1000, 64) in_=64 out=64       |   3.75 |  22.19 |   5.49 |    +46% |
| benchmark_GCNConv / edg=(2, 1000000) nod=(1000, 64) in_=64 out=64      |  10.19 | 220.14 |  64.50 |   +532% |
| benchmark_GCNConv / edg=(2, 1000) nod=(10000, 64) in_=64 out=64        |   0.74 |   0.65 |   1.21 |    +62% |
| benchmark_GCNConv / edg=(2, 10000) nod=(10000, 64) in_=64 out=64       |   1.06 |   2.82 |   1.97 |    +86% |
| benchmark_GCNConv / edg=(2, 100000) nod=(10000, 64) in_=64 out=64      |   1.98 |  25.14 |   6.54 |   +231% |
| benchmark_GCNConv / edg=(2, 1000000) nod=(10000, 64) in_=64 out=64     |  10.73 | 226.82 |  64.58 |   +501% |

Average benchmark:
| Operation                    | mlx_gpu | mlx_cpu | pyg_cpu | mlx_gpu/pyg_cpu speedup |
|------------------------------|-------|-------|-------|-----------------------|
| benchmark_gather         |   2.77 |   4.71 |   7.26 |   +161% |
| benchmark_gather_batch   |  17.52 |  37.98 |  48.74 |   +178% |
| benchmark_scatter        |   0.99 |  54.62 |  13.20 |  +1233% |
| benchmark_scatter_batch  |   4.94 | 487.53 | 138.67 |  +2704% |
| benchmark_GATConv        |   4.29 |  85.55 |  20.43 |   +376% |
