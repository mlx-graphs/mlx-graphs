
Platform macOS-14.3.1

mlx version: 0.4.0

mlx-graphs version: 0.0.3

torch version: 2.2.0

torch_geometric version: 2.5.0

Detailed benchmark:
| Operation                                                                  | mlx_gpu | mlx_cpu | pyg_cpu | mlx_gpu/pyg_cpu speedup |
|----------------------------------------------------------------------------|-------|-------|-------|-----------------------|
| benchmark_gather / edg=(2, 1000) nod=(10, 64)                          |   0.22 |   0.03 |   0.20 |     -6% |
| benchmark_gather / edg=(2, 10000) nod=(10, 64)                         |   0.38 |   0.14 |   0.40 |     +4% |
| benchmark_gather / edg=(2, 100000) nod=(10, 64)                        |   1.39 |   1.27 |   0.65 |    -53% |
| benchmark_gather / edg=(2, 1000000) nod=(10, 64)                       |   2.65 |  12.84 |  18.65 |   +603% |
| benchmark_gather / edg=(2, 1000) nod=(100, 64)                         |   0.27 |   0.03 |   0.17 |    -38% |
| benchmark_gather / edg=(2, 10000) nod=(100, 64)                        |   0.38 |   0.14 |   0.32 |    -14% |
| benchmark_gather / edg=(2, 100000) nod=(100, 64)                       |   1.37 |   1.32 |   0.68 |    -50% |
| benchmark_gather / edg=(2, 1000000) nod=(100, 64)                      |   2.67 |  12.77 |  18.00 |   +573% |
| benchmark_gather / edg=(2, 1000) nod=(1000, 64)                        |   0.22 |   0.03 |   0.17 |    -22% |
| benchmark_gather / edg=(2, 10000) nod=(1000, 64)                       |   0.85 |   0.17 |   0.46 |    -45% |
| benchmark_gather / edg=(2, 100000) nod=(1000, 64)                      |   1.30 |   1.62 |   0.72 |    -44% |
| benchmark_gather / edg=(2, 1000000) nod=(1000, 64)                     |   2.75 |  16.18 |  19.02 |   +592% |
| benchmark_gather / edg=(2, 1000) nod=(10000, 64)                       |   0.26 |   0.03 |   0.16 |    -38% |
| benchmark_gather / edg=(2, 10000) nod=(10000, 64)                      |   0.35 |   0.18 |   0.32 |     -7% |
| benchmark_gather / edg=(2, 100000) nod=(10000, 64)                     |   1.47 |   1.76 |   0.76 |    -48% |
| benchmark_gather / edg=(2, 1000000) nod=(10000, 64)                    |   2.69 |  16.96 |  18.15 |   +574% |
| benchmark_gather_batch / edg=(2, 1000) nod=(10, 64) bat=16             |   0.42 |   0.22 |   0.38 |    -10% |
| benchmark_gather_batch / edg=(2, 10000) nod=(10, 64) bat=16            |   1.32 |   2.04 |   0.97 |    -26% |
| benchmark_gather_batch / edg=(2, 100000) nod=(10, 64) bat=16           |   4.05 |  20.65 |  30.03 |   +641% |
| benchmark_gather_batch / edg=(2, 1000000) nod=(10, 64) bat=16          |  37.82 | 214.19 | 376.74 |   +896% |
| benchmark_gather_batch / edg=(2, 1000) nod=(100, 64) bat=16            |   0.43 |   0.22 |   0.39 |     -8% |
| benchmark_gather_batch / edg=(2, 10000) nod=(100, 64) bat=16           |   1.85 |   2.05 |   1.01 |    -45% |
| benchmark_gather_batch / edg=(2, 100000) nod=(100, 64) bat=16          |   4.06 |  21.04 |  27.83 |   +586% |
| benchmark_gather_batch / edg=(2, 1000) nod=(1000, 64) bat=16           |   0.41 |   0.27 |   0.42 |     +4% |
| benchmark_gather_batch / edg=(2, 10000) nod=(1000, 64) bat=16          |   0.84 |   2.60 |   1.06 |    +27% |
| benchmark_gather_batch / edg=(2, 1000) nod=(10000, 64) bat=16          |   0.27 |   0.29 |   0.36 |    +31% |
| benchmark_gather_batch / edg=(2, 1000) nod=(10, 64) bat=128            |   0.75 |   1.66 |   0.79 |     +4% |
| benchmark_gather_batch / edg=(2, 10000) nod=(10, 64) bat=128           |   3.31 |  16.03 |  23.92 |   +623% |
| benchmark_gather_batch / edg=(2, 100000) nod=(10, 64) bat=128          |  30.33 | 167.15 | 310.21 |   +922% |
| benchmark_gather_batch / edg=(2, 1000) nod=(100, 64) bat=128           |   0.51 |   1.68 |   0.84 |    +63% |
| benchmark_gather_batch / edg=(2, 10000) nod=(100, 64) bat=128          |   3.34 |  16.58 |  21.44 |   +541% |
| benchmark_gather_batch / edg=(2, 1000) nod=(1000, 64) bat=128          |   0.55 |   2.07 |   0.86 |    +55% |
| benchmark_gather_batch / edg=(2, 1000) nod=(10, 64) bat=1024           |   2.65 |  12.97 |  19.40 |   +630% |
| benchmark_gather_batch / edg=(2, 10000) nod=(10, 64) bat=1024          |  24.34 | 136.22 | 218.83 |   +798% |
| benchmark_gather_batch / edg=(2, 1000) nod=(100, 64) bat=1024          |   2.66 |  13.56 |  17.45 |   +555% |
| benchmark_scatter / edg=(2, 1000) nod=(10, 64) sca=add                 |   0.48 |   0.22 |   0.06 |    -86% |
| benchmark_scatter / edg=(2, 10000) nod=(10, 64) sca=add                |   0.49 |   1.98 |   0.42 |    -15% |
| benchmark_scatter / edg=(2, 100000) nod=(10, 64) sca=add               |   0.63 |  19.37 |   3.60 |   +472% |
| benchmark_scatter / edg=(2, 1000000) nod=(10, 64) sca=add              |   1.09 | 194.00 |  36.23 |  +3222% |
| benchmark_scatter / edg=(2, 1000) nod=(100, 64) sca=add                |   1.25 |   0.21 |   0.06 |    -94% |
| benchmark_scatter / edg=(2, 10000) nod=(100, 64) sca=add               |   0.50 |   1.97 |   0.37 |    -26% |
| benchmark_scatter / edg=(2, 100000) nod=(100, 64) sca=add              |   0.65 |  19.37 |   2.55 |   +292% |
| benchmark_scatter / edg=(2, 1000000) nod=(100, 64) sca=add             |   1.10 | 193.74 |  25.44 |  +2208% |
| benchmark_scatter / edg=(2, 1000) nod=(1000, 64) sca=add               |   0.46 |   0.22 |   0.07 |    -84% |
| benchmark_scatter / edg=(2, 10000) nod=(1000, 64) sca=add              |   0.46 |   1.98 |   0.31 |    -33% |
| benchmark_scatter / edg=(2, 100000) nod=(1000, 64) sca=add             |   0.65 |  19.37 |   2.14 |   +227% |
| benchmark_scatter / edg=(2, 1000000) nod=(1000, 64) sca=add            |   1.22 | 193.38 |  24.16 |  +1873% |
| benchmark_scatter / edg=(2, 1000) nod=(10000, 64) sca=add              |   0.46 |   0.30 |   0.20 |    -56% |
| benchmark_scatter / edg=(2, 10000) nod=(10000, 64) sca=add             |   0.56 |   2.04 |   0.49 |    -11% |
| benchmark_scatter / edg=(2, 100000) nod=(10000, 64) sca=add            |   0.51 |  19.47 |   2.79 |   +450% |
| benchmark_scatter / edg=(2, 1000000) nod=(10000, 64) sca=add           |   1.14 | 193.69 |  23.23 |  +1932% |
| benchmark_scatter / edg=(2, 1000) nod=(10, 64) sca=max                 |   1.51 |   0.21 |   0.18 |    -87% |
| benchmark_scatter / edg=(2, 10000) nod=(10, 64) sca=max                |   0.47 |   1.96 |   0.76 |    +60% |
| benchmark_scatter / edg=(2, 100000) nod=(10, 64) sca=max               |   0.63 |  19.35 |   6.39 |   +919% |
| benchmark_scatter / edg=(2, 1000000) nod=(10, 64) sca=max              |   1.20 | 193.45 |  59.07 |  +4819% |
| benchmark_scatter / edg=(2, 1000) nod=(100, 64) sca=max                |   0.49 |   0.22 |   0.17 |    -65% |
| benchmark_scatter / edg=(2, 10000) nod=(100, 64) sca=max               |   0.49 |   1.97 |   0.66 |    +36% |
| benchmark_scatter / edg=(2, 100000) nod=(100, 64) sca=max              |   0.64 |  19.35 |   5.10 |   +692% |
| benchmark_scatter / edg=(2, 1000000) nod=(100, 64) sca=max             |   1.12 | 193.37 |  46.94 |  +4089% |
| benchmark_scatter / edg=(2, 1000) nod=(1000, 64) sca=max               |   0.47 |   0.23 |   0.18 |    -61% |
| benchmark_scatter / edg=(2, 10000) nod=(1000, 64) sca=max              |   0.52 |   1.98 |   0.66 |    +27% |
| benchmark_scatter / edg=(2, 100000) nod=(1000, 64) sca=max             |   0.67 |  19.32 |   4.95 |   +638% |
| benchmark_scatter / edg=(2, 1000000) nod=(1000, 64) sca=max            |   1.07 | 193.08 |  44.63 |  +4067% |
| benchmark_scatter / edg=(2, 1000) nod=(10000, 64) sca=max              |   0.51 |   0.29 |   0.34 |    -33% |
| benchmark_scatter / edg=(2, 10000) nod=(10000, 64) sca=max             |   0.54 |   2.04 |   0.83 |    +54% |
| benchmark_scatter / edg=(2, 100000) nod=(10000, 64) sca=max            |   0.71 |  19.40 |   4.92 |   +590% |
| benchmark_scatter / edg=(2, 1000000) nod=(10000, 64) sca=max           |   1.19 | 193.88 |  46.12 |  +3766% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(10, 64) sca=add bat=16    |   0.44 |   3.15 |   0.59 |    +32% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(10, 64) sca=add bat=16   |   0.70 |  31.01 |   5.91 |   +743% |
| benchmark_scatter_batch / edg=(2, 100000) nod=(10, 64) sca=add bat=16  |   1.55 | 310.68 |  60.06 |  +3785% |
| benchmark_scatter_batch / edg=(2, 1000000) nod=(10, 64) sca=add bat=16 |  11.06 | 3117.94 | 613.96 |  +5452% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(100, 64) sca=add bat=16   |   0.53 |   3.14 |   0.49 |     -9% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(100, 64) sca=add bat=16  |   0.74 |  31.06 |   4.02 |   +444% |
| benchmark_scatter_batch / edg=(2, 100000) nod=(100, 64) sca=add bat=16 |   1.83 | 310.60 |  42.20 |  +2210% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(1000, 64) sca=add bat=16  |   0.51 |   3.15 |   0.51 |     +0% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(1000, 64) sca=add bat=16 |   0.68 |  31.03 |   3.74 |   +454% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(10000, 64) sca=add bat=16 |   0.34 |   3.28 |   0.64 |    +89% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(10, 64) sca=add bat=128   |   0.56 |  24.88 |   4.65 |   +732% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(10, 64) sca=add bat=128  |   1.88 | 248.88 |  46.75 |  +2393% |
| benchmark_scatter_batch / edg=(2, 100000) nod=(10, 64) sca=add bat=128 |   8.86 | 2488.31 | 494.03 |  +5473% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(100, 64) sca=add bat=128  |   0.63 |  24.83 |   3.29 |   +419% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(100, 64) sca=add bat=128 |   1.68 | 248.31 |  32.36 |  +1826% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(1000, 64) sca=add bat=128 |   0.43 |  24.81 |   2.93 |   +584% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(10, 64) sca=add bat=1024  |   1.02 | 198.68 |  37.00 |  +3541% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(10, 64) sca=add bat=1024 |   7.16 | 1988.61 | 388.06 |  +5318% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(100, 64) sca=add bat=1024 |   1.00 | 198.78 |  26.39 |  +2542% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(10, 64) sca=max bat=16    |   0.52 |   3.13 |   1.10 |   +110% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(10, 64) sca=max bat=16   |   0.79 |  31.03 |  10.22 |  +1200% |
| benchmark_scatter_batch / edg=(2, 100000) nod=(10, 64) sca=max bat=16  |   1.87 | 309.75 |  94.36 |  +4939% |
| benchmark_scatter_batch / edg=(2, 1000000) nod=(10, 64) sca=max bat=16 |  10.95 | 3104.30 | 921.52 |  +8316% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(100, 64) sca=max bat=16   |   0.52 |   3.14 |   0.92 |    +77% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(100, 64) sca=max bat=16  |   0.75 |  31.22 |   8.19 |   +993% |
| benchmark_scatter_batch / edg=(2, 100000) nod=(100, 64) sca=max bat=16 |   1.51 | 310.03 |  70.42 |  +4571% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(1000, 64) sca=max bat=16  |   0.53 |   3.14 |   0.92 |    +72% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(1000, 64) sca=max bat=16 |   1.06 |  30.96 |   7.88 |   +644% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(10000, 64) sca=max bat=16 |   0.50 |   3.28 |   1.19 |   +140% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(10, 64) sca=max bat=128   |   0.69 |  24.84 |   8.00 |  +1061% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(10, 64) sca=max bat=128  |   1.27 | 247.73 |  74.21 |  +5734% |
| benchmark_scatter_batch / edg=(2, 100000) nod=(10, 64) sca=max bat=128 |   8.86 | 2484.48 | 726.47 |  +8101% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(100, 64) sca=max bat=128  |   0.41 |  24.77 |   6.48 |  +1480% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(100, 64) sca=max bat=128 |   1.36 | 248.23 |  58.71 |  +4204% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(1000, 64) sca=max bat=128 |   0.47 |  24.76 |   6.24 |  +1235% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(10, 64) sca=max bat=1024  |   1.02 | 198.34 |  61.57 |  +5939% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(10, 64) sca=max bat=1024 |   7.16 | 1984.88 | 581.15 |  +8019% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(100, 64) sca=max bat=1024 |   1.02 | 198.29 |  48.24 |  +4638% |
| benchmark_GATConv / edg=(2, 1000) nod=(10, 64) in_=64 out=64           |   0.81 |   0.85 |   0.21 |    -73% |
| benchmark_GATConv / edg=(2, 10000) nod=(10, 64) in_=64 out=64          |   1.06 |   3.88 |   1.17 |     +9% |
| benchmark_GATConv / edg=(2, 100000) nod=(10, 64) in_=64 out=64         |   2.00 |  30.91 |   5.49 |   +173% |
| benchmark_GATConv / edg=(2, 1000000) nod=(10, 64) in_=64 out=64        |   4.79 | 299.35 |  59.14 |  +1135% |
| benchmark_GATConv / edg=(2, 1000) nod=(100, 64) in_=64 out=64          |   0.83 |   0.92 |   0.22 |    -73% |
| benchmark_GATConv / edg=(2, 10000) nod=(100, 64) in_=64 out=64         |   0.95 |   3.78 |   1.07 |    +13% |
| benchmark_GATConv / edg=(2, 100000) nod=(100, 64) in_=64 out=64        |   2.13 |  31.03 |   5.44 |   +155% |
| benchmark_GATConv / edg=(2, 1000000) nod=(100, 64) in_=64 out=64       |   4.71 | 300.54 |  55.65 |  +1080% |
| benchmark_GATConv / edg=(2, 1000) nod=(1000, 64) in_=64 out=64         |   0.79 |   0.96 |   0.51 |    -34% |
| benchmark_GATConv / edg=(2, 10000) nod=(1000, 64) in_=64 out=64        |   0.96 |   3.92 |   1.57 |    +63% |
| benchmark_GATConv / edg=(2, 100000) nod=(1000, 64) in_=64 out=64       |   2.12 |  31.25 |   5.45 |   +156% |
| benchmark_GATConv / edg=(2, 1000000) nod=(1000, 64) in_=64 out=64      |   4.61 | 301.49 |  53.92 |  +1069% |
| benchmark_GATConv / edg=(2, 1000) nod=(10000, 64) in_=64 out=64        |   1.12 |   1.91 |   2.50 |   +121% |
| benchmark_GATConv / edg=(2, 10000) nod=(10000, 64) in_=64 out=64       |   1.45 |   4.56 |   3.29 |   +127% |
| benchmark_GATConv / edg=(2, 100000) nod=(10000, 64) in_=64 out=64      |   2.55 |  32.03 |   6.75 |   +164% |
| benchmark_GATConv / edg=(2, 1000000) nod=(10000, 64) in_=64 out=64     |   4.77 | 303.05 |  55.20 |  +1056% |
| benchmark_GCNConv / edg=(2, 1000) nod=(10, 64) in_=64 out=64           |   0.70 |   0.56 |   0.16 |    -76% |
| benchmark_GCNConv / edg=(2, 10000) nod=(10, 64) in_=64 out=64          |   0.83 |   2.52 |   1.03 |    +24% |
| benchmark_GCNConv / edg=(2, 100000) nod=(10, 64) in_=64 out=64         |   1.95 |  22.06 |   4.99 |   +156% |
| benchmark_GCNConv / edg=(2, 1000000) nod=(10, 64) in_=64 out=64        |   4.68 | 216.99 |  53.46 |  +1041% |
| benchmark_GCNConv / edg=(2, 1000) nod=(100, 64) in_=64 out=64          |   0.75 |   0.56 |   0.16 |    -78% |
| benchmark_GCNConv / edg=(2, 10000) nod=(100, 64) in_=64 out=64         |   0.78 |   2.53 |   0.96 |    +22% |
| benchmark_GCNConv / edg=(2, 100000) nod=(100, 64) in_=64 out=64        |   1.93 |  22.05 |   4.15 |   +115% |
| benchmark_GCNConv / edg=(2, 1000000) nod=(100, 64) in_=64 out=64       |   4.47 | 217.43 |  49.33 |  +1004% |
| benchmark_GCNConv / edg=(2, 1000) nod=(1000, 64) in_=64 out=64         |   0.69 |   0.60 |   0.22 |    -68% |
| benchmark_GCNConv / edg=(2, 10000) nod=(1000, 64) in_=64 out=64        |   0.81 |   2.57 |   0.95 |    +17% |
| benchmark_GCNConv / edg=(2, 100000) nod=(1000, 64) in_=64 out=64       |   1.86 |  22.18 |   4.14 |   +122% |
| benchmark_GCNConv / edg=(2, 1000000) nod=(1000, 64) in_=64 out=64      |   4.39 | 218.74 |  47.37 |   +979% |
| benchmark_GCNConv / edg=(2, 1000) nod=(10000, 64) in_=64 out=64        |   0.76 |   0.80 |   1.83 |   +139% |
| benchmark_GCNConv / edg=(2, 10000) nod=(10000, 64) in_=64 out=64       |   1.04 |   2.88 |   2.61 |   +151% |
| benchmark_GCNConv / edg=(2, 100000) nod=(10000, 64) in_=64 out=64      |   2.04 |  22.72 |   5.43 |   +166% |
| benchmark_GCNConv / edg=(2, 1000000) nod=(10000, 64) in_=64 out=64     |   4.39 | 219.31 |  48.31 |  +1000% |

Average benchmark:
| Operation                    | mlx_gpu | mlx_cpu | pyg_cpu | mlx_gpu/pyg_cpu speedup |
|------------------------------|-------|-------|-------|-----------------------|
| benchmark_gather         |   1.20 |   4.09 |   4.93 |   +310% |
| benchmark_gather_batch   |   6.31 |  33.24 |  55.42 |   +778% |
| benchmark_scatter        |   0.75 |  53.79 |  10.75 |  +1339% |
| benchmark_scatter_batch  |   2.18 | 488.35 | 117.25 |  +5279% |
| benchmark_GATConv        |   2.23 |  84.40 |  16.10 |   +622% |
| benchmark_GCNConv        |   2.00 |  60.91 |  14.07 |   +602% |