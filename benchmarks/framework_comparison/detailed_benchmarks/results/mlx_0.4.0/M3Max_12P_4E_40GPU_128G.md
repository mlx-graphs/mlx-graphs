
Platform macOS-14.3.1

mlx version: 0.4.0

mlx-graphs version: 0.0.3

torch version: 2.2.0

torch_geometric version: 2.5.0

Detailed benchmark:
| Operation                                                                  | mlx_gpu | mlx_cpu | pyg_cpu | mlx_gpu/pyg_cpu speedup |
|----------------------------------------------------------------------------|-------|-------|-------|-----------------------|
| benchmark_gather / edg=(2, 1000) nod=(10, 64)                          |   0.23 |   0.04 |   0.25 |     +8% |
| benchmark_gather / edg=(2, 10000) nod=(10, 64)                         |   0.78 |   0.16 |   0.40 |    -48% |
| benchmark_gather / edg=(2, 100000) nod=(10, 64)                        |   1.65 |   1.41 |   0.79 |    -52% |
| benchmark_gather / edg=(2, 1000000) nod=(10, 64)                       |   4.29 |  13.88 |  18.26 |   +325% |
| benchmark_gather / edg=(2, 1000) nod=(100, 64)                         |   0.22 |   0.04 |   0.30 |    +32% |
| benchmark_gather / edg=(2, 10000) nod=(100, 64)                        |   0.78 |   0.15 |   0.42 |    -45% |
| benchmark_gather / edg=(2, 100000) nod=(100, 64)                       |   1.66 |   1.38 |   0.84 |    -49% |
| benchmark_gather / edg=(2, 1000000) nod=(100, 64)                      |   4.18 |  14.17 |  17.83 |   +326% |
| benchmark_gather / edg=(2, 1000) nod=(1000, 64)                        |   0.45 |   0.04 |   0.30 |    -33% |
| benchmark_gather / edg=(2, 10000) nod=(1000, 64)                       |   0.35 |   0.19 |   0.38 |     +9% |
| benchmark_gather / edg=(2, 100000) nod=(1000, 64)                      |   1.67 |   1.80 |   0.85 |    -49% |
| benchmark_gather / edg=(2, 1000000) nod=(1000, 64)                     |   4.23 |  17.96 |  17.81 |   +320% |
| benchmark_gather / edg=(2, 1000) nod=(10000, 64)                       |   0.66 |   0.05 |   0.26 |    -61% |
| benchmark_gather / edg=(2, 10000) nod=(10000, 64)                      |   0.79 |   0.23 |   0.42 |    -46% |
| benchmark_gather / edg=(2, 100000) nod=(10000, 64)                     |   1.17 |   1.92 |   0.93 |    -20% |
| benchmark_gather / edg=(2, 1000000) nod=(10000, 64)                    |   4.27 |  19.75 |  16.82 |   +293% |
| benchmark_gather_batch / edg=(2, 1000) nod=(10, 64) bat=16             |   0.84 |   0.24 |   0.45 |    -46% |
| benchmark_gather_batch / edg=(2, 10000) nod=(10, 64) bat=16            |   2.23 |   2.16 |   1.15 |    -48% |
| benchmark_gather_batch / edg=(2, 100000) nod=(10, 64) bat=16           |   4.04 |  22.50 |  28.21 |   +597% |
| benchmark_gather_batch / edg=(2, 1000000) nod=(10, 64) bat=16          |  37.86 | 222.44 | 351.86 |   +829% |
| benchmark_gather_batch / edg=(2, 1000) nod=(100, 64) bat=16            |   0.85 |   0.23 |   0.43 |    -49% |
| benchmark_gather_batch / edg=(2, 10000) nod=(100, 64) bat=16           |   2.22 |   2.17 |   1.18 |    -46% |
| benchmark_gather_batch / edg=(2, 100000) nod=(100, 64) bat=16          |   4.09 |  22.32 |  29.02 |   +609% |
| benchmark_gather_batch / edg=(2, 1000) nod=(1000, 64) bat=16           |   0.86 |   0.31 |   0.45 |    -47% |
| benchmark_gather_batch / edg=(2, 10000) nod=(1000, 64) bat=16          |   2.23 |   2.82 |   1.21 |    -45% |
| benchmark_gather_batch / edg=(2, 1000) nod=(10000, 64) bat=16          |   0.28 |   0.32 |   0.49 |    +74% |
| benchmark_gather_batch / edg=(2, 1000) nod=(10, 64) bat=128            |   0.75 |   1.75 |   0.97 |    +30% |
| benchmark_gather_batch / edg=(2, 10000) nod=(10, 64) bat=128           |   4.70 |  17.99 |  21.04 |   +347% |
| benchmark_gather_batch / edg=(2, 100000) nod=(10, 64) bat=128          |  30.35 | 180.31 | 289.83 |   +854% |
| benchmark_gather_batch / edg=(2, 1000) nod=(100, 64) bat=128           |   0.66 |   1.75 |   0.97 |    +45% |
| benchmark_gather_batch / edg=(2, 10000) nod=(100, 64) bat=128          |   5.01 |  18.06 |  23.70 |   +373% |
| benchmark_gather_batch / edg=(2, 1000) nod=(1000, 64) bat=128          |   0.54 |   2.27 |   1.14 |   +112% |
| benchmark_gather_batch / edg=(2, 1000) nod=(10, 64) bat=1024           |   2.65 |  14.28 |  18.24 |   +588% |
| benchmark_gather_batch / edg=(2, 10000) nod=(10, 64) bat=1024          |  24.37 | 146.87 | 235.82 |   +867% |
| benchmark_gather_batch / edg=(2, 1000) nod=(100, 64) bat=1024          |   2.68 |  14.04 |  17.87 |   +565% |
| benchmark_scatter / edg=(2, 1000) nod=(10, 64) sca=add                 |   1.30 |   0.26 |   0.06 |    -95% |
| benchmark_scatter / edg=(2, 10000) nod=(10, 64) sca=add                |   1.32 |   2.12 |   0.46 |    -65% |
| benchmark_scatter / edg=(2, 100000) nod=(10, 64) sca=add               |   1.44 |  20.60 |   3.79 |   +162% |
| benchmark_scatter / edg=(2, 1000000) nod=(10, 64) sca=add              |   2.34 | 205.36 |  38.08 |  +1525% |
| benchmark_scatter / edg=(2, 1000) nod=(100, 64) sca=add                |   1.30 |   0.27 |   0.07 |    -94% |
| benchmark_scatter / edg=(2, 10000) nod=(100, 64) sca=add               |   0.44 |   2.14 |   0.45 |     +2% |
| benchmark_scatter / edg=(2, 100000) nod=(100, 64) sca=add              |   1.45 |  20.67 |   3.02 |   +108% |
| benchmark_scatter / edg=(2, 1000000) nod=(100, 64) sca=add             |   1.30 | 205.72 |  27.80 |  +2040% |
| benchmark_scatter / edg=(2, 1000) nod=(1000, 64) sca=add               |   1.26 |   0.27 |   0.09 |    -93% |
| benchmark_scatter / edg=(2, 10000) nod=(1000, 64) sca=add              |   0.44 |   2.12 |   0.44 |     +0% |
| benchmark_scatter / edg=(2, 100000) nod=(1000, 64) sca=add             |   1.30 |  20.62 |   3.04 |   +133% |
| benchmark_scatter / edg=(2, 1000000) nod=(1000, 64) sca=add            |   2.70 | 204.73 |  25.09 |   +829% |
| benchmark_scatter / edg=(2, 1000) nod=(10000, 64) sca=add              |   1.35 |   0.34 |   0.28 |    -79% |
| benchmark_scatter / edg=(2, 10000) nod=(10000, 64) sca=add             |   0.70 |   2.24 |   0.56 |    -19% |
| benchmark_scatter / edg=(2, 100000) nod=(10000, 64) sca=add            |   1.49 |  20.99 |   3.19 |   +113% |
| benchmark_scatter / edg=(2, 1000000) nod=(10000, 64) sca=add           |   2.33 | 208.56 |  24.27 |   +941% |
| benchmark_scatter / edg=(2, 1000) nod=(10, 64) sca=max                 |   0.44 |   0.26 |   0.18 |    -58% |
| benchmark_scatter / edg=(2, 10000) nod=(10, 64) sca=max                |   0.70 |   2.10 |   1.00 |    +44% |
| benchmark_scatter / edg=(2, 100000) nod=(10, 64) sca=max               |   0.60 |  20.65 |   7.25 |  +1118% |
| benchmark_scatter / edg=(2, 1000000) nod=(10, 64) sca=max              |   2.45 | 205.81 |  64.92 |  +2546% |
| benchmark_scatter / edg=(2, 1000) nod=(100, 64) sca=max                |   1.16 |   0.26 |   0.20 |    -82% |
| benchmark_scatter / edg=(2, 10000) nod=(100, 64) sca=max               |   1.13 |   2.15 |   0.87 |    -22% |
| benchmark_scatter / edg=(2, 100000) nod=(100, 64) sca=max              |   1.46 |  20.74 |   7.00 |   +379% |
| benchmark_scatter / edg=(2, 1000000) nod=(100, 64) sca=max             |   2.56 | 205.54 |  50.08 |  +1852% |
| benchmark_scatter / edg=(2, 1000) nod=(1000, 64) sca=max               |   1.29 |   0.27 |   0.20 |    -84% |
| benchmark_scatter / edg=(2, 10000) nod=(1000, 64) sca=max              |   1.33 |   2.12 |   0.86 |    -35% |
| benchmark_scatter / edg=(2, 100000) nod=(1000, 64) sca=max             |   1.48 |  20.59 |   5.70 |   +284% |
| benchmark_scatter / edg=(2, 1000000) nod=(1000, 64) sca=max            |   2.45 | 205.11 |  50.31 |  +1955% |
| benchmark_scatter / edg=(2, 1000) nod=(10000, 64) sca=max              |   0.47 |   0.34 |   0.39 |    -18% |
| benchmark_scatter / edg=(2, 10000) nod=(10000, 64) sca=max             |   1.37 |   2.27 |   1.21 |    -11% |
| benchmark_scatter / edg=(2, 100000) nod=(10000, 64) sca=max            |   0.93 |  21.33 |   6.32 |   +578% |
| benchmark_scatter / edg=(2, 1000000) nod=(10000, 64) sca=max           |   2.66 | 207.59 |  47.05 |  +1666% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(10, 64) sca=add bat=16    |   1.32 |   3.33 |   0.78 |    -40% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(10, 64) sca=add bat=16   |   1.52 |  33.10 |   6.66 |   +336% |
| benchmark_scatter_batch / edg=(2, 100000) nod=(10, 64) sca=add bat=16  |   3.34 | 331.28 |  62.14 |  +1758% |
| benchmark_scatter_batch / edg=(2, 1000000) nod=(10, 64) sca=add bat=16 |  11.42 | 3297.23 | 632.17 |  +5435% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(100, 64) sca=add bat=16   |   1.14 |   3.36 |   0.58 |    -48% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(100, 64) sca=add bat=16  |   0.67 |  33.04 |   4.81 |   +616% |
| benchmark_scatter_batch / edg=(2, 100000) nod=(100, 64) sca=add bat=16 |   2.08 | 330.27 |  44.18 |  +2019% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(1000, 64) sca=add bat=16  |   1.33 |   3.35 |   0.71 |    -46% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(1000, 64) sca=add bat=16 |   0.71 |  32.90 |   5.07 |   +613% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(10000, 64) sca=add bat=16 |   0.43 |   3.55 |   0.78 |    +82% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(10, 64) sca=add bat=128   |   0.45 |  26.37 |   5.10 |  +1039% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(10, 64) sca=add bat=128  |   1.30 | 263.66 |  50.13 |  +3742% |
| benchmark_scatter_batch / edg=(2, 100000) nod=(10, 64) sca=add bat=128 |   9.18 | 2636.81 | 508.40 |  +5440% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(100, 64) sca=add bat=128  |   0.51 |  26.50 |   3.57 |   +596% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(100, 64) sca=add bat=128 |   1.46 | 263.96 |  36.58 |  +2410% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(1000, 64) sca=add bat=128 |   0.42 |  26.37 |   4.15 |   +888% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(10, 64) sca=add bat=1024  |   1.05 | 210.21 |  40.86 |  +3794% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(10, 64) sca=add bat=1024 |   7.43 | 2109.65 | 406.73 |  +5373% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(100, 64) sca=add bat=1024 |   1.05 | 211.61 |  31.50 |  +2893% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(10, 64) sca=max bat=16    |   0.48 |   3.37 |   1.08 |   +124% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(10, 64) sca=max bat=16   |   0.69 |  33.20 |  11.29 |  +1542% |
| benchmark_scatter_batch / edg=(2, 100000) nod=(10, 64) sca=max bat=16  |   2.14 | 329.14 |  97.28 |  +4449% |
| benchmark_scatter_batch / edg=(2, 1000000) nod=(10, 64) sca=max bat=16 |  11.48 | 3293.28 | 956.89 |  +8233% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(100, 64) sca=max bat=16   |   0.52 |   3.34 |   1.24 |   +138% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(100, 64) sca=max bat=16  |   0.67 |  32.97 |  10.21 |  +1420% |
| benchmark_scatter_batch / edg=(2, 100000) nod=(100, 64) sca=max bat=16 |   1.74 | 330.46 |  77.64 |  +4374% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(1000, 64) sca=max bat=16  |   0.50 |   3.40 |   1.09 |   +116% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(1000, 64) sca=max bat=16 |   1.11 |  32.91 |   9.84 |   +788% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(10000, 64) sca=max bat=16 |   0.39 |   3.58 |   1.48 |   +276% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(10, 64) sca=max bat=128   |   0.48 |  26.48 |   9.06 |  +1784% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(10, 64) sca=max bat=128  |   1.29 | 264.05 |  78.65 |  +6002% |
| benchmark_scatter_batch / edg=(2, 100000) nod=(10, 64) sca=max bat=128 |   9.26 | 2636.32 | 760.70 |  +8116% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(100, 64) sca=max bat=128  |   0.47 |  26.53 |   7.61 |  +1510% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(100, 64) sca=max bat=128 |   1.30 | 264.18 |  64.72 |  +4879% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(1000, 64) sca=max bat=128 |   0.44 |  26.41 |   7.67 |  +1653% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(10, 64) sca=max bat=1024  |   1.06 | 211.01 |  63.67 |  +5887% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(10, 64) sca=max bat=1024 |   7.53 | 2111.70 | 605.43 |  +7942% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(100, 64) sca=max bat=1024 |   1.05 | 211.05 |  52.96 |  +4930% |
| benchmark_GATConv / edg=(2, 1000) nod=(10, 64) in_=64 out=64           |   1.09 |   1.08 |   0.24 |    -78% |
| benchmark_GATConv / edg=(2, 10000) nod=(10, 64) in_=64 out=64          |   1.21 |   4.09 |   1.34 |    +10% |
| benchmark_GATConv / edg=(2, 100000) nod=(10, 64) in_=64 out=64         |   2.39 |  32.84 |   6.35 |   +165% |
| benchmark_GATConv / edg=(2, 1000000) nod=(10, 64) in_=64 out=64        |   4.73 | 318.36 |  62.35 |  +1217% |
| benchmark_GATConv / edg=(2, 1000) nod=(100, 64) in_=64 out=64          |   1.12 |   1.09 |   0.29 |    -74% |
| benchmark_GATConv / edg=(2, 10000) nod=(100, 64) in_=64 out=64         |   1.18 |   4.06 |   1.33 |    +12% |
| benchmark_GATConv / edg=(2, 100000) nod=(100, 64) in_=64 out=64        |   2.15 |  32.84 |   5.82 |   +170% |
| benchmark_GATConv / edg=(2, 1000000) nod=(100, 64) in_=64 out=64       |   4.63 | 317.73 |  59.71 |  +1190% |
| benchmark_GATConv / edg=(2, 1000) nod=(1000, 64) in_=64 out=64         |   1.11 |   1.15 |   0.67 |    -39% |
| benchmark_GATConv / edg=(2, 10000) nod=(1000, 64) in_=64 out=64        |   1.27 |   4.06 |   1.62 |    +27% |
| benchmark_GATConv / edg=(2, 100000) nod=(1000, 64) in_=64 out=64       |   2.10 |  33.19 |   6.33 |   +201% |
| benchmark_GATConv / edg=(2, 1000000) nod=(1000, 64) in_=64 out=64      |   4.78 | 319.75 |  56.97 |  +1091% |
| benchmark_GATConv / edg=(2, 1000) nod=(10000, 64) in_=64 out=64        |   1.39 |   2.05 |   2.53 |    +81% |
| benchmark_GATConv / edg=(2, 10000) nod=(10000, 64) in_=64 out=64       |   1.58 |   4.85 |   3.76 |   +138% |
| benchmark_GATConv / edg=(2, 100000) nod=(10000, 64) in_=64 out=64      |   2.35 |  34.57 |   7.47 |   +218% |
| benchmark_GATConv / edg=(2, 1000000) nod=(10000, 64) in_=64 out=64     |   4.59 | 325.53 |  58.23 |  +1168% |
| benchmark_GCNConv / edg=(2, 1000) nod=(10, 64) in_=64 out=64           |   0.99 |   0.58 |   0.22 |    -78% |
| benchmark_GCNConv / edg=(2, 10000) nod=(10, 64) in_=64 out=64          |   1.10 |   2.69 |   1.13 |     +2% |
| benchmark_GCNConv / edg=(2, 100000) nod=(10, 64) in_=64 out=64         |   2.11 |  23.43 |   5.64 |   +167% |
| benchmark_GCNConv / edg=(2, 1000000) nod=(10, 64) in_=64 out=64        |   4.35 | 230.13 |  55.48 |  +1174% |
| benchmark_GCNConv / edg=(2, 1000) nod=(100, 64) in_=64 out=64          |   1.02 |   0.61 |   0.22 |    -78% |
| benchmark_GCNConv / edg=(2, 10000) nod=(100, 64) in_=64 out=64         |   1.11 |   2.68 |   1.18 |     +5% |
| benchmark_GCNConv / edg=(2, 100000) nod=(100, 64) in_=64 out=64        |   2.13 |  23.43 |   5.13 |   +140% |
| benchmark_GCNConv / edg=(2, 1000000) nod=(100, 64) in_=64 out=64       |   4.31 | 230.01 |  51.44 |  +1093% |
| benchmark_GCNConv / edg=(2, 1000) nod=(1000, 64) in_=64 out=64         |   0.69 |   0.65 |   0.29 |    -57% |
| benchmark_GCNConv / edg=(2, 10000) nod=(1000, 64) in_=64 out=64        |   0.79 |   2.74 |   1.35 |    +71% |
| benchmark_GCNConv / edg=(2, 100000) nod=(1000, 64) in_=64 out=64       |   1.69 |  23.72 |   4.82 |   +184% |
| benchmark_GCNConv / edg=(2, 1000000) nod=(1000, 64) in_=64 out=64      |   4.82 | 231.97 |  49.43 |   +925% |
| benchmark_GCNConv / edg=(2, 1000) nod=(10000, 64) in_=64 out=64        |   0.78 |   1.08 |   1.75 |   +123% |
| benchmark_GCNConv / edg=(2, 10000) nod=(10000, 64) in_=64 out=64       |   1.24 |   3.14 |   2.71 |   +118% |
| benchmark_GCNConv / edg=(2, 100000) nod=(10000, 64) in_=64 out=64      |   1.52 |  24.50 |   6.25 |   +312% |
| benchmark_GCNConv / edg=(2, 1000000) nod=(10000, 64) in_=64 out=64     |   4.43 | 235.83 |  48.09 |   +985% |

Average benchmark:
| Operation                    | mlx_gpu | mlx_cpu | pyg_cpu | mlx_gpu/pyg_cpu speedup |
|------------------------------|-------|-------|-------|-----------------------|
| benchmark_gather         |   1.71 |   4.57 |   4.80 |   +180% |
| benchmark_gather_batch   |   6.70 |  35.41 |  53.90 |   +704% |
| benchmark_scatter        |   1.40 |  57.32 |  11.69 |   +732% |
| benchmark_scatter_batch  |   2.35 | 518.16 | 122.72 |  +5114% |
| benchmark_GATConv        |   2.35 |  89.83 |  17.19 |   +629% |
| benchmark_GCNConv        |   2.07 |  64.82 |  14.70 |   +610% |
