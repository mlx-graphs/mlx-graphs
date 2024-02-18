Platform macOS-14.2.1

mlx version: 0.3.0
mlx-graphs version: 0.0.2
torch version: 2.2.0
torch_geometric version: 2.5.0

Detailed benchmark:
| Operation                                                                  | mlx_gpu | mlx_cpu | pyg_cpu | mlx_gpu/pyg_cpu speedup |
|----------------------------------------------------------------------------|-------|-------|-------|-----------------------|
| benchmark_gather / edg=(2, 1000) nod=(10, 64)                          |   0.21 |   0.07 |   0.19 |    -10% |
| benchmark_gather / edg=(2, 10000) nod=(10, 64)                         |   0.82 |   0.48 |   0.81 |     -1% |
| benchmark_gather / edg=(2, 100000) nod=(10, 64)                        |   3.25 |   1.49 |   1.63 |    -49% |
| benchmark_gather / edg=(2, 1000000) nod=(10, 64)                       |   6.99 |  14.82 |  24.84 |   +255% |
| benchmark_gather / edg=(2, 1000) nod=(100, 64)                         |   0.94 |   0.07 |   0.20 |    -78% |
| benchmark_gather / edg=(2, 10000) nod=(100, 64)                        |   0.51 |   0.48 |   0.48 |     -6% |
| benchmark_gather / edg=(2, 100000) nod=(100, 64)                       |   2.19 |   1.60 |   1.63 |    -25% |
| benchmark_gather / edg=(2, 1000000) nod=(100, 64)                      |   7.21 |  14.91 |  25.62 |   +255% |
| benchmark_gather / edg=(2, 1000) nod=(1000, 64)                        |   0.17 |   0.08 |   0.15 |     -7% |
| benchmark_gather / edg=(2, 10000) nod=(1000, 64)                       |   0.52 |   0.56 |   0.99 |    +90% |
| benchmark_gather / edg=(2, 100000) nod=(1000, 64)                      |   1.57 |   1.81 |   1.71 |     +8% |
| benchmark_gather / edg=(2, 1000000) nod=(1000, 64)                     |   7.00 |  17.95 |  26.10 |   +272% |
| benchmark_gather / edg=(2, 1000) nod=(10000, 64)                       |   0.22 |   0.08 |   0.17 |    -22% |
| benchmark_gather / edg=(2, 10000) nod=(10000, 64)                      |   0.59 |   0.73 |   0.30 |    -49% |
| benchmark_gather / edg=(2, 100000) nod=(10000, 64)                     |   1.19 |   1.90 |   1.75 |    +46% |
| benchmark_gather / edg=(2, 1000000) nod=(10000, 64)                    |   7.32 |  18.48 |  26.64 |   +263% |
| benchmark_gather_batch / edg=(2, 1000) nod=(10, 64) bat=16             |   0.68 |   0.76 |   1.25 |    +83% |
| benchmark_gather_batch / edg=(2, 10000) nod=(10, 64) bat=16            |   1.83 |   2.40 |   2.57 |    +40% |
| benchmark_gather_batch / edg=(2, 100000) nod=(10, 64) bat=16           |  10.88 |  23.90 |  33.31 |   +206% |
| benchmark_gather_batch / edg=(2, 1000000) nod=(10, 64) bat=16          | 106.14 | 239.37 | 297.33 |   +180% |
| benchmark_gather_batch / edg=(2, 1000) nod=(100, 64) bat=16            |   0.73 |   0.82 |   0.50 |    -31% |
| benchmark_gather_batch / edg=(2, 10000) nod=(100, 64) bat=16           |   1.93 |   2.50 |   2.60 |    +34% |
| benchmark_gather_batch / edg=(2, 100000) nod=(100, 64) bat=16          |  11.04 |  23.97 |  32.09 |   +190% |
| benchmark_gather_batch / edg=(2, 1000) nod=(1000, 64) bat=16           |   1.06 |   0.32 |   0.42 |    -59% |
| benchmark_gather_batch / edg=(2, 10000) nod=(1000, 64) bat=16          |   2.16 |   2.93 |   2.68 |    +23% |
| benchmark_gather_batch / edg=(2, 1000) nod=(10000, 64) bat=16          |   0.34 |   0.35 |   0.46 |    +35% |
| benchmark_gather_batch / edg=(2, 1000) nod=(10, 64) bat=128            |   1.57 |   1.99 |   2.46 |    +56% |
| benchmark_gather_batch / edg=(2, 10000) nod=(10, 64) bat=128           |   9.12 |  19.11 |  28.70 |   +214% |
| benchmark_gather_batch / edg=(2, 100000) nod=(10, 64) bat=128          |  86.06 | 190.35 | 246.59 |   +186% |
| benchmark_gather_batch / edg=(2, 1000) nod=(100, 64) bat=128           |   1.49 |   1.94 |   2.10 |    +41% |
| benchmark_gather_batch / edg=(2, 10000) nod=(100, 64) bat=128          |   8.86 |  19.02 |  28.44 |   +220% |
| benchmark_gather_batch / edg=(2, 1000) nod=(1000, 64) bat=128          |   1.12 |   2.26 |   2.15 |    +92% |
| benchmark_gather_batch / edg=(2, 1000) nod=(10, 64) bat=1024           |   7.06 |  15.89 |  26.75 |   +278% |
| benchmark_gather_batch / edg=(2, 10000) nod=(10, 64) bat=1024          |  68.07 | 153.53 | 194.82 |   +186% |
| benchmark_gather_batch / edg=(2, 1000) nod=(100, 64) bat=128           |   1.49 |   1.94 |   2.10 |    +41% |
| benchmark_gather_batch / edg=(2, 10000) nod=(100, 64) bat=128          |   8.86 |  19.02 |  28.44 |   +220% |
| benchmark_gather_batch / edg=(2, 1000) nod=(1000, 64) bat=128          |   1.12 |   2.26 |   2.15 |    +92% |
| benchmark_gather_batch / edg=(2, 1000) nod=(10, 64) bat=1024           |   7.06 |  15.89 |  26.75 |   +278% |
| benchmark_gather_batch / edg=(2, 10000) nod=(10, 64) bat=1024          |  68.07 | 153.53 | 194.82 |   +186% |
| benchmark_gather_batch / edg=(2, 1000) nod=(100, 64) bat=1024          |   7.16 |  15.28 |  26.74 |   +273% |
| benchmark_scatter / edg=(2, 1000) nod=(10, 64) sca=add                 |   0.36 |   0.54 |   0.12 |    -65% |
| benchmark_scatter / edg=(2, 10000) nod=(10, 64) sca=add                |   0.67 |   4.88 |   1.13 |    +68% |
| benchmark_scatter / edg=(2, 100000) nod=(10, 64) sca=add               |   1.61 |  19.44 |   2.96 |    +83% |
| benchmark_scatter / edg=(2, 1000000) nod=(10, 64) sca=add              |   8.60 | 195.10 |  35.67 |   +314% |
| benchmark_scatter / edg=(2, 1000) nod=(100, 64) sca=add                |   0.33 |   0.57 |   0.20 |    -39% |
| benchmark_scatter / edg=(2, 10000) nod=(100, 64) sca=add               |   0.69 |   4.92 |   0.32 |    -53% |
| benchmark_scatter / edg=(2, 100000) nod=(100, 64) sca=add              |   4.19 |  19.43 |   2.76 |    -34% |
| benchmark_scatter / edg=(2, 1000000) nod=(100, 64) sca=add             |   8.65 | 194.63 |  31.27 |   +261% |
| benchmark_scatter / edg=(2, 1000) nod=(1000, 64) sca=add               |   0.88 |   0.49 |   0.16 |    -81% |
| benchmark_scatter / edg=(2, 10000) nod=(1000, 64) sca=add              |   0.74 |   5.05 |   0.31 |    -58% |
| benchmark_scatter / edg=(2, 100000) nod=(1000, 64) sca=add             |   1.57 |  19.42 |   2.76 |    +75% |
| benchmark_scatter / edg=(2, 1000000) nod=(1000, 64) sca=add            |   8.70 | 194.82 |  29.03 |   +233% |
| benchmark_scatter / edg=(2, 1000) nod=(10000, 64) sca=add              |   0.48 |   0.31 |   0.17 |    -65% |
| benchmark_scatter / edg=(2, 10000) nod=(10000, 64) sca=add             |   0.78 |   4.68 |   0.40 |    -49% |
| benchmark_scatter / edg=(2, 100000) nod=(10000, 64) sca=add            |   3.31 |  20.30 |   3.28 |      0% |
| benchmark_scatter / edg=(2, 1000000) nod=(10000, 64) sca=add           |   8.95 | 194.43 |  28.87 |   +222% |
| benchmark_scatter / edg=(2, 1000) nod=(10, 64) sca=max                 |   0.35 |   0.42 |   0.28 |    -21% |
| benchmark_scatter / edg=(2, 10000) nod=(10, 64) sca=max                |   0.68 |   2.02 |   0.78 |    +13% |
| benchmark_scatter / edg=(2, 100000) nod=(10, 64) sca=max               |   2.38 |  19.40 |   6.59 |   +176% |
| benchmark_scatter / edg=(2, 1000000) nod=(10, 64) sca=max              |   8.61 | 194.38 |  67.21 |   +680% |
| benchmark_scatter / edg=(2, 1000) nod=(100, 64) sca=max                |   0.37 |   0.54 |   0.28 |    -24% |
| benchmark_scatter / edg=(2, 10000) nod=(100, 64) sca=max               |   0.70 |   1.98 |   0.73 |     +4% |
| benchmark_scatter / edg=(2, 100000) nod=(100, 64) sca=max              |   1.65 |  19.39 |   6.39 |   +286% |
| benchmark_scatter / edg=(2, 1000000) nod=(100, 64) sca=max             |   8.66 | 194.32 |  63.99 |   +638% |
| benchmark_scatter / edg=(2, 1000) nod=(1000, 64) sca=max               |   0.38 |   0.57 |   0.49 |    +27% |
| benchmark_scatter / edg=(2, 10000) nod=(1000, 64) sca=max              |   0.70 |   2.69 |   0.70 |     +0% |
| benchmark_scatter / edg=(2, 10000) nod=(1000, 64) sca=max              |   0.70 |   2.69 |   0.70 |     +0% |
| benchmark_scatter / edg=(2, 100000) nod=(1000, 64) sca=max             |   4.93 |  19.53 |   6.48 |    +31% |
| benchmark_scatter / edg=(2, 1000000) nod=(1000, 64) sca=max            |   8.66 | 193.89 |  59.23 |   +583% |
| benchmark_scatter / edg=(2, 1000) nod=(10000, 64) sca=max              |   0.49 |   0.74 |   0.55 |    +12% |
| benchmark_scatter / edg=(2, 10000) nod=(10000, 64) sca=max             |   0.79 |   5.30 |   0.87 |     +9% |
| benchmark_scatter / edg=(2, 100000) nod=(10000, 64) sca=max            |   1.88 |  19.49 |   6.17 |   +227% |
| benchmark_scatter / edg=(2, 1000000) nod=(10000, 64) sca=max           |   8.85 | 193.81 |  61.40 |   +593% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(10, 64) sca=add bat=16    |   0.88 |   3.15 |   0.54 |    -38% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(10, 64) sca=add bat=16   |   5.93 |  31.09 |   5.00 |    -15% |
| benchmark_scatter_batch / edg=(2, 100000) nod=(10, 64) sca=add bat=16  |  14.23 | 311.09 |  59.60 |   +319% |
| benchmark_scatter_batch / edg=(2, 1000000) nod=(10, 64) sca=add bat=16 | 132.26 | 3113.64 | 639.35 |   +383% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(100, 64) sca=add bat=16   |   0.88 |   3.16 |   0.44 |    -50% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(100, 64) sca=add bat=16  |   2.27 |  31.09 |   4.61 |   +103% |
| benchmark_scatter_batch / edg=(2, 100000) nod=(100, 64) sca=add bat=16 |  13.60 | 310.38 |  49.62 |   +264% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(1000, 64) sca=add bat=16  |   0.89 |   3.15 |   0.42 |    -52% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(1000, 64) sca=add bat=16 |   2.29 |  31.07 |   4.79 |   +109% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(10000, 64) sca=add bat=16 |   0.50 |   3.29 |   0.57 |    +14% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(10, 64) sca=add bat=128   |   2.31 |  24.85 |   3.79 |    +64% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(10, 64) sca=add bat=128  |  10.94 | 248.21 |  48.76 |   +345% |
| benchmark_scatter_batch / edg=(2, 100000) nod=(10, 64) sca=add bat=128 | 105.85 | 2483.69 | 515.55 |   +387% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(100, 64) sca=add bat=128  |   1.89 |  24.98 |   3.51 |    +85% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(100, 64) sca=add bat=128 |  10.95 | 248.19 |  39.05 |   +256% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(1000, 64) sca=add bat=128 |   1.43 |  24.84 |   3.68 |   +156% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(10, 64) sca=add bat=1024  |   8.76 | 198.45 |  39.16 |   +347% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(10, 64) sca=add bat=1024 |  84.74 | 1984.30 | 404.31 |   +377% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(100, 64) sca=add bat=1024 |   8.84 | 198.25 |  31.90 |   +260% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(10, 64) sca=max bat=16    |   0.51 |   3.15 |   1.19 |   +134% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(10, 64) sca=max bat=16   |   2.29 |  31.01 |  10.70 |   +367% |
| benchmark_scatter_batch / edg=(2, 100000) nod=(10, 64) sca=max bat=16  |  13.57 | 310.00 | 114.21 |   +741% |
| benchmark_scatter_batch / edg=(2, 1000000) nod=(10, 64) sca=max bat=16 | 132.17 | 3101.72 | 1180.49 |   +793% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(100, 64) sca=max bat=16   |   0.91 |   3.14 |   1.10 |    +20% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(100, 64) sca=max bat=16  |   2.55 |  31.02 |  10.65 |   +317% |
| benchmark_scatter_batch / edg=(2, 100000) nod=(100, 64) sca=max bat=16 |  13.57 | 309.89 | 102.12 |   +652% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(1000, 64) sca=max bat=16  |   0.91 |   3.15 |   1.05 |    +15% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(1000, 64) sca=max bat=16 |   2.33 |  30.95 |   9.71 |   +317% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(10000, 64) sca=max bat=16 |   0.46 |   3.28 |   1.11 |   +139% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(10, 64) sca=max bat=128   |   2.00 |  24.77 |   8.41 |   +319% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(10, 64) sca=max bat=128  |  10.95 | 247.77 |  93.31 |   +752% |
| benchmark_scatter_batch / edg=(2, 100000) nod=(10, 64) sca=max bat=128 | 105.77 | 2484.40 | 949.21 |   +797% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(100, 64) sca=max bat=128  |   1.48 |  24.84 |   8.31 |   +460% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(100, 64) sca=max bat=128 |  10.95 | 248.09 |  82.96 |   +657% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(1000, 64) sca=max bat=128 |   1.44 |  24.79 |   8.05 |   +458% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(10, 64) sca=max bat=1024  |   8.77 | 198.22 |  68.53 |   +681% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(10, 64) sca=max bat=1024 |  84.69 | 1984.33 | 751.86 |   +787% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(100, 64) sca=max bat=1024 |   8.83 | 198.56 |  64.42 |   +629% |
| benchmark_GATConv / edg=(2, 1000) nod=(10, 64) in_=64 out=64           |   0.77 |   0.48 |   0.20 |    -74% |
| benchmark_GATConv / edg=(2, 10000) nod=(10, 64) in_=64 out=64          |   1.48 |   3.22 |   0.86 |    -41% |
| benchmark_GATConv / edg=(2, 100000) nod=(10, 64) in_=64 out=64         |   3.19 |  30.29 |   6.15 |    +92% |
| benchmark_GATConv / edg=(2, 1000000) nod=(10, 64) in_=64 out=64        |  18.18 | 302.38 |  71.28 |   +292% |
| benchmark_GATConv / edg=(2, 1000) nod=(100, 64) in_=64 out=64          |   0.90 |   1.21 |   0.28 |    -69% |
| benchmark_GATConv / edg=(2, 10000) nod=(100, 64) in_=64 out=64         |   1.52 |   3.29 |   0.94 |    -38% |
| benchmark_GATConv / edg=(2, 100000) nod=(100, 64) in_=64 out=64        |   3.64 |  30.28 |   6.40 |    +75% |
| benchmark_GATConv / edg=(2, 1000000) nod=(100, 64) in_=64 out=64       |  18.04 | 302.50 |  70.46 |   +290% |
| benchmark_GATConv / edg=(2, 1000) nod=(1000, 64) in_=64 out=64         |   1.19 |   1.36 |   0.40 |    -66% |
| benchmark_GATConv / edg=(2, 10000) nod=(1000, 64) in_=64 out=64        |   1.68 |   3.33 |   1.16 |    -30% |
| benchmark_GATConv / edg=(2, 100000) nod=(1000, 64) in_=64 out=64       |   3.59 |  30.49 |   6.55 |    +82% |
| benchmark_GATConv / edg=(2, 1000000) nod=(1000, 64) in_=64 out=64      |  18.14 | 302.94 |  68.04 |   +275% |
| benchmark_GATConv / edg=(2, 1000) nod=(10000, 64) in_=64 out=64        |   1.05 |   2.79 |   1.81 |    +73% |
| benchmark_GATConv / edg=(2, 10000) nod=(10000, 64) in_=64 out=64       |   1.28 |   4.00 |   2.56 |    +99% |
| benchmark_GATConv / edg=(2, 100000) nod=(10000, 64) in_=64 out=64      |   3.45 |  31.27 |   8.00 |   +131% |
| benchmark_GATConv / edg=(2, 1000000) nod=(10000, 64) in_=64 out=64     |  18.35 | 305.70 |  69.84 |   +280% |
| benchmark_GCNConv / edg=(2, 1000) nod=(10, 64) in_=64 out=64           |   0.70 |   0.87 |   0.35 |    -50% |
| benchmark_GCNConv / edg=(2, 10000) nod=(10, 64) in_=64 out=64          |   1.55 |   5.83 |   0.71 |    -54% |
| benchmark_GCNConv / edg=(2, 100000) nod=(10, 64) in_=64 out=64         |   2.76 |  22.08 |   4.87 |    +76% |
| benchmark_GCNConv / edg=(2, 1000000) nod=(10, 64) in_=64 out=64        |  16.24 | 219.03 |  61.05 |   +275% |
| benchmark_GCNConv / edg=(2, 1000) nod=(100, 64) in_=64 out=64          |   0.77 |   0.88 |   0.46 |    -40% |
| benchmark_GCNConv / edg=(2, 10000) nod=(100, 64) in_=64 out=64         |   1.58 |   2.36 |   0.79 |    -49% |
| benchmark_GCNConv / edg=(2, 100000) nod=(100, 64) in_=64 out=64        |   2.82 |  22.01 |   5.37 |    +90% |
| benchmark_GCNConv / edg=(2, 1000000) nod=(100, 64) in_=64 out=64       |  16.18 | 219.48 |  63.10 |   +289% |
| benchmark_GCNConv / edg=(2, 1000) nod=(1000, 64) in_=64 out=64         |   0.97 |   0.98 |   0.63 |    -35% |
| benchmark_GCNConv / edg=(2, 10000) nod=(1000, 64) in_=64 out=64        |   1.77 |   2.42 |   0.89 |    -49% |
| benchmark_GCNConv / edg=(2, 100000) nod=(1000, 64) in_=64 out=64       |   6.47 |  22.21 |   5.26 |    -18% |
| benchmark_GCNConv / edg=(2, 1000000) nod=(1000, 64) in_=64 out=64      |  17.05 | 224.40 |  64.95 |   +280% |
| benchmark_GCNConv / edg=(2, 1000) nod=(10000, 64) in_=64 out=64        |   1.40 |   0.62 |   1.62 |    +15% |
| benchmark_GCNConv / edg=(2, 10000) nod=(10000, 64) in_=64 out=64       |   1.66 |   2.77 |   2.01 |    +21% |
| benchmark_GCNConv / edg=(2, 100000) nod=(10000, 64) in_=64 out=64      |   2.90 |  22.65 |   6.82 |   +135% |
| benchmark_GCNConv / edg=(2, 1000000) nod=(10000, 64) in_=64 out=64     |  16.27 | 222.24 |  65.14 |   +300% |

Average benchmark:
| Operation                    | mlx_gpu | mlx_cpu | pyg_cpu | mlx_gpu/pyg_cpu speedup |
|------------------------------|-------|-------|-------|-----------------------|
| benchmark_gather         |   2.54 |   4.72 |   7.07 |   +178% |
| benchmark_gather_batch   |  17.23 |  37.72 |  49.05 |   +184% |
| benchmark_scatter        |   3.14 |  54.61 |  13.17 |   +318% |
| benchmark_scatter_batch  |  21.41 | 487.89 | 140.05 |   +554% |
| benchmark_GATConv        |   6.03 |  84.72 |  19.68 |   +226% |
| benchmark_GCNConv        |   5.69 |  61.93 |  17.75 |   +211% |
