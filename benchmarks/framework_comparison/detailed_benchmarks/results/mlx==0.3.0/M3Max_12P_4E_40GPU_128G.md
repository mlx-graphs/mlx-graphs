
Platform macOS-14.3.1

mlx version: 0.3.0

mlx-graphs version: 0.0.3

torch version: 2.2.0

torch_geometric version: 2.5.0

Detailed benchmark:
| Operation                                                                  | mlx_gpu | mlx_cpu | pyg_cpu | mlx_gpu/pyg_cpu speedup |
|----------------------------------------------------------------------------|-------|-------|-------|-----------------------|
| benchmark_gather / edg=(2, 1000) nod=(10, 64)                          |   0.20 |   0.03 |   0.15 |    -24% |
| benchmark_gather / edg=(2, 10000) nod=(10, 64)                         |   0.34 |   0.14 |   0.31 |    -11% |
| benchmark_gather / edg=(2, 100000) nod=(10, 64)                        |   1.25 |   1.31 |   0.66 |    -47% |
| benchmark_gather / edg=(2, 1000000) nod=(10, 64)                       |   2.61 |  12.63 |  18.26 |   +599% |
| benchmark_gather / edg=(2, 1000) nod=(100, 64)                         |   0.24 |   0.03 |   0.18 |    -22% |
| benchmark_gather / edg=(2, 10000) nod=(100, 64)                        |   0.32 |   0.13 |   0.40 |    +28% |
| benchmark_gather / edg=(2, 100000) nod=(100, 64)                       |   1.31 |   1.28 |   0.69 |    -47% |
| benchmark_gather / edg=(2, 1000000) nod=(100, 64)                      |   3.33 |  12.84 |  19.02 |   +471% |
| benchmark_gather / edg=(2, 1000) nod=(1000, 64)                        |   0.27 |   0.03 |   0.17 |    -35% |
| benchmark_gather / edg=(2, 10000) nod=(1000, 64)                       |   0.38 |   0.17 |   0.28 |    -26% |
| benchmark_gather / edg=(2, 100000) nod=(1000, 64)                      |   1.32 |   1.61 |   0.74 |    -43% |
| benchmark_gather / edg=(2, 1000000) nod=(1000, 64)                     |   3.36 |  16.07 |  18.03 |   +437% |
| benchmark_gather / edg=(2, 1000) nod=(10000, 64)                       |   0.27 |   0.03 |   0.17 |    -35% |
| benchmark_gather / edg=(2, 10000) nod=(10000, 64)                      |   0.38 |   0.18 |   0.43 |    +14% |
| benchmark_gather / edg=(2, 100000) nod=(10000, 64)                     |   1.36 |   1.72 |   0.75 |    -45% |
| benchmark_gather / edg=(2, 1000000) nod=(10000, 64)                    |   2.66 |  17.08 |  17.71 |   +565% |
| benchmark_gather_batch / edg=(2, 1000) nod=(10, 64) bat=16             |   0.45 |   0.21 |   0.56 |    +25% |
| benchmark_gather_batch / edg=(2, 10000) nod=(10, 64) bat=16            |   0.87 |   2.01 |   0.96 |    +10% |
| benchmark_gather_batch / edg=(2, 100000) nod=(10, 64) bat=16           |   4.06 |  20.79 |  26.86 |   +562% |
| benchmark_gather_batch / edg=(2, 1000000) nod=(10, 64) bat=16          |  37.82 | 207.52 | 377.02 |   +896% |
| benchmark_gather_batch / edg=(2, 1000) nod=(100, 64) bat=16            |   0.45 |   0.23 |   0.59 |    +31% |
| benchmark_gather_batch / edg=(2, 10000) nod=(100, 64) bat=16           |   1.97 |   2.01 |   0.97 |    -50% |
| benchmark_gather_batch / edg=(2, 100000) nod=(100, 64) bat=16          |   4.11 |  20.74 |  30.17 |   +634% |
| benchmark_gather_batch / edg=(2, 1000) nod=(1000, 64) bat=16           |   0.28 |   0.28 |   0.35 |    +22% |
| benchmark_gather_batch / edg=(2, 10000) nod=(1000, 64) bat=16          |   0.85 |   2.62 |   1.04 |    +23% |
| benchmark_gather_batch / edg=(2, 1000) nod=(10000, 64) bat=16          |   0.28 |   0.29 |   0.62 |   +118% |
| benchmark_gather_batch / edg=(2, 1000) nod=(10, 64) bat=128            |   0.74 |   1.66 |   0.80 |     +7% |
| benchmark_gather_batch / edg=(2, 10000) nod=(10, 64) bat=128           |   3.31 |  16.69 |  24.23 |   +631% |
| benchmark_gather_batch / edg=(2, 100000) nod=(10, 64) bat=128          |  30.34 | 167.64 | 308.21 |   +915% |
| benchmark_gather_batch / edg=(2, 1000) nod=(100, 64) bat=128           |   0.73 |   1.69 |   0.80 |     +9% |
| benchmark_gather_batch / edg=(2, 10000) nod=(100, 64) bat=128          |   3.34 |  16.22 |  24.27 |   +627% |
| benchmark_gather_batch / edg=(2, 1000) nod=(1000, 64) bat=128          |   0.55 |   2.08 |   0.86 |    +54% |
| benchmark_gather_batch / edg=(2, 1000) nod=(10, 64) bat=1024           |   2.65 |  13.30 |  18.92 |   +612% |
| benchmark_gather_batch / edg=(2, 10000) nod=(10, 64) bat=1024          |  24.34 | 136.00 | 246.61 |   +913% |
| benchmark_gather_batch / edg=(2, 1000) nod=(100, 64) bat=1024          |   2.68 |  13.49 |  19.02 |   +610% |
| benchmark_scatter / edg=(2, 1000) nod=(10, 64) sca=add                 |   0.48 |   0.21 |   0.05 |    -88% |
| benchmark_scatter / edg=(2, 10000) nod=(10, 64) sca=add                |   0.59 |   1.98 |   0.44 |    -24% |
| benchmark_scatter / edg=(2, 100000) nod=(10, 64) sca=add               |   1.78 |  19.38 |   3.82 |   +114% |
| benchmark_scatter / edg=(2, 1000000) nod=(10, 64) sca=add              |   3.33 | 193.90 |  34.71 |   +942% |
| benchmark_scatter / edg=(2, 1000) nod=(100, 64) sca=add                |   0.46 |   0.21 |   0.06 |    -87% |
| benchmark_scatter / edg=(2, 10000) nod=(100, 64) sca=add               |   0.61 |   1.96 |   0.35 |    -41% |
| benchmark_scatter / edg=(2, 100000) nod=(100, 64) sca=add              |   1.85 |  19.34 |   2.53 |    +37% |
| benchmark_scatter / edg=(2, 1000000) nod=(100, 64) sca=add             |   3.36 | 193.89 |  26.14 |   +678% |
| benchmark_scatter / edg=(2, 1000) nod=(1000, 64) sca=add               |   0.48 |   0.22 |   0.07 |    -85% |
| benchmark_scatter / edg=(2, 10000) nod=(1000, 64) sca=add              |   0.63 |   1.96 |   0.33 |    -46% |
| benchmark_scatter / edg=(2, 100000) nod=(1000, 64) sca=add             |   1.78 |  19.44 |   2.29 |    +28% |
| benchmark_scatter / edg=(2, 1000000) nod=(1000, 64) sca=add            |   3.35 | 193.41 |  22.64 |   +575% |
| benchmark_scatter / edg=(2, 1000) nod=(10000, 64) sca=add              |   0.52 |   0.29 |   0.20 |    -62% |
| benchmark_scatter / edg=(2, 10000) nod=(10000, 64) sca=add             |   0.63 |   2.04 |   0.64 |     +0% |
| benchmark_scatter / edg=(2, 100000) nod=(10000, 64) sca=add            |   1.92 |  19.52 |   2.63 |    +36% |
| benchmark_scatter / edg=(2, 1000000) nod=(10000, 64) sca=add           |   3.34 | 193.55 |  22.49 |   +572% |
| benchmark_scatter / edg=(2, 1000) nod=(10, 64) sca=max                 |   0.39 |   0.22 |   0.16 |    -58% |
| benchmark_scatter / edg=(2, 10000) nod=(10, 64) sca=max                |   0.57 |   1.96 |   0.72 |    +25% |
| benchmark_scatter / edg=(2, 100000) nod=(10, 64) sca=max               |   1.73 |  19.39 |   6.48 |   +274% |
| benchmark_scatter / edg=(2, 1000000) nod=(10, 64) sca=max              |   3.34 | 193.33 |  58.43 |  +1651% |
| benchmark_scatter / edg=(2, 1000) nod=(100, 64) sca=max                |   0.38 |   0.22 |   0.17 |    -55% |
| benchmark_scatter / edg=(2, 10000) nod=(100, 64) sca=max               |   0.58 |   1.97 |   0.63 |     +7% |
| benchmark_scatter / edg=(2, 100000) nod=(100, 64) sca=max              |   1.67 |  19.38 |   5.40 |   +222% |
| benchmark_scatter / edg=(2, 1000000) nod=(100, 64) sca=max             |   3.31 | 193.24 |  48.08 |  +1350% |
| benchmark_scatter / edg=(2, 1000) nod=(1000, 64) sca=max               |   0.46 |   0.23 |   0.19 |    -57% |
| benchmark_scatter / edg=(2, 10000) nod=(1000, 64) sca=max              |   0.59 |   1.97 |   0.62 |     +5% |
| benchmark_scatter / edg=(2, 100000) nod=(1000, 64) sca=max             |   1.74 |  19.32 |   4.75 |   +172% |
| benchmark_scatter / edg=(2, 1000000) nod=(1000, 64) sca=max            |   3.34 | 192.73 |  45.48 |  +1261% |
| benchmark_scatter / edg=(2, 1000) nod=(10000, 64) sca=max              |   0.81 |   0.29 |   0.36 |    -54% |
| benchmark_scatter / edg=(2, 10000) nod=(10000, 64) sca=max             |   0.63 |   2.03 |   0.73 |    +16% |
| benchmark_scatter / edg=(2, 100000) nod=(10000, 64) sca=max            |   2.64 |  19.44 |   4.85 |    +83% |
| benchmark_scatter / edg=(2, 1000000) nod=(10000, 64) sca=max           |   3.39 | 192.98 |  43.27 |  +1174% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(10, 64) sca=add bat=16    |   0.66 |   3.12 |   0.63 |     -4% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(10, 64) sca=add bat=16   |   0.96 |  31.01 |   5.37 |   +457% |
| benchmark_scatter_batch / edg=(2, 100000) nod=(10, 64) sca=add bat=16  |   5.07 | 310.20 |  59.96 |  +1081% |
| benchmark_scatter_batch / edg=(2, 1000000) nod=(10, 64) sca=add bat=16 |  46.61 | 3104.84 | 609.67 |  +1207% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(100, 64) sca=add bat=16   |   0.58 |   3.14 |   0.51 |    -11% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(100, 64) sca=add bat=16  |   2.39 |  31.04 |   3.88 |    +62% |
| benchmark_scatter_batch / edg=(2, 100000) nod=(100, 64) sca=add bat=16 |   5.05 | 309.66 |  40.81 |   +708% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(1000, 64) sca=add bat=16  |   0.46 |   3.15 |   0.50 |     +8% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(1000, 64) sca=add bat=16 |   1.19 |  30.94 |   3.54 |   +197% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(10000, 64) sca=add bat=16 |   0.45 |   3.27 |   0.66 |    +45% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(10, 64) sca=add bat=128   |   0.96 |  24.77 |   4.74 |   +393% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(10, 64) sca=add bat=128  |   4.15 | 248.17 |  46.60 |  +1022% |
| benchmark_scatter_batch / edg=(2, 100000) nod=(10, 64) sca=add bat=128 |  37.39 | 2482.22 | 492.52 |  +1217% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(100, 64) sca=add bat=128  |   1.03 |  24.80 |   3.26 |   +216% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(100, 64) sca=add bat=128 |   4.13 | 247.69 |  32.46 |   +685% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(1000, 64) sca=add bat=128 |   0.72 |  24.78 |   3.17 |   +341% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(10, 64) sca=add bat=1024  |   3.31 | 198.55 |  37.43 |  +1030% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(10, 64) sca=add bat=1024 |  30.00 | 1990.17 | 393.62 |  +1211% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(100, 64) sca=add bat=1024 |   3.33 | 199.22 |  27.37 |   +721% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(10, 64) sca=max bat=16    |   0.69 |   3.12 |   1.11 |    +60% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(10, 64) sca=max bat=16   |   0.93 |  30.98 |   9.32 |   +901% |
| benchmark_scatter_batch / edg=(2, 100000) nod=(10, 64) sca=max bat=16  |   5.06 | 310.03 |  92.24 |  +1723% |
| benchmark_scatter_batch / edg=(2, 1000000) nod=(10, 64) sca=max bat=16 |  46.59 | 3100.20 | 921.22 |  +1877% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(100, 64) sca=max bat=16   |   0.46 |   3.13 |   0.99 |   +115% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(100, 64) sca=max bat=16  |   2.57 |  30.96 |   7.85 |   +205% |
| benchmark_scatter_batch / edg=(2, 100000) nod=(100, 64) sca=max bat=16 |   5.07 | 309.31 |  70.22 |  +1286% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(1000, 64) sca=max bat=16  |   0.72 |   3.14 |   0.89 |    +23% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(1000, 64) sca=max bat=16 |   0.94 |  30.91 |   7.95 |   +742% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(10000, 64) sca=max bat=16 |   0.46 |   3.28 |   1.17 |   +153% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(10, 64) sca=max bat=128   |   1.12 |  24.76 |   8.33 |   +643% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(10, 64) sca=max bat=128  |   4.15 | 247.43 |  74.37 |  +1690% |
| benchmark_scatter_batch / edg=(2, 100000) nod=(10, 64) sca=max bat=128 |  37.37 | 2481.14 | 735.56 |  +1868% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(100, 64) sca=max bat=128  |   1.09 |  24.81 |   6.45 |   +489% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(100, 64) sca=max bat=128 |   4.11 | 247.27 |  57.84 |  +1307% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(1000, 64) sca=max bat=128 |   0.78 |  24.75 |   6.16 |   +684% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(10, 64) sca=max bat=1024  |   3.35 | 198.28 |  60.94 |  +1721% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(10, 64) sca=max bat=1024 |  30.00 | 1983.61 | 581.67 |  +1839% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(100, 64) sca=max bat=1024 |   3.35 | 198.07 |  48.01 |  +1334% |
| benchmark_GATConv / edg=(2, 1000) nod=(10, 64) in_=64 out=64           |   0.76 |   0.67 |   0.20 |    -74% |
| benchmark_GATConv / edg=(2, 10000) nod=(10, 64) in_=64 out=64          |   1.05 |   3.38 |   1.06 |     +0% |
| benchmark_GATConv / edg=(2, 100000) nod=(10, 64) in_=64 out=64         |   3.56 |  30.28 |   5.91 |    +65% |
| benchmark_GATConv / edg=(2, 1000000) nod=(10, 64) in_=64 out=64        |   7.24 | 299.15 |  59.84 |   +725% |
| benchmark_GATConv / edg=(2, 1000) nod=(100, 64) in_=64 out=64          |   0.82 |   0.68 |   0.25 |    -69% |
| benchmark_GATConv / edg=(2, 10000) nod=(100, 64) in_=64 out=64         |   1.04 |   3.38 |   1.15 |    +11% |
| benchmark_GATConv / edg=(2, 100000) nod=(100, 64) in_=64 out=64        |   1.50 |  30.21 |   5.21 |   +247% |
| benchmark_GATConv / edg=(2, 1000000) nod=(100, 64) in_=64 out=64       |   7.13 | 299.14 |  56.00 |   +685% |
| benchmark_GATConv / edg=(2, 1000) nod=(1000, 64) in_=64 out=64         |   0.81 |   0.73 |   0.50 |    -38% |
| benchmark_GATConv / edg=(2, 10000) nod=(1000, 64) in_=64 out=64        |   1.06 |   3.43 |   1.40 |    +32% |
| benchmark_GATConv / edg=(2, 100000) nod=(1000, 64) in_=64 out=64       |   1.23 |  30.52 |   5.59 |   +356% |
| benchmark_GATConv / edg=(2, 1000000) nod=(1000, 64) in_=64 out=64      |   7.00 | 300.87 |  52.89 |   +655% |
| benchmark_GATConv / edg=(2, 1000) nod=(10000, 64) in_=64 out=64        |   1.15 |   1.32 |   2.37 |   +106% |
| benchmark_GATConv / edg=(2, 10000) nod=(10000, 64) in_=64 out=64       |   1.38 |   4.44 |   3.36 |   +143% |
| benchmark_GATConv / edg=(2, 100000) nod=(10000, 64) in_=64 out=64      |   3.52 |  31.26 |   7.38 |   +109% |
| benchmark_GATConv / edg=(2, 1000000) nod=(10000, 64) in_=64 out=64     |   7.11 | 301.85 |  55.12 |   +675% |
| benchmark_GCNConv / edg=(2, 1000) nod=(10, 64) in_=64 out=64           |   0.66 |   0.56 |   0.16 |    -75% |
| benchmark_GCNConv / edg=(2, 10000) nod=(10, 64) in_=64 out=64          |   0.91 |   2.52 |   1.00 |    +10% |
| benchmark_GCNConv / edg=(2, 100000) nod=(10, 64) in_=64 out=64         |   2.88 |  22.07 |   4.71 |    +63% |
| benchmark_GCNConv / edg=(2, 1000000) nod=(10, 64) in_=64 out=64        |   6.49 | 218.10 |  54.97 |   +747% |
| benchmark_GCNConv / edg=(2, 1000) nod=(100, 64) in_=64 out=64          |   0.69 |   0.56 |   0.17 |    -75% |
| benchmark_GCNConv / edg=(2, 10000) nod=(100, 64) in_=64 out=64         |   0.99 |   2.53 |   0.97 |     -1% |
| benchmark_GCNConv / edg=(2, 100000) nod=(100, 64) in_=64 out=64        |   1.40 |  22.11 |   4.97 |   +254% |
| benchmark_GCNConv / edg=(2, 1000000) nod=(100, 64) in_=64 out=64       |   6.33 | 217.04 |  48.00 |   +657% |
| benchmark_GCNConv / edg=(2, 1000) nod=(1000, 64) in_=64 out=64         |   0.74 |   0.71 |   0.23 |    -69% |
| benchmark_GCNConv / edg=(2, 10000) nod=(1000, 64) in_=64 out=64        |   0.97 |   2.57 |   0.90 |     -7% |
| benchmark_GCNConv / edg=(2, 100000) nod=(1000, 64) in_=64 out=64       |   2.10 |  22.18 |   4.20 |   +100% |
| benchmark_GCNConv / edg=(2, 1000000) nod=(1000, 64) in_=64 out=64      |   6.29 | 218.75 |  47.55 |   +655% |
| benchmark_GCNConv / edg=(2, 1000) nod=(10000, 64) in_=64 out=64        |   0.81 |   0.81 |   1.67 |   +106% |
| benchmark_GCNConv / edg=(2, 10000) nod=(10000, 64) in_=64 out=64       |   1.02 |   2.85 |   2.74 |   +169% |
| benchmark_GCNConv / edg=(2, 100000) nod=(10000, 64) in_=64 out=64      |   1.68 |  22.65 |   5.29 |   +215% |
| benchmark_GCNConv / edg=(2, 1000000) nod=(10000, 64) in_=64 out=64     |   6.29 | 219.64 |  47.73 |   +658% |

Average benchmark:
| Operation                    | mlx_gpu | mlx_cpu | pyg_cpu | mlx_gpu/pyg_cpu speedup |
|------------------------------|-------|-------|-------|-----------------------|
| benchmark_gather         |   1.23 |   4.08 |   4.87 |   +297% |
| benchmark_gather_batch   |   6.31 |  32.92 |  56.99 |   +803% |
| benchmark_scatter        |   1.58 |  53.75 |  10.62 |   +570% |
| benchmark_scatter_batch  |   7.82 | 487.52 | 117.34 |  +1400% |
| benchmark_GATConv        |   2.90 |  83.83 |  16.14 |   +457% |
| benchmark_GCNConv        |   2.52 |  60.98 |  14.08 |   +459% |
