Platform macOS-14.2

mlx version: 0.4.0

mlx-graphs version: 0.0.2

torch version: 2.1.2

torch_geometric version: 2.4.0

Detailed benchmark:
| Operation                                                                  | mlx_gpu | mlx_cpu | pyg_cpu | mlx_gpu/pyg_cpu speedup |
|----------------------------------------------------------------------------|-------|-------|-------|-----------------------|
| benchmark_gather / edg=(2, 1000) nod=(10, 64)                          |   0.36 |   0.12 |   0.37 |     +4% |
| benchmark_gather / edg=(2, 10000) nod=(10, 64)                         |   0.56 |   0.21 |   0.37 |    -34% |
| benchmark_gather / edg=(2, 100000) nod=(10, 64)                        |   1.46 |   1.92 |   2.46 |    +68% |
| benchmark_gather / edg=(2, 1000000) nod=(10, 64)                       |   8.08 |  19.05 |  35.82 |   +343% |
| benchmark_gather / edg=(2, 1000) nod=(100, 64)                         |   0.36 |   0.18 |   0.45 |    +24% |
| benchmark_gather / edg=(2, 10000) nod=(100, 64)                        |   0.63 |   0.69 |   1.39 |   +120% |
| benchmark_gather / edg=(2, 100000) nod=(100, 64)                       |   1.99 |   1.93 |   2.46 |    +23% |
| benchmark_gather / edg=(2, 1000000) nod=(100, 64)                      |   8.24 |  19.00 |  34.83 |   +322% |
| benchmark_gather / edg=(2, 1000) nod=(1000, 64)                        |   0.28 |   0.05 |   0.15 |    -45% |
| benchmark_gather / edg=(2, 10000) nod=(1000, 64)                       |   0.67 |   1.05 |   0.62 |     -7% |
| benchmark_gather / edg=(2, 100000) nod=(1000, 64)                      |   1.45 |   2.17 |   2.83 |    +94% |
| benchmark_gather / edg=(2, 1000000) nod=(1000, 64)                     |   8.60 |  21.36 |  35.96 |   +318% |
| benchmark_gather / edg=(2, 1000) nod=(10000, 64)                       |   0.36 |   0.05 |   0.14 |    -61% |
| benchmark_gather / edg=(2, 10000) nod=(10000, 64)                      |   0.66 |   1.30 |   0.38 |    -42% |
| benchmark_gather / edg=(2, 100000) nod=(10000, 64)                     |   1.76 |   3.89 |   2.88 |    +64% |
| benchmark_gather / edg=(2, 1000000) nod=(10000, 64)                    |   8.73 |  63.34 |  35.93 |   +311% |
| benchmark_gather_batch / edg=(2, 1000) nod=(10, 64) bat=16             |   0.79 |   0.34 |   0.52 |    -34% |
| benchmark_gather_batch / edg=(2, 10000) nod=(10, 64) bat=16            |   1.95 |   3.05 |   3.87 |    +98% |
| benchmark_gather_batch / edg=(2, 100000) nod=(10, 64) bat=16           |  12.66 |  30.40 |  42.99 |   +239% |
| benchmark_gather_batch / edg=(2, 1000000) nod=(10, 64) bat=16          | 123.80 | 306.80 | 374.75 |   +202% |
| benchmark_gather_batch / edg=(2, 1000) nod=(100, 64) bat=16            |   0.79 |   0.33 |   0.52 |    -33% |
| benchmark_gather_batch / edg=(2, 10000) nod=(100, 64) bat=16           |   2.05 |   3.06 |   3.91 |    +90% |
| benchmark_gather_batch / edg=(2, 100000) nod=(100, 64) bat=16          |  13.02 |  30.45 |  43.50 |   +234% |
| benchmark_gather_batch / edg=(2, 1000) nod=(1000, 64) bat=16           |   0.78 |   0.38 |   0.51 |    -34% |
| benchmark_gather_batch / edg=(2, 10000) nod=(1000, 64) bat=16          |   2.05 |   3.41 |   3.99 |    +94% |
| benchmark_gather_batch / edg=(2, 1000) nod=(10000, 64) bat=16          |   0.45 |   1.42 |   0.61 |    +36% |
| benchmark_gather_batch / edg=(2, 1000) nod=(10, 64) bat=128            |   1.60 |   2.50 |   3.16 |    +97% |
| benchmark_gather_batch / edg=(2, 10000) nod=(10, 64) bat=128           |  10.23 |  24.40 |  37.43 |   +265% |
| benchmark_gather_batch / edg=(2, 100000) nod=(10, 64) bat=128          |  99.14 | 243.70 | 312.24 |   +214% |
| benchmark_gather_batch / edg=(2, 1000) nod=(100, 64) bat=128           |   1.52 |   2.45 |   3.17 |   +107% |
| benchmark_gather_batch / edg=(2, 10000) nod=(100, 64) bat=128          |  10.45 |  24.58 |  38.05 |   +264% |
| benchmark_gather_batch / edg=(2, 1000) nod=(1000, 64) bat=128          |   1.33 |   2.88 |   3.21 |   +141% |
| benchmark_gather_batch / edg=(2, 1000) nod=(10, 64) bat=1024           |   8.18 |  19.51 |  37.73 |   +360% |
| benchmark_gather_batch / edg=(2, 10000) nod=(10, 64) bat=1024          |  79.58 | 196.47 | 246.31 |   +209% |
| benchmark_gather_batch / edg=(2, 1000) nod=(100, 64) bat=1024          |   8.44 |  19.46 |  36.35 |   +330% |
| benchmark_scatter / edg=(2, 1000) nod=(10, 64) sca=add                 |   0.86 |   1.20 |   0.25 |    -70% |
| benchmark_scatter / edg=(2, 10000) nod=(10, 64) sca=add                |   1.74 |   2.41 |   0.64 |    -63% |
| benchmark_scatter / edg=(2, 100000) nod=(10, 64) sca=add               |  12.37 |  23.58 |   6.06 |    -51% |
| benchmark_scatter / edg=(2, 1000000) nod=(10, 64) sca=add              | 131.69 | 234.30 |  56.46 |    -57% |
| benchmark_scatter / edg=(2, 1000) nod=(100, 64) sca=add                |   0.48 |   0.29 |   0.08 |    -84% |
| benchmark_scatter / edg=(2, 10000) nod=(100, 64) sca=add               |   0.84 |   2.35 |   0.58 |    -30% |
| benchmark_scatter / edg=(2, 100000) nod=(100, 64) sca=add              |   2.77 |  23.52 |   5.35 |    +93% |
| benchmark_scatter / edg=(2, 1000000) nod=(100, 64) sca=add             |  15.73 | 234.22 |  48.28 |   +206% |
| benchmark_scatter / edg=(2, 1000) nod=(1000, 64) sca=add               |   0.65 |   0.29 |   0.08 |    -87% |
| benchmark_scatter / edg=(2, 10000) nod=(1000, 64) sca=add              |   0.70 |   2.43 |   0.59 |    -14% |
| benchmark_scatter / edg=(2, 100000) nod=(1000, 64) sca=add             |   4.20 |  23.44 |   5.24 |    +24% |
| benchmark_scatter / edg=(2, 1000000) nod=(1000, 64) sca=add            |   7.55 | 237.81 | 102.49 |  +1257% |
| benchmark_scatter / edg=(2, 1000) nod=(10000, 64) sca=add              |   6.18 |   0.45 |   0.30 |    -95% |
| benchmark_scatter / edg=(2, 10000) nod=(10000, 64) sca=add             |   0.77 |   3.60 |   0.77 |      0% |
| benchmark_scatter / edg=(2, 100000) nod=(10000, 64) sca=add            |   2.62 |  28.62 |   5.86 |   +123% |
| benchmark_scatter / edg=(2, 1000000) nod=(10000, 64) sca=add           |  11.83 | 300.87 |  59.36 |   +401% |
| benchmark_scatter / edg=(2, 1000) nod=(10, 64) sca=max                 |   3.12 |   0.35 |   0.24 |    -92% |
| benchmark_scatter / edg=(2, 10000) nod=(10, 64) sca=max                |   0.67 |   3.38 |   1.45 |   +117% |
| benchmark_scatter / edg=(2, 100000) nod=(10, 64) sca=max               |   5.00 |  27.45 |  12.85 |   +156% |
| benchmark_scatter / edg=(2, 1000000) nod=(10, 64) sca=max              |   6.99 | 265.14 | 142.35 |  +1936% |
| benchmark_scatter / edg=(2, 1000) nod=(100, 64) sca=max                |   4.94 |   0.40 |   0.28 |    -94% |
| benchmark_scatter / edg=(2, 10000) nod=(100, 64) sca=max               |   0.78 |   3.44 |   1.37 |    +75% |
| benchmark_scatter / edg=(2, 100000) nod=(100, 64) sca=max              |   2.58 |  30.23 |  13.41 |   +420% |
| benchmark_scatter / edg=(2, 1000000) nod=(100, 64) sca=max             |   6.70 | 270.26 | 114.03 |  +1601% |
| benchmark_scatter / edg=(2, 1000) nod=(1000, 64) sca=max               |   0.83 |   0.37 |   0.29 |    -64% |
| benchmark_scatter / edg=(2, 10000) nod=(1000, 64) sca=max              |   0.75 |   3.76 |   1.49 |    +98% |
| benchmark_scatter / edg=(2, 100000) nod=(1000, 64) sca=max             |   2.48 |  29.99 |  11.65 |   +369% |
| benchmark_scatter / edg=(2, 1000000) nod=(1000, 64) sca=max            |   3.95 | 269.15 | 117.29 |  +2867% |
| benchmark_scatter / edg=(2, 1000) nod=(10000, 64) sca=max              |   5.23 |   0.51 |   0.51 |    -90% |
| benchmark_scatter / edg=(2, 10000) nod=(10000, 64) sca=max             |   0.74 |   3.54 |   1.61 |   +118% |
| benchmark_scatter / edg=(2, 100000) nod=(10000, 64) sca=max            |   5.10 |  29.80 |  11.70 |   +129% |
| benchmark_scatter / edg=(2, 1000000) nod=(10000, 64) sca=max           |   4.14 | 282.07 | 120.61 |  +2810% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(10, 64) sca=add bat=16    |   8.82 |   4.09 |   1.01 |    -88% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(10, 64) sca=add bat=16   |  42.77 |  36.92 |   9.22 |    -78% |
| benchmark_scatter_batch / edg=(2, 100000) nod=(10, 64) sca=add bat=16  | 305.73 | 434.22 | 109.17 |    -64% |
| benchmark_scatter_batch / edg=(2, 1000000) nod=(10, 64) sca=add bat=16 | 2306.40 | 4384.02 | 1522.30 |    -33% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(100, 64) sca=add bat=16   |   4.83 |   5.42 |   1.09 |    -77% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(100, 64) sca=add bat=16  |   6.02 |  43.80 |   9.37 |    +55% |
| benchmark_scatter_batch / edg=(2, 100000) nod=(100, 64) sca=add bat=16 |  38.48 | 427.09 |  92.25 |   +139% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(1000, 64) sca=add bat=16  |   0.84 |   5.41 |   0.96 |    +14% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(1000, 64) sca=add bat=16 |   2.50 |  43.01 |   8.44 |   +237% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(10000, 64) sca=add bat=16 |   4.94 |   4.84 |   1.31 |    -73% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(10, 64) sca=add bat=128   |  48.51 |  29.29 |   7.49 |    -84% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(10, 64) sca=add bat=128  | 194.94 | 297.31 | 150.20 |    -22% |
| benchmark_scatter_batch / edg=(2, 100000) nod=(10, 64) sca=add bat=128 | 2443.32 | 3432.01 | 1215.32 |    -50% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(100, 64) sca=add bat=128  |   3.80 |  33.34 |   7.53 |    +98% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(100, 64) sca=add bat=128 |  27.24 | 303.16 |  72.88 |   +167% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(1000, 64) sca=add bat=128 |   2.12 |  29.23 |   6.66 |   +214% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(10, 64) sca=add bat=1024  | 199.34 | 236.45 |  74.29 |    -62% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(10, 64) sca=add bat=1024 | 1473.74 | 2689.64 | 854.32 |    -42% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(100, 64) sca=add bat=1024 |  12.17 | 237.51 |  52.55 |   +331% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(10, 64) sca=max bat=16    |   0.53 |   3.73 |   1.58 |   +195% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(10, 64) sca=max bat=16   |   0.72 |  37.33 |  17.23 |  +2298% |
| benchmark_scatter_batch / edg=(2, 100000) nod=(10, 64) sca=max bat=16  |   5.87 | 374.04 | 142.65 |  +2330% |
| benchmark_scatter_batch / edg=(2, 1000000) nod=(10, 64) sca=max bat=16 |  41.54 | 3753.09 | 1468.61 |  +3435% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(100, 64) sca=max bat=16   |   2.28 |   3.78 |   1.70 |    -25% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(100, 64) sca=max bat=16  |   1.37 |  37.53 |  15.97 |  +1062% |
| benchmark_scatter_batch / edg=(2, 100000) nod=(100, 64) sca=max bat=16 |   7.80 | 374.76 | 139.60 |  +1689% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(1000, 64) sca=max bat=16  |   0.59 |   3.82 |   1.70 |   +187% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(1000, 64) sca=max bat=16 |   0.78 |  37.47 |  16.02 |  +1941% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(10000, 64) sca=max bat=16 |   0.60 |   4.25 |   1.99 |   +231% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(10, 64) sca=max bat=128   |   1.48 |  29.97 |  14.32 |   +868% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(10, 64) sca=max bat=128  |   4.38 | 300.15 | 120.86 |  +2662% |
| benchmark_scatter_batch / edg=(2, 100000) nod=(10, 64) sca=max bat=128 |  20.87 | 2949.28 | 1094.40 |  +5142% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(100, 64) sca=max bat=128  |   0.61 |  29.46 |  12.56 |  +1945% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(100, 64) sca=max bat=128 |   2.56 | 294.65 | 104.89 |  +4001% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(1000, 64) sca=max bat=128 |   0.75 |  29.22 |  12.34 |  +1552% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(10, 64) sca=max bat=1024  |   2.04 | 234.86 |  92.80 |  +4453% |
| benchmark_scatter_batch / edg=(2, 10000) nod=(10, 64) sca=max bat=1024 |  16.47 | 2328.80 | 869.82 |  +5181% |
| benchmark_scatter_batch / edg=(2, 1000) nod=(100, 64) sca=max bat=1024 |   2.17 | 233.36 |  84.88 |  +3813% |
| benchmark_GATConv / edg=(2, 1000) nod=(10, 64) in_=64 out=64           |   2.19 |   2.58 |   0.26 |    -88% |
| benchmark_GATConv / edg=(2, 10000) nod=(10, 64) in_=64 out=64          |   7.56 |   4.37 |   1.54 |    -79% |
| benchmark_GATConv / edg=(2, 100000) nod=(10, 64) in_=64 out=64         |  47.26 |  36.50 |   9.06 |    -80% |
| benchmark_GATConv / edg=(2, 1000000) nod=(10, 64) in_=64 out=64        | 515.89 | 355.77 |  99.41 |    -80% |
| benchmark_GATConv / edg=(2, 1000) nod=(100, 64) in_=64 out=64          |   1.72 |   2.95 |   0.29 |    -82% |
| benchmark_GATConv / edg=(2, 10000) nod=(100, 64) in_=64 out=64         |   3.06 |   4.44 |   1.66 |    -45% |
| benchmark_GATConv / edg=(2, 100000) nod=(100, 64) in_=64 out=64        |   7.34 |  36.52 |   9.81 |    +33% |
| benchmark_GATConv / edg=(2, 1000000) nod=(100, 64) in_=64 out=64       |  96.90 | 355.79 | 100.39 |     +3% |
| benchmark_GATConv / edg=(2, 1000) nod=(1000, 64) in_=64 out=64         |   1.80 |   2.94 |   0.67 |    -62% |
| benchmark_GATConv / edg=(2, 10000) nod=(1000, 64) in_=64 out=64        |   2.34 |   4.54 |   2.06 |    -11% |
| benchmark_GATConv / edg=(2, 100000) nod=(1000, 64) in_=64 out=64       |   6.37 |  37.17 |  10.29 |    +61% |
| benchmark_GATConv / edg=(2, 1000000) nod=(1000, 64) in_=64 out=64      |  16.99 | 360.25 | 101.40 |   +496% |
| benchmark_GATConv / edg=(2, 1000) nod=(10000, 64) in_=64 out=64        |   2.47 |   1.93 |   2.72 |    +10% |
| benchmark_GATConv / edg=(2, 10000) nod=(10000, 64) in_=64 out=64       |   3.01 |   5.47 |   4.12 |    +36% |
| benchmark_GATConv / edg=(2, 100000) nod=(10000, 64) in_=64 out=64      |   2.47 |  38.67 |  11.77 |   +377% |
| benchmark_GATConv / edg=(2, 1000000) nod=(10000, 64) in_=64 out=64     |  13.24 | 387.34 |  99.18 |   +649% |
| benchmark_GCNConv / edg=(2, 1000) nod=(10, 64) in_=64 out=64           |   1.24 |   0.60 |   0.20 |    -83% |
| benchmark_GCNConv / edg=(2, 10000) nod=(10, 64) in_=64 out=64          |   3.25 |   2.90 |   1.32 |    -59% |
| benchmark_GCNConv / edg=(2, 100000) nod=(10, 64) in_=64 out=64         |  36.48 |  26.61 |   8.45 |    -76% |
| benchmark_GCNConv / edg=(2, 1000000) nod=(10, 64) in_=64 out=64        | 484.95 | 260.54 |  92.03 |    -81% |
| benchmark_GCNConv / edg=(2, 1000) nod=(100, 64) in_=64 out=64          |   0.91 |   0.59 |   0.25 |    -72% |
| benchmark_GCNConv / edg=(2, 10000) nod=(100, 64) in_=64 out=64         |   2.95 |   3.01 |   1.45 |    -50% |
| benchmark_GCNConv / edg=(2, 100000) nod=(100, 64) in_=64 out=64        |   7.21 |  26.83 |   7.79 |     +8% |
| benchmark_GCNConv / edg=(2, 1000000) nod=(100, 64) in_=64 out=64       |  91.51 | 265.07 |  92.93 |     +1% |
| benchmark_GCNConv / edg=(2, 1000) nod=(1000, 64) in_=64 out=64         |   1.81 |   0.61 |   0.40 |    -77% |
| benchmark_GCNConv / edg=(2, 10000) nod=(1000, 64) in_=64 out=64        |   1.18 |   2.98 |   1.38 |    +16% |
| benchmark_GCNConv / edg=(2, 100000) nod=(1000, 64) in_=64 out=64       |   2.80 |  26.70 |   8.89 |   +217% |
| benchmark_GCNConv / edg=(2, 1000000) nod=(1000, 64) in_=64 out=64      |  16.85 | 262.76 |  91.60 |   +443% |
| benchmark_GCNConv / edg=(2, 1000) nod=(10000, 64) in_=64 out=64        |   2.01 |   1.03 |   1.75 |    -13% |
| benchmark_GCNConv / edg=(2, 10000) nod=(10000, 64) in_=64 out=64       |   1.29 |   3.50 |   3.35 |   +159% |
| benchmark_GCNConv / edg=(2, 100000) nod=(10000, 64) in_=64 out=64      |   1.93 |  27.59 |  10.01 |   +417% |
| benchmark_GCNConv / edg=(2, 1000000) nod=(10000, 64) in_=64 out=64     |  12.37 | 273.68 |  92.33 |   +646% |

Average benchmark:
| Operation                    | mlx_gpu | mlx_cpu | pyg_cpu | mlx_gpu/pyg_cpu speedup |
|------------------------------|-------|-------|-------|-----------------------|
| benchmark_gather         |   2.76 |   8.52 |   9.82 |   +255% |
| benchmark_gather_batch   |  19.94 |  48.19 |  62.78 |   +214% |
| benchmark_scatter        |   7.97 |  73.10 |  26.36 |   +230% |
| benchmark_scatter_batch  | 190.52 | 624.64 | 221.32 |    +16% |
| benchmark_GATConv        |  45.66 | 102.33 |  28.41 |    -37% |
| benchmark_GCNConv        |  41.80 |  74.06 |  25.88 |    -38% |
