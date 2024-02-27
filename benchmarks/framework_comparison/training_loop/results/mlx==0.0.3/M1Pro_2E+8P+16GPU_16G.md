Platform macOS-14.2

mlx version: 0.4.0

mlx-graphs version: 0.0.3

torch version: 2.2.1

torch_geometric version: 2.5.0

dgl version: 2.0.0

| Dataset | Framework | Layer | Time/epoch |
| --- | --- | --- | --- |
| BZR_MD | dgl | GCNConv | 0.061s |
| BZR_MD | dgl | GATConv | 0.168s |
| BZR_MD | pyg | GCNConv | 0.135s |
| BZR_MD | pyg | GATConv | 0.186s |
| BZR_MD | mxg | GCNConv | 0.073s |
| BZR_MD | mxg | GATConv | 0.086s |
| BZR_MD | mxg(compiled) | GCNConv | 0.057s |
| BZR_MD | mxg(compiled) | GATConv | 0.072s |
| MUTAG | dgl | GCNConv | 0.014s |
| MUTAG | dgl | GATConv | 0.027s |
| MUTAG | pyg | GCNConv | 0.028s |
| MUTAG | pyg | GATConv | 0.037s |
| MUTAG | mxg | GCNConv | 0.022s |
| MUTAG | mxg | GATConv | 0.028s |
| MUTAG | mxg(compiled) | GCNConv | 0.012s |
| MUTAG | mxg(compiled) | GATConv | 0.016s |
| DD | dgl | GCNConv | 1.140s |
| DD | dgl | GATConv | 2.562s |
| DD | pyg | GCNConv | 2.616s |
| DD | pyg | GATConv | 3.270s |
| DD | mxg | GCNConv | 1.212s |
| DD | mxg | GATConv | 1.148s |
| DD | mxg(compiled) | GCNConv | 0.806s |
| DD | mxg(compiled) | GATConv | 0.995s |
| NCI-H23 | dgl | GCNConv | 5.247s |
| NCI-H23 | dgl | GATConv | 9.364s |
| NCI-H23 | pyg | GCNConv | 8.935s |
| NCI-H23 | pyg | GATConv | 11.790s |
| NCI-H23 | mxg | GCNConv | 5.541s |
| NCI-H23 | mxg | GATConv | 7.245s |
| NCI-H23 | mxg(compiled) | GCNConv | 10.973s |
| NCI-H23 | mxg(compiled) | GATConv | 11.709s |
