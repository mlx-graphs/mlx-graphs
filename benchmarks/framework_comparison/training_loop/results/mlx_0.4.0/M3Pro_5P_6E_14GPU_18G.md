Platform macOS-14.2.1

mlx version: 0.4.0

mlx-graphs version: 0.0.3

torch version: 2.2.0

torch_geometric version: 2.5.0

dgl version: 2.0.0

| Dataset | Framework | Layer | Time/epoch |
| --- | --- | --- | --- |
| BZR_MD | dgl | GCNConv | 0.042s |
| BZR_MD | dgl | GATConv | 0.117s |
| BZR_MD | pyg | GCNConv | 0.077s |
| BZR_MD | pyg | GATConv | 0.115s |
| BZR_MD | mxg | GCNConv | 0.037s |
| BZR_MD | mxg | GATConv | 0.048s |
| BZR_MD | mxg(compiled) | GCNConv | 0.031s |
| BZR_MD | mxg(compiled) | GATConv | 0.043s |
| MUTAG | dgl | GCNConv | 0.011s |
| MUTAG | dgl | GATConv | 0.021s |
| MUTAG | pyg | GCNConv | 0.018s |
| MUTAG | pyg | GATConv | 0.024s |
| MUTAG | mxg | GCNConv | 0.012s |
| MUTAG | mxg | GATConv | 0.015s |
| MUTAG | mxg(compiled) | GCNConv | 0.007s |
| MUTAG | mxg(compiled) | GATConv | 0.009s |
| DD | dgl | GCNConv | 0.819s |
| DD | dgl | GATConv | 1.801s |
| DD | pyg | GCNConv | 1.450s |
| DD | pyg | GATConv | 1.978s |
| DD | mxg | GCNConv | 0.490s |
| DD | mxg | GATConv | 0.691s |
| DD | mxg(compiled) | GCNConv | 0.543s |
| DD | mxg(compiled) | GATConv | 0.736s |
| NCI-H23 | dgl | GCNConv | 2.938s |
| NCI-H23 | dgl | GATConv | 5.381s |
| NCI-H23 | pyg | GCNConv | 4.804s |
| NCI-H23 | pyg | GATConv | 6.898s |
| NCI-H23 | mxg | GCNConv | 3.384s |
| NCI-H23 | mxg | GATConv | 4.384s |
| NCI-H23 | mxg(compiled) | GCNConv | 12.538s |
| NCI-H23 | mxg(compiled) | GATConv | 13.026s |
