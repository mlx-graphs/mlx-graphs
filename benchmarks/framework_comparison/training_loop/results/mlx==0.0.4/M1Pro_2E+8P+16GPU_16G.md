Platform macOS-14.2

mlx version: 0.4.0

mlx-graphs version: 0.0.2

torch version: 2.1.2

torch_geometric version: 2.4.0

dgl version: 2.0.0

| Dataset | Framework | Layer | Time/epoch |
| --- | --- | --- | --- |
| BZR_MD | dgl | GCNConv | 0.068s |
| BZR_MD | dgl | GATConv | 0.170s |
| BZR_MD | pyg | GCNConv | 0.121s |
| BZR_MD | pyg | GATConv | 0.170s |
| BZR_MD | mxg | GCNConv | 0.081s |
| BZR_MD | mxg | GATConv | 0.095s |
| BZR_MD | mxg(compiled) | GCNConv | 0.058s |
| BZR_MD | mxg(compiled) | GATConv | 0.077s |
| MUTAG | dgl | GCNConv | 0.019s |
| MUTAG | dgl | GATConv | 0.037s |
| MUTAG | pyg | GCNConv | 0.026s |
| MUTAG | pyg | GATConv | 0.036s |
| MUTAG | mxg | GCNConv | 0.021s |
| MUTAG | mxg | GATConv | 0.030s |
| MUTAG | mxg(compiled) | GCNConv | 0.013s |
| MUTAG | mxg(compiled) | GATConv | 0.016s |
| DD | dgl | GCNConv | 1.093s |
| DD | dgl | GATConv | 2.456s |
| DD | pyg | GCNConv | 2.098s |
| DD | pyg | GATConv | 2.754s |
| DD | mxg | GCNConv | 0.892s |
| DD | mxg | GATConv | 1.086s |
| DD | mxg(compiled) | GCNConv | 0.806s |
| DD | mxg(compiled) | GATConv | 0.993s |
| NCI-H23 | dgl | GCNConv | 5.162s |
| NCI-H23 | dgl | GATConv | 8.909s |
| NCI-H23 | pyg | GCNConv | 7.906s |
| NCI-H23 | pyg | GATConv | 11.841s |
| NCI-H23 | mxg | GCNConv | 5.338s |
| NCI-H23 | mxg | GATConv | 7.239s |
| NCI-H23 | mxg(compiled) | GCNConv | 10.723s |
| NCI-H23 | mxg(compiled) | GATConv | 11.627s |
