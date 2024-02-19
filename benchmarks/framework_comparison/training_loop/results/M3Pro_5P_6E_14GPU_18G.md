Platform macOS-14.2.1

mlx version: 0.3.0

mlx-graphs version: 0.0.2

torch version: 2.2.0

torch_geometric version: 2.5.0

| Dataset | Framework | Layer | Time/epoch |
|---------|-----------|-------|------------|
| BZR_MD | dgl | GCNConv | 0.042s |
| BZR_MD | dgl | GATConv | 0.118s |
| BZR_MD | pyg | GCNConv | 0.078s |
| BZR_MD | pyg | GATConv | 0.119s |
| BZR_MD | mxg | GCNConv | 0.046s |
| BZR_MD | mxg | GATConv | 0.058s |
| BZR_MD | mxg(compiled) | GCNConv | 0.042s |
| BZR_MD | mxg(compiled) | GATConv | 0.058s |
| MUTAG | dgl | GCNConv | 0.010s |
| MUTAG | dgl | GATConv | 0.018s |
| MUTAG | pyg | GCNConv | 0.016s |
| MUTAG | pyg | GATConv | 0.022s |
| MUTAG | mxg | GCNConv | 0.012s |
| MUTAG | mxg | GATConv | 0.014s |
| MUTAG | mxg(compiled) | GCNConv | 0.007s |
| MUTAG | mxg(compiled) | GATConv | 0.009s |
| DD | dgl | GCNConv | 0.822s |
| DD | dgl | GATConv | 1.803s |
| DD | pyg | GCNConv | 1.454s |
| DD | pyg | GATConv | 1.968s |
| DD | mxg | GCNConv | 0.584s |
| DD | mxg | GATConv | 0.810s |
| DD | mxg(compiled) | GCNConv | 0.690s |
| DD | mxg(compiled) | GATConv | 0.920s |
| NCI-H23 | dgl | GCNConv | 3.104s |
| NCI-H23 | dgl | GATConv | 5.500s |
| NCI-H23 | pyg | GCNConv | 4.998s |
| NCI-H23 | pyg | GATConv | 7.116s |
| NCI-H23 | mxg | GCNConv | 3.289s |
| NCI-H23 | mxg | GATConv | 4.325s |
| NCI-H23 | mxg(compiled) | GCNConv | 12.550s |
| NCI-H23 | mxg(compiled) | GATConv | 13.134s |
