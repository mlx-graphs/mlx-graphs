Platform macOS-14.2

mlx version: 0.3.0

mlx-graphs version: 0.0.2

torch version: 2.1.2

torch_geometric version: 2.4.0

dgl version: 2.0.0

| Dataset | Framework | Layer | Time/epoch |
| --- | --- | --- | --- |
| BZR_MD | dgl | GCNConv | 0.066s |
| BZR_MD | dgl | GATConv | 0.174s |
| BZR_MD | pyg | GCNConv | 0.130s |
| BZR_MD | pyg | GATConv | 0.175s |
| BZR_MD | mxg | GCNConv | 0.101s |
| BZR_MD | mxg | GATConv | 0.113s |
| BZR_MD | mxg(compiled) | GCNConv | 0.062s |
| BZR_MD | mxg(compiled) | GATConv | 0.083s |
| MUTAG | dgl | GCNConv | 0.017s |
| MUTAG | dgl | GATConv | 0.032s |
| MUTAG | pyg | GCNConv | 0.025s |
| MUTAG | pyg | GATConv | 0.037s |
| MUTAG | mxg | GCNConv | 0.024s |
| MUTAG | mxg | GATConv | 0.031s |
| MUTAG | mxg(compiled) | GCNConv | 0.013s |
| MUTAG | mxg(compiled) | GATConv | 0.017s |
| DD | dgl | GCNConv | 1.079s |
| DD | dgl | GATConv | 2.486s |
| DD | pyg | GCNConv | 2.186s |
| DD | pyg | GATConv | 2.889s |
| DD | mxg | GCNConv | 1.413s |
| DD | mxg | GATConv | 1.481s |
| DD | mxg(compiled) | GCNConv | 1.002s |
| DD | mxg(compiled) | GATConv | 1.235s |
| NCI-H23 | dgl | GCNConv | 5.046s |
| NCI-H23 | dgl | GATConv | 8.803s |
| NCI-H23 | pyg | GCNConv | 7.253s |
| NCI-H23 | pyg | GATConv | 10.505s |
| NCI-H23 | mxg | GCNConv | 5.585s |
| NCI-H23 | mxg | GATConv | 7.601s |
| NCI-H23 | mxg(compiled) | GCNConv | 11.256s |
| NCI-H23 | mxg(compiled) | GATConv | 12.603s |
