
Platform macOS-14.3.1

mlx version: 0.4.0

mlx-graphs version: 0.0.3

torch version: 2.2.0

torch_geometric version: 2.5.0

dgl version: 2.0.0

| Dataset | Framework | Layer | Time/epoch |
| --- | --- | --- | --- |
| BZR_MD | dgl | GCNConv | 0.047s |
| BZR_MD | dgl | GATConv | 0.126s |
| BZR_MD | pyg | GCNConv | 0.067s |
| BZR_MD | pyg | GATConv | 0.100s |
| BZR_MD | mxg | GCNConv | 0.031s |
| BZR_MD | mxg | GATConv | 0.034s |
| BZR_MD | mxg(compiled) | GCNConv | 0.018s |
| BZR_MD | mxg(compiled) | GATConv | 0.024s |
| MUTAG | dgl | GCNConv | 0.013s |
| MUTAG | dgl | GATConv | 0.025s |
| MUTAG | pyg | GCNConv | 0.020s |
| MUTAG | pyg | GATConv | 0.030s |
| MUTAG | mxg | GCNConv | 0.012s |
| MUTAG | mxg | GATConv | 0.014s |
| MUTAG | mxg(compiled) | GCNConv | 0.007s |
| MUTAG | mxg(compiled) | GATConv | 0.008s |
| DD | dgl | GCNConv | 0.706s |
| DD | dgl | GATConv | 1.752s |
| DD | pyg | GCNConv | 1.885s |
| DD | pyg | GATConv | 2.395s |
| DD | mxg | GCNConv | 0.482s |
| DD | mxg | GATConv | 0.695s |
| DD | mxg(compiled) | GCNConv | 0.261s |
| DD | mxg(compiled) | GATConv | 0.329s |
| NCI-H23 | dgl | GCNConv | 4.390s |
| NCI-H23 | dgl | GATConv | 7.475s |
| NCI-H23 | pyg | GCNConv | 7.080s |
| NCI-H23 | pyg | GATConv | 9.835s |
| NCI-H23 | mxg | GCNConv | 3.384s |
| NCI-H23 | mxg | GATConv | 4.226s |
| NCI-H23 | mxg(compiled) | GCNConv | 5.593s |
| NCI-H23 | mxg(compiled) | GATConv | 5.972s |