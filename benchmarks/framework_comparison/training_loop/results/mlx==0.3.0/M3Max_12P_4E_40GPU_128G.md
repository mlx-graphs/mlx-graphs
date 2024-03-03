
Platform macOS-14.3.1

mlx version: 0.3.0

mlx-graphs version: 0.0.3

torch version: 2.2.0

torch_geometric version: 2.5.0

dgl version: 2.0.0

| Dataset | Framework | Layer | Time/epoch |
| --- | --- | --- | --- |
| BZR_MD | dgl | GCNConv | 0.047s |
| BZR_MD | dgl | GATConv | 0.125s |
| BZR_MD | pyg | GCNConv | 0.065s |
| BZR_MD | pyg | GATConv | 0.098s |
| BZR_MD | mxg | GCNConv | 0.034s |
| BZR_MD | mxg | GATConv | 0.039s |
| BZR_MD | mxg(compiled) | GCNConv | 0.023s |
| BZR_MD | mxg(compiled) | GATConv | 0.030s |
| MUTAG | dgl | GCNConv | 0.012s |
| MUTAG | dgl | GATConv | 0.026s |
| MUTAG | pyg | GCNConv | 0.021s |
| MUTAG | pyg | GATConv | 0.030s |
| MUTAG | mxg | GCNConv | 0.012s |
| MUTAG | mxg | GATConv | 0.015s |
| MUTAG | mxg(compiled) | GCNConv | 0.007s |
| MUTAG | mxg(compiled) | GATConv | 0.008s |
| DD | dgl | GCNConv | 0.702s |
| DD | dgl | GATConv | 1.724s |
| DD | pyg | GCNConv | 1.864s |
| DD | pyg | GATConv | 2.389s |
| DD | mxg | GCNConv | 0.524s |
| DD | mxg | GATConv | 0.614s |
| DD | mxg(compiled) | GCNConv | 0.313s |
| DD | mxg(compiled) | GATConv | 0.392s |
| NCI-H23 | dgl | GCNConv | 4.252s |
| NCI-H23 | dgl | GATConv | 7.366s |
| NCI-H23 | pyg | GCNConv | 6.828s |
| NCI-H23 | pyg | GATConv | 9.779s |
| NCI-H23 | mxg | GCNConv | 3.344s |
| NCI-H23 | mxg | GATConv | 4.319s |
| NCI-H23 | mxg(compiled) | GCNConv | 5.673s |
| NCI-H23 | mxg(compiled) | GATConv | 6.110s |
