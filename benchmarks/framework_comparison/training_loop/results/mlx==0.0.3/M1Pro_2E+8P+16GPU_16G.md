Platform macOS-14.2

mlx version: 0.4.0

mlx-graphs version: 0.0.2

torch version: 2.2.1

torch_geometric version: 2.5.0

dgl version: 2.0.0

| Dataset | Framework | Layer | Time/epoch |
| --- | --- | --- | --- |
| BZR_MD | dgl | GCNConv | 0.061s |
| BZR_MD | dgl | GATConv | 0.166s |
| BZR_MD | pyg | GCNConv | 0.130s |
| BZR_MD | pyg | GATConv | 0.186s |
| BZR_MD | mxg | GCNConv | 0.074s |
| BZR_MD | mxg | GATConv | 0.090s |
| BZR_MD | mxg(compiled) | GCNConv | 0.057s |
| BZR_MD | mxg(compiled) | GATConv | 0.071s |
| MUTAG | dgl | GCNConv | 0.015s |
| MUTAG | dgl | GATConv | 0.027s |
| MUTAG | pyg | GCNConv | 0.029s |
| MUTAG | pyg | GATConv | 0.039s |
| MUTAG | mxg | GCNConv | 0.022s |
| MUTAG | mxg | GATConv | 0.026s |
| MUTAG | mxg(compiled) | GCNConv | 0.012s |
| MUTAG | mxg(compiled) | GATConv | 0.015s |
| DD | dgl | GCNConv | 1.159s |
| DD | dgl | GATConv | 2.553s |
| DD | pyg | GCNConv | 2.491s |
| DD | pyg | GATConv | 3.249s |
| DD | mxg | GCNConv | 1.239s |
| DD | mxg | GATConv | 1.249s |
| DD | mxg(compiled) | GCNConv | 0.822s |
| DD | mxg(compiled) | GATConv | 0.999s |
| NCI-H23 | dgl | GCNConv | 7.013s |
| NCI-H23 | dgl | GATConv | 11.468s |
| NCI-H23 | pyg | GCNConv | 11.412s |
| NCI-H23 | pyg | GATConv | 12.725s |
| NCI-H23 | mxg | GCNConv | 5.295s |
| NCI-H23 | mxg | GATConv | 7.121s |
| NCI-H23 | mxg(compiled) | GCNConv | 10.848s |
| NCI-H23 | mxg(compiled) | GATConv | 11.697s |
