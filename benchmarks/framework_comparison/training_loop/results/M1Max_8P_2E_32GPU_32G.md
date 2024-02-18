| Dataset | Framework | Layer | Time/epoch |
|---------|-----------|-------|------------|
| BZR_MD | dgl | GCNConv | 0.130s |
| BZR_MD | dgl | GATConv | 0.340s |
| BZR_MD | pyg | GCNConv | 0.241s |
| BZR_MD | pyg | GATConv | 0.341s |
| BZR_MD | mxg (compiled) | GCNConv | 0.024s |
| BZR_MD | mxg (compiled) | GATConv | 0.113s |
| MUTAG | dgl | GCNConv | 0.036s |
| MUTAG | dgl | GATConv | 0.063s |
| MUTAG | pyg | GCNConv | 0.052s |
| MUTAG | pyg | GATConv | 0.080s |
| MUTAG | mxg (compiled) | GCNConv | 0.016s |
| MUTAG | mxg (compiled) | GATConv | 0.036s |
| DD | dgl | GCNConv | 2.243s |
| DD | dgl | GATConv | 4.838s |
| DD | pyg | GCNConv | 4.176s |
| DD | pyg | GATConv | 5.380s |
| DD | mxg (compiled) | GCNConv | 0.743s |
| DD | mxg (compiled) | GATConv | 1.535s |
