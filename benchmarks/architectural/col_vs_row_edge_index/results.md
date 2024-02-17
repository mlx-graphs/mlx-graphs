
| operation | [N, 2] index | [2, N] index | [N, 2] speedup |
| --- | --- | --- | --- |
| get_src_dst_features | 5.41μs | 3.20μs | -69.03% |
| sort_edge_index | 6.83μs | 7.65μs | 10.73% |
| sort_edge_index_and_features | 7.61μs | 8.46μs | 10.00% |
| to_edge_index | 44.94μs | 45.01μs | 0.15% |
| to_sparse_adjacency_matrix | 48.48μs | 51.43μs | 5.73% |
| to_adjacency_matrix | 108.09μs | 94.16μs | -14.80% |
| is_undirected | 289.07μs | 253.21μs | -14.16% |
