
| operation | [N, 2] index | [2, N] index | [N, 2] speedup |
| --- | --- | --- | --- |
| get_src_dst_features | 5.89μs | 3.45μs | -70.48% |
| sort_edge_index | 7.35μs | 8.21μs | 10.48% |
| sort_edge_index_and_features | 8.24μs | 9.10μs | 9.42% |
| to_edge_index | 49.65μs | 49.66μs | 0.01% |
| to_sparse_adjacency_matrix | 52.00μs | 55.37μs | 6.08% |
| to_adjacency_matrix | 138.23μs | 114.80μs | -20.41% |
| is_undirected | 297.11μs | 262.34μs | -13.25% |
