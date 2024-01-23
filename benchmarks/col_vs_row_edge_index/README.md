# `[2, N]` VS `[N, 2]` edge index

In this benchmark we compare the performance of different functions when operating on an `edge_index` in row-format (i.e., with shape `[2, num_edges]`, where the first row contains the indices of the source nodes and the second row the ones of the destination nodes of each edge) or in col-format (i.e., with shape `[num_edges, 2]`, where source and destination indices are stored in the two columns).

To execute, simply run `python launcher.py`. The results are saved in `results.md`.
