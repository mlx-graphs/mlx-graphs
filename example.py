import timeit

import mlx.core as mx

from mlx_graphs.utils import remove_self_loops

edge_index_1 = mx.ones((2, 1_000_000))
edge_index_2 = mx.array([[0, 0, 1, 1], [0, 1, 0, 2]])
edge_index = mx.concatenate([edge_index_1, edge_index_2], 1)

times = timeit.Timer(lambda: remove_self_loops(edge_index)).repeat(repeat=10, number=1)

print(min(times))
