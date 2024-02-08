import timeit

import mlx.core as mx
import scipy as sp
from col_index_functions import (
    get_src_dst_features_COL,
    is_undirected_COL,
    sort_edge_index_and_features_COL,
    sort_edge_index_COL,
    to_adjacency_matrix_COL,
    to_edge_index_COL,
    to_sparse_adjacency_matrix_COL,
)
from row_index_functions import (
    get_src_dst_features_ROW,
    is_undirected_ROW,
    sort_edge_index_and_features_ROW,
    sort_edge_index_ROW,
    to_adjacency_matrix_ROW,
    to_edge_index_ROW,
    to_sparse_adjacency_matrix_ROW,
)
from utils import to_markdown_table

N = 1000

adj_matrix = mx.array(sp.sparse.random(100, 100, 0.1).todense())
edge_index_COL = mx.random.randint(0, N, [N, 2])
edge_index_ROW = edge_index_COL.transpose()
features = mx.random.normal([N, 100])


# warmup
_ = timeit.Timer(lambda: sort_edge_index_COL(edge_index_COL)).timeit(number=100000)

# functions to test
comparison_list = [
    (
        "get_src_dst_features",
        get_src_dst_features_COL,
        [edge_index_COL, features],
        get_src_dst_features_ROW,
        [edge_index_ROW, features],
    ),
    (
        "sort_edge_index",
        sort_edge_index_COL,
        [edge_index_COL],
        sort_edge_index_ROW,
        [edge_index_ROW],
    ),
    (
        "sort_edge_index_and_features",
        sort_edge_index_and_features_COL,
        [edge_index_COL, features],
        sort_edge_index_and_features_ROW,
        [edge_index_ROW, features],
    ),
    ("to_edge_index", to_edge_index_COL, [adj_matrix], to_edge_index_ROW, [adj_matrix]),
    (
        "to_sparse_adjacency_matrix",
        to_sparse_adjacency_matrix_COL,
        [adj_matrix],
        to_sparse_adjacency_matrix_ROW,
        [adj_matrix],
    ),
    (
        "to_adjacency_matrix",
        to_adjacency_matrix_COL,
        [edge_index_COL],
        to_adjacency_matrix_ROW,
        [edge_index_ROW],
    ),
    (
        "is_undirected",
        is_undirected_COL,
        [edge_index_COL],
        is_undirected_ROW,
        [edge_index_ROW],
    ),
]

# performance test
tab = [["operation", "[N, 2] index", "[2, N] index", "[N, 2] speedup"]]
number_iters = 1000
number_repeat = 100
for name, f_N2, arg_N2, f_2N, arg_2N in comparison_list:
    time_N2 = min(
        timeit.Timer(lambda: f_N2(*arg_N2)).repeat(
            repeat=number_repeat, number=number_iters
        )
    )
    time_2N = min(
        timeit.Timer(lambda: f_2N(*arg_2N)).repeat(
            repeat=number_repeat, number=number_iters
        )
    )
    tab.append(
        [
            name,
            "{:.2f}\u03BCs".format(time_N2 * 1000000 / number_iters),
            "{:.2f}\u03BCs".format(time_2N * 1000000 / number_iters),
            "{:.2f}%".format(-(time_N2 - time_2N) / time_2N * 100),
        ]
    )
md_tab = to_markdown_table(tab)
with open("results.md", "w") as f:
    f.write(md_tab)
