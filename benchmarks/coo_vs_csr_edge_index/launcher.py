import mlx.core as mx
import scipy as sp
import timeit
from utils import to_markdown_table
from coo_functions import (
    get_src_dst_features_COO,
    sort_edge_index_COO,
    sort_edge_index_and_features_COO,
    to_edge_index_COO,
    to_sparse_adjacency_matrix_COO,
    to_adjacency_matrix_COO,
    is_undirected_COO,
)
from csr_functions import (
    get_src_dst_features_CSR,
    sort_edge_index_CSR,
    sort_edge_index_and_features_CSR,
    to_edge_index_CSR,
    to_sparse_adjacency_matrix_CSR,
    to_adjacency_matrix_CSR,
    is_undirected_CSR,
)

N = 1000

adj_matrix = mx.array(sp.sparse.random(100, 100, 0.1).todense())
edge_index_CSR = mx.random.randint(0, N, [N, 2])
edge_index_COO = edge_index_CSR.transpose()
features = mx.random.normal([N, 100])


# warmup
_ = timeit.Timer(lambda: sort_edge_index_CSR(edge_index_CSR)).timeit(number=100000)

# functions to test
comparison_list = [
    (
        "get_src_dst_features",
        get_src_dst_features_CSR,
        [edge_index_CSR, features],
        get_src_dst_features_COO,
        [edge_index_COO, features],
    ),
    (
        "sort_edge_index",
        sort_edge_index_CSR,
        [edge_index_CSR],
        sort_edge_index_COO,
        [edge_index_COO],
    ),
    (
        "sort_edge_index_and_features",
        sort_edge_index_and_features_CSR,
        [edge_index_CSR, features],
        sort_edge_index_and_features_COO,
        [edge_index_COO, features],
    ),
    ("to_edge_index", to_edge_index_CSR, [adj_matrix], to_edge_index_COO, [adj_matrix]),
    (
        "to_sparse_adjacency_matrix",
        to_sparse_adjacency_matrix_CSR,
        [adj_matrix],
        to_sparse_adjacency_matrix_COO,
        [adj_matrix],
    ),
    (
        "to_adjacency_matrix",
        to_adjacency_matrix_CSR,
        [edge_index_CSR],
        to_adjacency_matrix_COO,
        [edge_index_COO],
    ),
    (
        "is_undirected",
        is_undirected_CSR,
        [edge_index_CSR],
        is_undirected_COO,
        [edge_index_COO],
    ),
]

# performance test
tab = [["operation", "[N, 2] index", "[2, N] index", "[N, 2] speedup"]]
number_iters = 1000
number_repeat = 1
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
