from typing import List

import numpy as np

# import random


def sample_nodes(
    edge_index: np.ndarray,
    num_neighbors: List[int],
    batch_size: int,
    input_nodes: List[int] = None,
):
    all_nodes = np.unique(edge_index)
    if input_nodes is None or len(input_nodes) != batch_size:
        input_nodes = np.random.choice(all_nodes, size=batch_size, replace=False)
    input_id = input_nodes

    sampled_edges = []
    n_id = set(input_id)
    e_id = []

    edge_list = list(map(tuple, edge_index.T))

    for num in num_neighbors:
        current_edges = []
        for node in list(n_id):
            # find edges starting from current node
            indices = np.where(edge_index[0] == node)[0]
            if len(indices) > num:
                indices = np.random.choice(indices, size=num, replace=False)
            current_edges.extend(edge_index[:, indices].T)

        n_id.update(np.array(current_edges)[:, 1])
        sampled_edges.extend(current_edges)

        # Update e_id with indices of current_edges in edge_list
        current_edges_tuples = [tuple(edge) for edge in current_edges]
        e_id.extend([edge_list.index(edge) for edge in current_edges_tuples])

    n_id = list(n_id)
    input_id = list(input_id)

    sampled_edges = np.array(sampled_edges)
    unique_edges = np.unique(sampled_edges, axis=0)

    return unique_edges.tolist(), n_id, e_id, input_id
