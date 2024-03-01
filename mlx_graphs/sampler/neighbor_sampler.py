from typing import List, Tuple

import numpy as np


def sample_nodes(
    edge_index: np.ndarray,
    num_neighbors: List[int],
    batch_size: int,
    input_nodes: np.ndarray = None,
) -> Tuple[np.array, np.array, np.array, np.array]:
    r"""GraphSage neighbor sampler implementation.


    Args:
        edge_index : the edge index representing the original graph
        to sample from nodes.
        num_neighbors : number of neighbors to sample for each hop.
        batch_size : number of seed nodes to take into consideration
        to build a computational graph for `batch_size` number of seed nodes.
        input_nodes : seed_nodes to take into consideration,
        if not provided `batch_size` number of seed nodes will be randomly sampled.

    Returns:
        sampled_edges_array : the sampled edge_index.
        n_id_list : a reference to the sampled nodes.
        e_id : a reference to the indices of sampled edges.
        input_nodes : a reference to the seed nodes.

    """

    if not isinstance(num_neighbors, list):
        try:
            num_neighbors = list(num_neighbors)
        except TypeError:
            raise ValueError("num_neighbors must be a list.")

    if not num_neighbors:
        raise ValueError("num_neighbors cannot be an empty list.")

    all_nodes = np.unique(edge_index)
    if input_nodes is None or len(input_nodes) != batch_size:
        input_nodes = np.random.choice(all_nodes, size=batch_size, replace=False)

    sampled_edges = []
    sampled_edges_indices = []
    # Will include all unique nodes encountered
    n_id = set(input_nodes)
    e_id = []
    visited_nodes = set(input_nodes)
    current_layer_nodes = input_nodes

    structured_edge_index = np.core.records.fromarrays(
        edge_index, names="source, target", formats="i8, i8"
    )

    for num in num_neighbors:
        # to collect unique nodes found in this layer
        next_layer_nodes = set()

        for node in current_layer_nodes:
            # Filter edges starting from current node, avoiding revisited nodes
            mask = (structured_edge_index.source == node) & (
                ~np.isin(structured_edge_index.target, list(visited_nodes))
            )
            possible_edges_masked_indices = np.where(mask)[0]
            possible_edges = structured_edge_index[mask]

            num_edges_to_select = min(len(possible_edges), num)
            if num_edges_to_select > 0:
                selected_indices = np.random.choice(
                    len(possible_edges), size=num_edges_to_select, replace=False
                )
                selected_edges = possible_edges[selected_indices]

                original_indices = possible_edges_masked_indices[selected_indices]

                next_layer_nodes.update(selected_edges["target"])
                sampled_edges.extend(selected_edges.tolist())
                sampled_edges_indices.extend(original_indices.tolist())

                visited_nodes.update(selected_edges["target"])

        # nodes to process in next layer
        current_layer_nodes = np.array(list(next_layer_nodes - n_id), dtype=int)
        n_id.update(next_layer_nodes)

    sampled_edges_array = np.array(sampled_edges, dtype=int).T
    e_id = np.array(sampled_edges_indices)

    n_id_list = np.array(list(n_id))  # Convert n_id to a list for the output

    return sampled_edges_array, n_id_list, e_id, input_nodes
