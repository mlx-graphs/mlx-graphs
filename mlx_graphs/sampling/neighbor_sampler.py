from typing import List, Tuple

import mlx.core as mx
import numpy as np

from mlx_graphs.data import GraphData


def sample_nodes(
    edge_index: np.ndarray, num_neighbors: List[int], input_node: int
) -> Tuple[np.array, List[int], List[int], int]:
    """GraphSage neighbor sampler implementation.

    Args:
        edge_index : the edge index representing the original graph
        to sample from nodes.
        num_neighbors : number of neighbors to sample for each hop.
        input_nodes : seed_nodes to take into consideration.

    Returns:
        sampled_edges_array : the sampled edge_index.
        n_id_list : a reference to the sampled nodes.
        e_id : a reference to the indices of sampled edges.
        input_nodes : a reference to the seed nodes.

    """
    sampled_edges = []
    sampled_edges_indices = []
    n_id = set([input_node])  # Will include all unique nodes encountered
    visited_nodes = set([input_node])
    current_layer_nodes = np.array([input_node], dtype=np.int32)

    for num in num_neighbors:
        next_layer_nodes = set()
        for node in current_layer_nodes:
            # Find indices of outgoing edges from the current node
            indices = np.where(edge_index[0] == node)[0]
            possible_targets = edge_index[1][indices]

            # Filter out already visited nodes
            targets = np.setdiff1d(
                possible_targets, np.array(list(visited_nodes), dtype=np.int32)
            )

            if len(targets) > 0:
                num_edges_to_select = min(len(targets), num)
                selected_targets_idx = np.random.choice(
                    range(len(targets)), size=num_edges_to_select, replace=False
                )
                selected_targets = targets[selected_targets_idx]
                selected_indices = indices[selected_targets_idx]

                next_layer_nodes.update(selected_targets)
                visited_nodes.update(selected_targets)
                sampled_edges_indices.extend(selected_indices.tolist())

                for target in selected_targets:
                    sampled_edges.append([node, target])

        current_layer_nodes = np.array(list(next_layer_nodes - n_id), dtype=np.int32)
        n_id.update(next_layer_nodes)

    sampled_edges_array = np.array(sampled_edges).T
    n_id_list = list(n_id)

    return sampled_edges_array, n_id_list, sampled_edges_indices, input_node


def sampler(
    graph: GraphData, input_nodes: List[int], num_neighbors: List[int], batch_size: int
) -> List[GraphData]:
    """Function that samples subgraphs from a graph using
    Neighbor Sampling strategy with batching."""

    if not isinstance(graph, GraphData):
        raise ValueError("graph must be a GraphData object")

    if not isinstance(num_neighbors, list):
        try:
            num_neighbors = list(num_neighbors)
        except TypeError:
            raise ValueError("num_neighbors must be a list.")

    if not num_neighbors:
        raise ValueError("num_neighbors cannot be an empty list.")

    if not isinstance(input_nodes, list):
        try:
            input_nodes = list(input_nodes)
        except TypeError:
            raise ValueError("input_nodes must be a list.")

    if not input_nodes:
        raise ValueError("input_nodes cannot be an empty list.")

    batched_graphs = []

    for i in range(0, len(input_nodes), batch_size):
        batch_nodes = input_nodes[i : i + batch_size]

        # lists to store aggregated values for the current batch
        batch_sampled_edges = np.empty([2, 0], dtype=int)
        batch_n_id = np.empty([1, 0], dtype=int)
        batch_e_id = []
        batch_input_node = []

        for seed_node in batch_nodes:
            sampled_edges, n_id, e_id, input_node = sample_nodes(
                edge_index=np.array(graph.edge_index),
                num_neighbors=num_neighbors,
                input_node=seed_node,
            )
            if sampled_edges.size == 0:
                sampled_edges = np.empty([2, 0], dtype=int)
            batch_sampled_edges = np.append(batch_sampled_edges, sampled_edges, axis=1)
            batch_n_id = np.append(batch_n_id, n_id)
            batch_e_id.extend(e_id)
            batch_input_node.append(input_node)

        subgraph_node_features = graph.node_features[mx.array(batch_n_id)]
        subgraph = GraphData(
            edge_index=mx.array(batch_sampled_edges),
            node_features=subgraph_node_features,
            n_id=mx.array(batch_n_id),
            e_id=mx.array(e_id),
            input_nodes=mx.array(input_node),
        )

        batched_graphs.append(subgraph)

    return batched_graphs


def collate_subgraphs(subgraphs: List[GraphData], batch_size: int, **kwargs):
    pass
