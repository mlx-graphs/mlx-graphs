from typing import List, Tuple

import mlx.core as mx
import numpy as np

from mlx_graphs.data import GraphData


def sample_nodes(
    edge_index: np.ndarray,
    num_neighbors: List[int],
    input_node: int = None,
) -> Tuple[np.array, np.array, np.array, np.array]:
    r"""GraphSage neighbor sampler implementation.


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

    if not isinstance(num_neighbors, list):
        try:
            num_neighbors = list(num_neighbors)
        except TypeError:
            raise ValueError("num_neighbors must be a list.")

    if not num_neighbors:
        raise ValueError("num_neighbors cannot be an empty list.")

    sampled_edges = []
    sampled_edges_indices = []
    # Will include all unique nodes encountered
    n_id = set([input_node])
    e_id = []
    visited_nodes = set([input_node])
    current_layer_nodes = [input_node]

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
    e_id = sampled_edges_indices

    n_id_list = list(n_id)

    return sampled_edges_array, n_id_list, e_id, input_node


def sampler(
    graph: GraphData, input_nodes: List[int], num_neighbors: List[int], batch_size: int
) -> List[GraphData]:
    """Function that samples subgraphs from a graph using Neighbor Sampling strategy.

    Args:
        graph : Original `GraphData` object to sample from.
        input_nodes : seed_nodes to take into consideration.
        num_neighbors : number of neighbors to sample for each hop.
        batch_size : number of seed nodes to take into consideration
        to build a computational graph for `batch_size` number of seed nodes.

    Returns:
        list of samples for each seed node.
    """

    graphs = []
    for seed_node in input_nodes:
        sampled_edges, n_id, e_id, input_node = sample_nodes(
            edge_index=np.array(graph.edge_index),
            num_neighbors=num_neighbors,
            input_node=seed_node,
        )
        n_id = [int(id_) for id_ in n_id]
        subgraph_node_features = graph.node_features[mx.array(n_id)]
        subgraph = GraphData(
            edge_index=mx.array(sampled_edges),
            node_features=subgraph_node_features,
            n_id=mx.array(n_id),
            e_id=mx.array(e_id),
            input_nodes=mx.array(input_node),
        )
        graphs.append(subgraph)

    return graphs


def collate_subgraphs(subgraphs: List[GraphData], batch_size: int, **kwargs):
    pass
