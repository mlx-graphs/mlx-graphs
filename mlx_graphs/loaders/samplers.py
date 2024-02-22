import math

import numpy as np
from dataloaders import Dataloader


class NeighborSamplingLoader(Dataloader):
    def __init__(self):
        self.nodes_per_batch: list[list[int]] = None
        self.batch_index = 0

    def __next__(self):
        ...

    def greedy_neighbor_sampling(self, graph, samples_per_hop, batch_size):
        node_to_neighbors = {}
        unique_nodes = np.unique(graph.edge_index)

        if self.nodes_per_batch is None:
            num_nodes = len(unique_nodes)
            num_batches = math.ceil(num_nodes / batch_size)
            self.nodes_per_batch = [
                unique_nodes[batch : batch + batch_size]
                for batch in list(range(0, num_batches * batch_size, batch_size))
            ]

        nodes = self.nodes_per_batch[self.batch_index]
        edge_index = []
        for node in nodes:
            for neighbor in node_to_neighbors[node]:
                for num_samples in samples_per_hop:
                    neighbor_neighborhood = node_to_neighbors[neighbor]

                    random_indices = np.random.permutation(num_samples)
                    sampled_neighbors = neighbor_neighborhood[random_indices]
                    for n in sampled_neighbors:
                        edge_index.append((neighbor, n))

                edge_index.append((node, neighbor))
                # TODO: Need to handle "recusrively"

        self.batch_index += 1
