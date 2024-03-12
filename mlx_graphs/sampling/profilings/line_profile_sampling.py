import logging
import time

import mlx.core as mx
from line_profiler import LineProfiler

from mlx_graphs.datasets import PlanetoidDataset
from mlx_graphs.sampling.neighbor_sampler import sample_nodes, sampler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


DATASET = "Cora"
INPUT_NODES = 2
NUM_NEIGHBORS = [10, 2]
BATCH_SIZE = 1
dataset = PlanetoidDataset(DATASET)
graph = dataset[0]
flat_edge_index = graph.edge_index.flatten()
input_nodes = [
    mx.random.randint(low=1, high=flat_edge_index.shape[0] + 1, shape=(1,)).item()
    for _ in range(INPUT_NODES)
]


num_nodes = graph.num_nodes
num_edges = graph.edge_index.shape[1]


def main():
    start_time = time.time()

    sampler(graph, input_nodes, NUM_NEIGHBORS, BATCH_SIZE)

    end_time = time.time()

    total_time = end_time - start_time
    logger.info(f"Sampler execution time: {total_time} seconds")


if __name__ == "__main__":
    lp = LineProfiler()

    lp.add_function(sampler)

    lp.add_function(sample_nodes)

    lp_wrapper = lp(main)

    lp_wrapper()

    lp.print_stats()
