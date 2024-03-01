import numpy as np
import pytest

from mlx_graphs.sampler.neighbor_sampler import sample_nodes


def test_sample_nodes_no_hop():
    edge_index_numpy = np.array(
        [
            [0, 0, 0, 1, 1, 2, 2, 3, 3, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 1, 1, 2, 2, 3, 3],
        ]
    )
    with pytest.raises(ValueError):
        sampled_edge, n_id, e_id, input_nodes = sample_nodes(
            edge_index=edge_index_numpy, num_neighbors=[], batch_size=1, input_nodes=[0]
        )


def test_sample_nodes_one_hop():
    edge_index_numpy = np.array(
        [
            [0, 0, 0, 1, 1, 2, 2, 3, 3, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 1, 1, 2, 2, 3, 3],
        ]
    )

    sampled_edge, n_id, e_id, input_nodes = sample_nodes(
        edge_index=edge_index_numpy, num_neighbors=[2], batch_size=1, input_nodes=[0]
    )

    assert sampled_edge.shape == (2, 2), "sampled edges shape is incorrect"
    assert n_id.shape == (3,), "Incorrect reference to sampled nodes"
    assert e_id.shape == (2,), "Incorrect reference to sampled edges"
    assert input_nodes == [0], "Incorrect reference to input nodes"


def test_sample_nodes_two_hop():
    edge_index_numpy = np.array(
        [
            [0, 0, 0, 1, 1, 2, 2, 3, 3, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 1, 1, 2, 2, 3, 3],
        ]
    )

    sampled_edge, n_id, e_id, input_nodes = sample_nodes(
        edge_index=edge_index_numpy, num_neighbors=[3, 2], batch_size=1, input_nodes=[0]
    )

    assert sampled_edge.shape == (2, 9), "sampled edges shape is incorrect"
    assert n_id.shape == (10,), "Incorrect reference to sampled nodes"
    assert e_id.shape == (9,), "Incorrect reference to sampled edges"
    assert input_nodes == [0], "Incorrect reference to input nodes"
