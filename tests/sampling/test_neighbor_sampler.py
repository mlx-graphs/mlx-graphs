import numpy as np

from mlx_graphs.sampling.neighbor_sampler import sample_nodes


def test_sample_nodes_one_hop():
    edge_index_numpy = np.array(
        [
            [0, 0, 0, 1, 1, 2, 2, 3, 3, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 1, 1, 2, 2, 3, 3],
        ]
    )

    sampled_edge, n_id, e_id, input_node = sample_nodes(
        edge_index=edge_index_numpy, num_neighbors=[2], input_node=0
    )

    assert sampled_edge.shape == (2, 2), "sampled edges shape is incorrect"
    assert len(n_id) == 3, "Incorrect reference to sampled nodes"
    assert len(e_id) == 2, "Incorrect reference to sampled edges"
    assert input_node == 0, "Incorrect reference to input nodes"


def test_sample_nodes_two_hop():
    edge_index_numpy = np.array(
        [
            [0, 0, 0, 1, 1, 2, 2, 3, 3, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 1, 1, 2, 2, 3, 3],
        ]
    )

    sampled_edge, n_id, e_id, input_node = sample_nodes(
        edge_index=edge_index_numpy, num_neighbors=[3, 2], input_node=0
    )

    assert sampled_edge.shape == (2, 9), "sampled edges shape is incorrect"
    assert len(n_id) == 10, "Incorrect reference to sampled nodes"
    assert len(e_id) == 9, "Incorrect reference to sampled edges"
    assert input_node == 0, "Incorrect reference to input nodes"


def test_sample_nodes_edge_cases():
    edge_index_numpy = np.array(
        [
            [0, 0, 0, 1, 1, 2, 2, 3, 3, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 1, 1, 2, 2, 3, 3],
        ]
    )
    sampled_edge, n_id, e_id, input_node = sample_nodes(
        edge_index=edge_index_numpy, num_neighbors=[0], input_node=0
    )

    assert np.array_equal(sampled_edge, np.array([]))
    assert np.array_equal(n_id, np.array([0]))
    assert np.array_equal(e_id, np.array([]))
    assert input_node == 0

    edge_index_numpy = np.array(
        [
            [0, 0, 0, 1, 1, 2, 2, 3, 3, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 1, 1, 2, 2, 3, 3],
        ]
    )
    sampled_edge, n_id, e_id, input_node = sample_nodes(
        edge_index=edge_index_numpy, num_neighbors=[-1], input_node=0
    )

    assert np.array_equal(sampled_edge, np.array([]))
    assert np.array_equal(n_id, np.array([0]))
    assert np.array_equal(e_id, np.array([]))
    assert input_node == 0

    edge_index_numpy = np.array(
        [
            [0, 0, 0, 1, 1, 2, 2, 3, 3, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 1, 1, 2, 2, 3, 3],
        ]
    )
    sampled_edge, n_id, e_id, input_node = sample_nodes(
        edge_index=edge_index_numpy, num_neighbors=[1], input_node=100
    )

    assert np.array_equal(sampled_edge, np.array([]))
    assert np.array_equal(n_id, np.array([100]))
    assert np.array_equal(e_id, np.array([]))
    assert input_node == 100

    edge_index_numpy = np.array(
        [
            [0, 0, 0, 1, 1, 2, 2, 3, 3, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 1, 1, 2, 2, 3, 3],
        ]
    )
    sampled_edge, n_id, e_id, input_node = sample_nodes(
        edge_index=edge_index_numpy, num_neighbors=[100], input_node=0
    )

    assert sampled_edge.shape == (2, 3)
    assert n_id == [0, 1, 2, 3]
    assert len(e_id) == 3
    assert input_node == 0
