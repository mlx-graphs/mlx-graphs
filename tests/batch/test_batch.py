import mlx.core as mx
from mlx_graphs.data.batch import batch
from mlx_graphs.data.data import GraphData
from mlx_graphs.data.collate import collate


def test_batching():
    node_features1 = mx.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 0, 1, 1], [0, 0, 1, 1]])
    edge_index1 = mx.array([[0, 1, 1, 2, 3], [1, 0, 2, 3, 1]])
    node_features2 = mx.array([[1, 1, 0, 1], [1, 0, 1, 1], [1, 0, 0, 1]])
    edge_index2 = mx.array([[0, 1, 2, 3, 1], [1, 0, 1, 2, 2]])
    g1 = GraphData(node_features=node_features1, edge_index=edge_index1)
    g2 = GraphData(node_features=node_features2, edge_index=edge_index2)

    graph_batch = batch([g1, g2], collate_fn=collate)

    expected_global_node_features = mx.array(
        [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 0, 1, 1],
            [0, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 0, 1, 1],
            [1, 0, 0, 1],
        ]
    )

    expected_global_edge_index = mx.array(
        [[0, 1, 1, 2, 3, 4, 5, 6, 7, 5], [1, 0, 2, 3, 1, 5, 6, 5, 6, 6]]
    )

    assert mx.array_equal(
        graph_batch.node_features, expected_global_node_features
    ), "error in constructing global node features"
    assert mx.array_equal(
        graph_batch.edge_index, expected_global_edge_index
    ), "error in constructing global edge index"


def test_batch_indexing():
    node_features1 = mx.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 0, 1, 1], [0, 0, 1, 1]])
    edge_index1 = mx.array([[0, 1, 1, 2, 3], [1, 0, 2, 3, 1]])
    node_features2 = mx.array([[1, 1, 0, 1], [1, 0, 1, 1], [1, 0, 0, 1]])
    edge_index2 = mx.array([[0, 1, 2, 3, 1], [1, 0, 1, 2, 2]])
    g1 = GraphData(node_features=node_features1, edge_index=edge_index1)
    g2 = GraphData(node_features=node_features2, edge_index=edge_index2)

    graph_batch = batch([g1, g2])
    reconstructed_g2 = graph_batch[1]
    assert mx.array_equal(
        reconstructed_g2.edge_index, edge_index2
    ), "incorrect edge_index, error in indexing batch"
    assert mx.array_equal(
        reconstructed_g2.node_features, node_features2
    ), "incorrect node_features, error in indexing batch"
