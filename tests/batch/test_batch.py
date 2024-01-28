import pytest

import mlx.core as mx
from mlx_graphs.data.batch import batch
from mlx_graphs.data.data import GraphData


def test_batching():
    node_features1 = mx.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 0, 1, 1], [0, 0, 1, 1]])
    edge_index1 = mx.array([[0, 1, 1, 2, 3], [1, 0, 2, 3, 1]])

    node_features2 = mx.array([[1, 1, 0, 1], [1, 0, 1, 1], [1, 0, 0, 1]])
    edge_index2 = mx.array([[0, 1, 2, 3, 1], [1, 0, 1, 2, 2]])

    g1 = GraphData(node_features=node_features1, edge_index=edge_index1)
    g2 = GraphData(node_features=node_features2, edge_index=edge_index2)
    graph_batch = batch([g1, g2])

    y_hat_edge_index = mx.array(
        [[0, 1, 1, 2, 3, 4, 5, 6, 7, 5], [1, 0, 2, 3, 1, 5, 4, 5, 6, 6]]
    )

    # Batch default collating
    assert mx.array_equal(
        graph_batch.node_features,
        mx.concatenate([node_features1, node_features2], axis=0),
    ), "Batch node features collating failed"
    assert mx.array_equal(
        graph_batch.edge_index, y_hat_edge_index
    ), "Batch edge index collating failed"

    # Batch indexing
    assert mx.array_equal(
        graph_batch[0].edge_index, edge_index1
    ), "Simple batch indexing with edge_index failed"
    assert mx.array_equal(
        graph_batch[1].edge_index, edge_index2
    ), "Simple batch indexing with node features failed"
    assert mx.array_equal(
        graph_batch[0].node_features, node_features1
    ), "Simple batch indexing with edge_index failed"
    assert mx.array_equal(
        graph_batch[1].node_features, node_features2
    ), "Simple batch indexing with node features failed"
    assert graph_batch.num_graphs == 2, "Batch num graphs failed"

    # Custom attributes 1D
    custom_attr1D_1 = mx.array([1, 2, 3, 4, 5])
    custom_attr1D_2 = mx.array([6, 7, 8, 9, 10])

    g1 = GraphData(
        node_features=node_features1,
        edge_index=edge_index1,
        custom_attr=custom_attr1D_1,
    )
    g2 = GraphData(
        node_features=node_features2,
        edge_index=edge_index2,
        custom_attr=custom_attr1D_2,
    )
    graph_batch = batch([g1, g2])

    assert mx.array_equal(
        graph_batch.custom_attr,
        mx.concatenate([custom_attr1D_1, custom_attr1D_2], axis=0),
    ), "Batch default indexing with custom attribute 1D failed"
    assert mx.array_equal(
        graph_batch[0].custom_attr, custom_attr1D_1
    ), "Batch default indexing with custom attribute 1D failed"
    assert mx.array_equal(
        graph_batch[1].custom_attr, custom_attr1D_2
    ), "Batch default indexing with custom attribute 1D failed"

    # Custom attributes 2D
    custom_attr2D_1 = mx.array([[1, 2], [3, 4], [5, 6]])
    custom_attr2D_2 = mx.array([[7, 8], [9, 10], [11, 12]])

    g1 = GraphData(
        node_features=node_features1,
        edge_index=edge_index1,
        custom_attr=custom_attr2D_1,
    )
    g2 = GraphData(
        node_features=node_features2,
        edge_index=edge_index2,
        custom_attr=custom_attr2D_2,
    )
    graph_batch = batch([g1, g2])

    assert mx.array_equal(
        graph_batch.custom_attr,
        mx.concatenate([custom_attr2D_1, custom_attr2D_2], axis=0),
    ), "Batch default indexing with custom attribute 2D failed"
    assert mx.array_equal(
        graph_batch[0].custom_attr, custom_attr2D_1
    ), "Batch default indexing with custom attribute 2D failed"
    assert mx.array_equal(
        graph_batch[1].custom_attr, custom_attr2D_2
    ), "Batch default indexing with custom attribute 2D failed"

    # Custom attributes with custom __cat_dim__ & __incr__
    custom_attr_1 = mx.array([[1, 2], [3, 4], [5, 6]])
    custom_attr_2 = mx.array([[7, 8], [9, 10], [11, 12]])

    class CustomData(GraphData):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __cat_dim__(self, key):
            if key == "custom_attr" or "index" in key:
                return 1
            return 0

        def __inc__(self, key):
            if key == "custom_attr" or "index" in key:
                return True
            return False

    g1 = CustomData(
        node_features=node_features1, edge_index=edge_index1, custom_attr=custom_attr_1
    )
    g2 = CustomData(
        node_features=node_features2, edge_index=edge_index2, custom_attr=custom_attr_2
    )
    graph_batch = batch([g1, g2])

    expect = mx.concatenate([custom_attr_1, custom_attr_2], axis=1)
    expect[:, 2:] += len(node_features1)

    assert mx.array_equal(
        graph_batch.custom_attr, expect
    ), "Batch with custom __cat_dim__ and __incr__ failed"
    assert mx.array_equal(
        graph_batch[0].custom_attr, custom_attr_1
    ), "Batch indexing with custom __cat_dim__ and __incr__ failed"
    assert mx.array_equal(
        graph_batch[1].custom_attr, custom_attr_2
    ), "Batch indexing with custom __cat_dim__ and __incr__ failed"

    # Batch backward indexing
    mock_edge_index = mx.array([[0, 1, 0, 1], [0, 1, 0, 1]])
    mock_node_features = mx.ones((5, 5))
    graphs = [
        GraphData(edge_index=mock_edge_index * i, node_features=mock_node_features * i)
        for i in range(100)
    ]

    graph_batch = batch(graphs)

    assert mx.array_equal(
        graph_batch[-1].node_features, mock_node_features * 99
    ) and mx.array_equal(
        graph_batch[-1].edge_index, mock_edge_index * 99
    ), "Batch backward indexing failed"

    assert mx.array_equal(
        graph_batch[-50].node_features, mock_node_features * 50
    ) and mx.array_equal(
        graph_batch[-50].edge_index, mock_edge_index * 50
    ), "Batch backward indexing failed"

    with pytest.raises(IndexError):
        graph_batch[-101]

    with pytest.raises(IndexError):
        graph_batch[101]

    # Batch slicing
    assert len(graph_batch[0:30]) == 30, "Batch slicing size failed"
    assert all(
        [isinstance(b, GraphData) for b in graph_batch[0:30]]
    ), "Batch slicing type failed"

    assert all(
        [
            mx.array_equal(b.node_features, mock_node_features * i)
            and mx.array_equal(b.edge_index, mock_edge_index * i)
            for i, b in enumerate(graph_batch[:10])
        ]
    ), "Batch start-slicing attributes failed"

    assert all(
        [
            mx.array_equal(b.node_features, mock_node_features * (i + 10))
            and mx.array_equal(b.edge_index, mock_edge_index * (i + 10))
            for i, b in enumerate(graph_batch[10:20])
        ]
    ), "Batch mid-slicing attributes failed"

    assert all(
        [
            mx.array_equal(b.node_features, mock_node_features * (i + 90))
            and mx.array_equal(b.edge_index, mock_edge_index * (i + 90))
            for i, b in enumerate(graph_batch[90:])
        ]
    ), "Batch end-slicing attributes failed"

    assert all(
        [
            mx.array_equal(b.node_features, mock_node_features * (i + 80))
            and mx.array_equal(b.edge_index, mock_edge_index * (i + 80))
            for i, b in enumerate(graph_batch[80:-10])
        ]
    ), "Batch end-slicing attributes with backward indexing failed"

    assert all(
        [
            mx.array_equal(b.node_features, mock_node_features * (i + 90))
            and mx.array_equal(b.edge_index, mock_edge_index * (i + 90))
            for i, b in enumerate(graph_batch[-10:])
        ]
    ), "Batch end-slicing attributes with backward indexing failed"

    assert all(
        [
            mx.array_equal(b.node_features, mock_node_features * (i * 2))
            and mx.array_equal(b.edge_index, mock_edge_index * (i * 2))
            for i, b in enumerate(graph_batch[:10:2])
        ]
    ), "Batch slicing attributes with loop jump failed"

    with pytest.raises(IndexError):
        graph_batch[:101]

    with pytest.raises(IndexError):
        graph_batch[:-101]

    with pytest.raises(IndexError):
        graph_batch[3:2]

    with pytest.raises(IndexError):
        graph_batch[-3:2]

    # Inconsistent GraphData list
    with pytest.raises(ValueError):
        g1 = GraphData(
            node_features=node_features1,
            edge_index=edge_index1,
            custom_attr=custom_attr_1,
        )
        g2 = GraphData(node_features=node_features2, edge_index=edge_index2)
        graph_batch = batch([g1, g2])

    with pytest.raises(ValueError):
        g1 = GraphData(node_features=node_features1, edge_index=edge_index1)
        g2 = GraphData(
            node_features=node_features2,
            edge_index=edge_index2,
            custom_attr=custom_attr_1,
        )
        graph_batch = batch([g1, g2])

    with pytest.raises(ValueError):

        class Data:
            def __init__(self, edge_index, node_features):
                self.edge_index = edge_index
                self.node_features = node_features

        g1 = GraphData(node_features=node_features1, edge_index=edge_index1)
        g2 = Data(node_features=node_features2, edge_index=edge_index2)
        graph_batch = batch([g1, g2])
