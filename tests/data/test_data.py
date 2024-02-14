import mlx.core as mx

from mlx_graphs.data import GraphData


def test_data_repr():
    # kwargs
    data = GraphData(a=2)
    assert data.a == 2, "extra kwarg not assigned correctly"  # type: ignore
    assert "a" in data.to_dict(), "extra kwarg not in dict"

    data = GraphData(
        node_features=mx.array([1, 2, 3, 4, 5]),
        edge_index=mx.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]),
    )
    assert (
        data.__repr__()
        == """GraphData(
	edge_index(shape=(2, 5), int32)
	node_features(shape=(5,), int32))"""
    ), "GraphData printing failed"

    data = GraphData(
        node_features=mx.ones((5, 100), mx.float32),
        edge_index=mx.zeros((2, 10000), mx.int32),
        new_attr=32,
    )
    assert (
        data.__repr__()
        == """GraphData(
	edge_index(shape=(2, 10000), int32)
	node_features(shape=(5, 100), float32)
	new_attr=32)"""
    ), "GraphData printing failed"


def test_data_num_classes():
    data = GraphData(
        node_features=mx.array([1, 2, 3, 4, 5]),
        node_labels=mx.expand_dims(mx.arange(10), 0),
    )
    assert data.num_node_classes == 10, "GraphData num_classes failed"

    data = GraphData(
        node_features=mx.array([1, 2, 3, 4, 5]),
        edge_labels=mx.expand_dims(mx.arange(10), 0),
    )
    assert data.num_edge_classes == 10, "GraphData num_classes failed"

    data = GraphData(
        node_features=mx.array([1, 2, 3, 4, 5]),
        graph_labels=mx.expand_dims(mx.arange(10), 0),
    )
    assert data.num_graph_classes == 10, "GraphData num_classes failed"

    data = GraphData(
        node_features=mx.array([1, 2, 3, 4, 5]),
        node_labels=mx.array([0.1, 0.4, 0.6, 0.8]),
    )
    assert data.num_node_classes == 4, "GraphData num_classes failed"

    data = GraphData(
        node_features=mx.array([1, 2, 3, 4, 5]), node_labels=mx.array([1, 2, 3, 4])
    )
    assert data.num_node_classes == 4, "GraphData num_classes failed"

    data = GraphData(
        node_features=mx.array([1, 2, 3, 4, 5]),
        node_labels=mx.array(mx.expand_dims(mx.array([1, 2, 3, 4]), 1)),
    )
    assert data.num_node_classes == 4, "GraphData num_classes failed"

    data = GraphData()
    assert data.num_node_classes == 0, "GraphData num_classes failed"
    assert data.num_edge_classes == 0, "GraphData num_classes failed"
    assert data.num_graph_classes == 0, "GraphData num_classes failed"


def test_data_num_features():
    data = GraphData(
        node_features=mx.array([1, 2, 3, 4, 5]),
    )
    assert data.num_node_features == 1, "GraphData num_features failed"

    data = GraphData(
        node_features=mx.ones((5, 10)),
    )
    assert data.num_node_features == 10, "GraphData num_features failed"

    data = GraphData(
        edge_features=mx.ones((5, 10)),
    )
    assert data.num_edge_features == 10, "GraphData num_features failed"

    data = GraphData(
        graph_features=mx.ones((5, 10)),
    )
    assert data.num_graph_features == 10, "GraphData num_features failed"

    data = GraphData()
    assert data.num_node_features == 0, "GraphData num_features failed"
    assert data.num_edge_features == 0, "GraphData num_features failed"
    assert data.num_graph_features == 0, "GraphData num_features failed"
