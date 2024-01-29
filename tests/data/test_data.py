import mlx.core as mx

from mlx_graphs.data.data import GraphData


def test_data():
    # kwargs
    data = GraphData(a=2)
    assert data.a == 2, "extra kwarg not assigned correctly"  # type: ignore
    assert "a" in data.to_dict(), "extra kwarg not in dict"

    # Printing
    data = GraphData(
        node_features=mx.array([1, 2, 3, 4, 5]),
        edge_index=mx.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]),
    )
    assert (
        data.__repr__()
        == """GraphData(
	edge_index=[2, 5], int32
	node_features=[5], int32)"""
    ), "GraphData printing failed"

    data = GraphData(
        node_features=mx.ones((5, 100), mx.float32),
        edge_index=mx.zeros((2, 10000), mx.int32),
        new_attr=32,
    )
    assert (
        data.__repr__()
        == """GraphData(
	edge_index=[2, 10000], int32
	node_features=[5, 100], float32
	new_attr=32)"""
    ), "GraphData printing failed"
