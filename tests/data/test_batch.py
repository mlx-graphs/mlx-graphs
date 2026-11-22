import mlx.core as mx
import numpy as np
import pytest

from mlx_graphs.data import GraphData, batch, unbatch


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
    assert len(graph_batch) == 2, "Batch num graphs failed"

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

    # Custom attributes with custom __cat_dim__ & __inc__
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
                return len(self.node_features)
            return None

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
    ), "Batch with custom __cat_dim__ and __inc__ failed"
    assert mx.array_equal(
        graph_batch[0].custom_attr, custom_attr_1
    ), "Batch indexing with custom __cat_dim__ and __inc__ failed"
    assert mx.array_equal(
        graph_batch[1].custom_attr, custom_attr_2
    ), "Batch indexing with custom __cat_dim__ and __inc__ failed"

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

    assert mx.array_equal(
        graph_batch[98:][-1].node_features, mock_node_features * 99
    ), "Batch indexing with last index failed"

    # Slices are clipped like those of Python lists
    assert len(graph_batch[:101]) == 100
    assert graph_batch[:-101] == []
    assert graph_batch[3:2] == []
    assert graph_batch[-3:2] == []

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

    # Batch indices
    g1 = GraphData(node_features=node_features1, edge_index=edge_index1)
    g2 = GraphData(node_features=node_features2, edge_index=edge_index2)
    g3 = GraphData(node_features=node_features2, edge_index=edge_index2)

    graph_batch = batch([g1, g2, g3])
    assert mx.array_equal(
        graph_batch.batch_indices, mx.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
    ), "Batch batch_indices failed"

    graph_batch = batch([g1])
    assert mx.array_equal(
        graph_batch.batch_indices, mx.array([0, 0, 0, 0])
    ), "Batch batch_indices single graph failed"

    # Slicing with lists and arrays
    graph_batch = batch([g1, g2, g3])
    indices = [0, 1, 1]
    sliced = graph_batch[indices]

    assert mx.array_equal(sliced[0].node_features, node_features1) and mx.array_equal(
        sliced[0].edge_index, edge_index1
    ), "Batch slicing with list failed"
    assert mx.array_equal(sliced[1].node_features, node_features2) and mx.array_equal(
        sliced[1].edge_index, edge_index2
    ), "Batch slicing with list failed"
    assert mx.array_equal(sliced[2].node_features, node_features2) and mx.array_equal(
        sliced[2].edge_index, edge_index2
    ), "Batch slicing with list failed"

    indices = mx.array([0, 1, 1])
    sliced = graph_batch[indices]

    assert mx.array_equal(sliced[0].node_features, node_features1) and mx.array_equal(
        sliced[0].edge_index, edge_index1
    ), "Batch slicing with list failed"
    assert mx.array_equal(sliced[1].node_features, node_features2) and mx.array_equal(
        sliced[1].edge_index, edge_index2
    ), "Batch slicing with list failed"
    assert mx.array_equal(sliced[2].node_features, node_features2) and mx.array_equal(
        sliced[2].edge_index, edge_index2
    ), "Batch slicing with list failed"

    indices = mx.array([-1, -2])
    sliced = graph_batch[indices]

    assert mx.array_equal(sliced[0].node_features, node_features2) and mx.array_equal(
        sliced[0].edge_index, edge_index2
    ), "Batch slicing with list failed"
    assert mx.array_equal(sliced[1].node_features, node_features2) and mx.array_equal(
        sliced[1].edge_index, edge_index2
    ), "Batch slicing with list failed"


def test_batching_with_isolated_nodes():
    # Graph 1 has 4 nodes but node 3 is isolated, so it never appears in edge_index
    graph1 = GraphData(edge_index=mx.array([[0, 1, 1, 2], [1, 0, 2, 1]]), num_nodes=4)
    graph2 = GraphData(edge_index=mx.array([[0, 1], [1, 0]]), num_nodes=2)

    assert graph1.num_nodes == 4

    batched = batch([graph1, graph2])

    expected_edge_index = mx.array([[0, 1, 1, 2, 4, 5], [1, 0, 2, 1, 5, 4]])
    assert mx.array_equal(batched.edge_index, expected_edge_index)
    assert mx.array_equal(batched.batch_indices, mx.array([0, 0, 0, 0, 1, 1]))
    assert batched.num_nodes == 6

    # Unbatching gives back the original edge_index and node count
    assert mx.array_equal(batched[0].edge_index, graph1.edge_index)
    assert batched[0].num_nodes == 4
    assert mx.array_equal(batched[1].edge_index, graph2.edge_index)
    assert batched[1].num_nodes == 2


def test_batching_with_empty_first_graph():
    # The first graph has no node: the edges of the next graphs must still be
    # offset, and every graph gets its batch indices.
    empty = GraphData(
        edge_index=mx.zeros((2, 0), dtype=mx.int32), node_features=mx.zeros((0, 1))
    )
    graph = GraphData(
        edge_index=mx.array([[0, 1], [1, 2]]), node_features=mx.ones((3, 1))
    )
    batched = batch([empty, graph, graph])
    assert batched.edge_index.tolist() == [[0, 1, 3, 4], [1, 2, 4, 5]]
    assert batched.batch_indices.tolist() == [1, 1, 1, 2, 2, 2]
    assert batched.ptr.tolist() == [0, 0, 3, 6]
    assert batched[2].edge_index.tolist() == [[0, 1], [1, 2]]


def test_num_nodes_explicit_overrides_inference():
    edge_index = mx.array([[0, 1], [1, 0]])
    assert GraphData(edge_index=edge_index).num_nodes == 2
    assert GraphData(edge_index=edge_index, num_nodes=5).num_nodes == 5


def test_batching_scalar_and_object_attributes():
    # As in PyG: scalars become one entry per graph, other objects a list,
    # and unbatching gives each graph its original value back.
    graphs = [
        GraphData(
            edge_index=mx.array([[0, 1], [1, 0]]),
            node_features=mx.ones((2, 3)),
            seed_count=i + 1,
            weight=0.5 * i,
            is_train=i % 2 == 0,
            name=f"graph_{i}",
        )
        for i in range(3)
    ]
    batched = batch(graphs)
    assert batched.seed_count.tolist() == [1, 2, 3]
    assert batched.weight.tolist() == [0.0, 0.5, 1.0]
    assert batched.is_train.tolist() == [True, False, True]
    assert batched.name == ["graph_0", "graph_1", "graph_2"]
    assert batched.edge_index.tolist() == [[0, 1, 2, 3, 4, 5], [1, 0, 3, 2, 5, 4]]

    for i, graph in enumerate(batched[:]):
        assert graph.seed_count == i + 1 and isinstance(graph.seed_count, int)
        assert graph.weight == 0.5 * i
        assert graph.is_train is (i % 2 == 0)
        assert graph.name == f"graph_{i}"
        assert mx.array_equal(graph.edge_index, graphs[i].edge_index)


def test_batching_numpy_scalar_attributes():
    # NumPy scalars (e.g. from `np.sum` or `array[i]`) have a `shape`, but are
    # scalars like Python numbers.
    graphs = [
        GraphData(
            edge_index=mx.array([[0], [1]]),
            count=np.int64(i),
            score=np.float32(0.5 * i),
            flag=np.bool_(i % 2 == 0),
        )
        for i in range(3)
    ]
    batched = batch(graphs)
    assert batched.count.tolist() == [0, 1, 2]
    assert batched.score.tolist() == [0.0, 0.5, 1.0]
    assert batched.flag.tolist() == [True, False, True]
    for i, graph in enumerate(batched[:]):
        assert graph.count == i and graph.score == 0.5 * i
        assert graph.flag == (i % 2 == 0)


@pytest.mark.parametrize("dtype", [mx.int32, mx.int64, mx.int16])
def test_unbatching_keeps_index_dtype(dtype):
    graphs = [
        GraphData(edge_index=mx.array([[0, 1], [1, 0]], dtype=dtype)),
        GraphData(edge_index=mx.array([[0, 2], [2, 1]], dtype=dtype)),
    ]
    batched = batch(graphs)
    assert batched.edge_index.dtype == dtype
    for graph, original in zip(unbatch(batched), graphs):
        assert graph.edge_index.dtype == dtype
        assert graph.edge_index.tolist() == original.edge_index.tolist()


def test_batch_indexing_like_a_list():
    graphs = [
        GraphData(edge_index=mx.array([[0], [1]]), node_features=mx.full((2, 1), i))
        for i in range(5)
    ]
    batched = batch(graphs)
    ids = list(range(5))

    def values(out):
        return [int(g.node_features[0, 0].item()) for g in out]

    with pytest.raises(IndexError):
        batched[5]
    with pytest.raises(IndexError):
        batched[-6]
    with pytest.raises(IndexError):
        batched[[0, 5]]
    assert values([batched[np.int64(-1)], batched[mx.array(2)]]) == [4, 2]
    for s in [slice(0, 0), slice(None, None, -1), slice(-2, None), slice(4, 0, -2),
              slice(1, 100), slice(None, None, 2), slice(3, 1)]:  # fmt: skip
        assert values(batched[s]) == ids[s], s
    assert values(batched[mx.array([1])]) == [1]
    assert values(batched[[3]]) == [3]
    assert values(batched[(4, -5)]) == [4, 0]
    assert values(batched[np.array([2, 2])]) == [2, 2]
    assert values(batched[[]]) == []
    assert values(batched[mx.array([True, False, False, True, False])]) == [0, 3]


def test_batching_zero_dim_array_attributes():
    # As in PyG, 0-dim arrays are stacked (one entry per graph), incremented
    # if needed, and unbatched back to 0-dim arrays.
    graphs = [
        GraphData(
            edge_index=mx.array([[0, 1], [1, 0]], dtype=mx.int32),
            node_features=mx.ones((2 + i, 1)),
            weight=mx.array(0.5 * i),
            root_index=mx.array(i, dtype=mx.int32),
        )
        for i in range(3)
    ]
    batched = batch(graphs)
    assert batched.weight.tolist() == [0.0, 0.5, 1.0]
    # offset by the number of nodes of the previous graphs (2, 3)
    assert batched.root_index.tolist() == [0, 1 + 2, 2 + 5]
    for i, graph in enumerate(unbatch(batched)):
        assert graph.weight.ndim == 0 and graph.weight.item() == 0.5 * i
        assert graph.root_index.ndim == 0 and graph.root_index.item() == i
        assert graph.root_index.dtype == mx.int32


def test_unbatch_round_trip_with_every_attribute_kind():
    from mlx_graphs.loaders import Dataloader

    graphs = [
        GraphData(
            edge_index=mx.array([[0, 1], [1, i]], dtype=mx.int32),
            node_features=mx.full((3 + i, 2), i, dtype=mx.float32),
            graph_labels=mx.array([i]),
            pair_index=mx.array([[0], [i]], dtype=mx.int32),  # incremented, dim 1
            score=0.5 * i,  # scalar
            name=f"g{i}",  # list
            center=mx.array(i),  # 0-dim
            num_nodes=3 + i,
        )
        for i in range(4)
    ]
    batched = next(iter(Dataloader(graphs, batch_size=4, follow_batch=["x"])))
    for out, graph in zip(unbatch(batched), graphs):
        assert set(out.keys()) == set(graph.keys())
        for key in graph.keys():
            expected, value = getattr(graph, key), getattr(out, key)
            if isinstance(expected, mx.array):
                assert value.dtype == expected.dtype, key
                assert value.shape == expected.shape, key
                assert mx.array_equal(value, expected), key
            else:
                assert value == expected, key


def test_batch_index_vectors_are_int32():
    from mlx_graphs.loaders import Dataloader

    graphs = [
        GraphData(edge_index=mx.array([[0], [1]]), node_features=mx.ones((2 + i, 1)))
        for i in range(3)
    ]
    batched = next(iter(Dataloader(graphs, batch_size=3, follow_batch=["x"])))
    assert batched.batch_indices.dtype == mx.int32
    assert batched.batch.dtype == mx.int32
    assert batched.ptr.dtype == mx.int32
    assert batched.ptr.tolist() == [0, 2, 5, 9]
    assert batched.x_batch.dtype == mx.int32
