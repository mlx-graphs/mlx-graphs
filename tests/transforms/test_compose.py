import mlx.core as mx

from mlx_graphs.data import GraphData
from mlx_graphs.transforms.base_transform import BaseTransform
from mlx_graphs.transforms.compose import Compose
from mlx_graphs.transforms.features import Constant, GCNNorm
from mlx_graphs.transforms.splits import RandomLinkSplit
from mlx_graphs.transforms.structure import (
    AddSelfLoops,
    RemoveSelfLoops,
    ToUndirected,
)


def test_compose_applies_transforms_in_order():
    graph = GraphData(edge_index=mx.array([[0, 1, 1], [1, 1, 2]]))

    # Removing the self-loops before adding them leaves one per node...
    transform = Compose([RemoveSelfLoops(), AddSelfLoops()])
    assert transform(graph).edge_index.tolist() == [[0, 1, 0, 1, 2], [1, 2, 0, 1, 2]]

    # ... and the other way around leaves none.
    transform = Compose([AddSelfLoops(), RemoveSelfLoops()])
    assert transform(graph).edge_index.tolist() == [[0, 1], [1, 2]]


def test_compose_matches_manual_chaining():
    graph = GraphData(
        edge_index=mx.array([[0, 1, 1, 3], [1, 1, 2, 0]]),
        node_features=mx.ones((4, 2)),
    )
    transforms = [RemoveSelfLoops(), ToUndirected(), Constant(2.0), GCNNorm()]

    expected = graph
    for transform in transforms:
        expected = transform(expected)
    composed = Compose(transforms)(graph)

    assert sorted(composed.to_dict()) == sorted(expected.to_dict())
    assert mx.array_equal(composed.edge_index, expected.edge_index)
    assert mx.array_equal(composed.node_features, expected.node_features)
    assert mx.array_equal(composed.edge_weights, expected.edge_weights)


def test_compose_does_not_modify_its_input():
    edge_index = mx.array([[0, 1, 1], [1, 1, 2]])
    graph = GraphData(edge_index=edge_index)

    transformed = Compose([RemoveSelfLoops(), Constant()])(graph)

    assert transformed is not graph
    assert graph.edge_index is edge_index
    assert graph.node_features is None
    assert not hasattr(graph, "_num_nodes")


def test_compose_empty_and_plain_callables():
    graph = GraphData(edge_index=mx.array([[0, 1], [1, 2]]))
    assert Compose([])(graph).edge_index.tolist() == [[0, 1], [1, 2]]

    def tag(data):
        data.tag = "seen"
        return data

    transformed = Compose([tag, AddSelfLoops()])(graph)
    assert transformed.tag == "seen"
    assert transformed.num_edges == 5


def test_compose_maps_over_multiple_graphs():
    # RandomLinkSplit returns three graphs: the next transforms apply to each.
    mx.random.seed(0)
    graph = GraphData(
        edge_index=mx.array([[0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7, 0]]),
        node_features=mx.ones((8, 2)),
    )
    transform = Compose([RandomLinkSplit(num_val=1, num_test=2), Constant(3.0)])
    splits = transform(graph)

    assert isinstance(splits, tuple) and len(splits) == 3
    for split in splits:
        assert split.node_features.shape == (8, 3)
        assert split.node_features[:, -1].tolist() == [3.0] * 8
        assert split.edge_label_index is not None


def test_compose_repr_and_base_class():
    transform = Compose([RemoveSelfLoops(), ToUndirected(reduce="mean")])
    assert isinstance(transform, BaseTransform)
    assert str(transform) == (
        "Compose([\n  RemoveSelfLoops(),\n  ToUndirected(reduce='mean')\n])"
    )
    assert str(Compose([])) == "Compose([])"
