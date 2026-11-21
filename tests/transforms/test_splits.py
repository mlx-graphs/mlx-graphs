import warnings

import mlx.core as mx
import numpy as np
import pytest

from mlx_graphs.data import GraphData, batch
from mlx_graphs.transforms.splits import RandomLinkSplit, RandomNodeSplit


def edge_set(edges) -> set:
    edges = np.asarray(edges)
    return set(zip(edges[0].tolist(), edges[1].tolist()))


def reverse(edges: set) -> set:
    return {(j, i) for i, j in edges}


def directed_graph(seed: int, num_nodes: int = 40, num_edges: int = 150) -> GraphData:
    """Random directed graph without duplicate edges, with identifying edge
    features. The last 5 nodes are isolated."""
    rng = np.random.default_rng(seed)
    pairs = rng.choice((num_nodes - 5) ** 2, size=num_edges, replace=False)
    edges = np.stack([pairs // (num_nodes - 5), pairs % (num_nodes - 5)])
    return GraphData(
        edge_index=mx.array(edges),
        node_features=mx.ones((num_nodes, 3)),
        edge_features=mx.arange(num_edges),
    )


def undirected_graph(seed: int, num_nodes: int = 40, num_pairs: int = 100):
    """Random undirected graph: both directions of every edge are present."""
    rng = np.random.default_rng(seed)
    low, high = np.triu_indices(num_nodes - 5, k=1)
    chosen = rng.choice(low.size, size=num_pairs, replace=False)
    low, high = low[chosen], high[chosen]
    edges = np.stack([np.concatenate([low, high]), np.concatenate([high, low])])
    edges = edges[:, rng.permutation(edges.shape[1])]
    return GraphData(edge_index=mx.array(edges), node_features=mx.ones((num_nodes, 3)))


def masks(graph: GraphData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    for name in ("train_mask", "val_mask", "test_mask"):
        mask = getattr(graph, name)
        assert mask.dtype == mx.bool_, name
        assert mask.shape == (graph.num_nodes,), name
    return (
        np.asarray(graph.train_mask),
        np.asarray(graph.val_mask),
        np.asarray(graph.test_mask),
    )


def labeled_graph(seed: int, num_nodes: int = 200, num_classes: int = 4) -> GraphData:
    rng = np.random.default_rng(seed)
    return GraphData(
        edge_index=mx.array(rng.integers(0, num_nodes, size=(2, 300))),
        node_features=mx.ones((num_nodes, 2)),
        node_labels=mx.array(rng.integers(0, num_classes, size=num_nodes)),
    )


def test_repr():
    assert str(RandomNodeSplit()) == "RandomNodeSplit(split='train_rest')"
    assert str(RandomNodeSplit("random")) == "RandomNodeSplit(split='random')"
    assert str(RandomLinkSplit()) == "RandomLinkSplit(num_val=0.1, num_test=0.2)"


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("num_val, num_test", [(20, 50), (0.1, 0.25), (0, 0.333)])
def test_random_node_split_train_rest(seed, num_val, num_test):
    torch = pytest.importorskip("torch")
    pyg_data = pytest.importorskip("torch_geometric.data")
    pyg_transforms = pytest.importorskip("torch_geometric.transforms")

    graph = labeled_graph(seed)
    mx.random.seed(seed)
    split = RandomNodeSplit(num_val=num_val, num_test=num_test)(graph)
    train, val, test = masks(split)

    # Disjoint and covering.
    assert np.array_equal(train.astype(int) + val + test, np.ones(200, dtype=int))
    # Same sizes as PyG.
    expected = pyg_transforms.RandomNodeSplit(num_val=num_val, num_test=num_test)(
        pyg_data.Data(x=torch.ones(200, 2), y=torch.zeros(200, dtype=torch.long))
    )
    assert train.sum() == int(expected.train_mask.sum())
    assert val.sum() == int(expected.val_mask.sum())
    assert test.sum() == int(expected.test_mask.sum())
    # The input graph is untouched.
    assert not hasattr(graph, "train_mask")


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("split", ["random", "test_rest"])
def test_random_node_split_per_class(seed, split):
    torch = pytest.importorskip("torch")
    pyg_data = pytest.importorskip("torch_geometric.data")
    pyg_transforms = pytest.importorskip("torch_geometric.transforms")

    graph = labeled_graph(seed)
    labels = np.asarray(graph.node_labels)
    mx.random.seed(seed)
    kwargs = dict(split=split, num_val=30, num_test=0.2, num_train_per_class=10)
    result = RandomNodeSplit(**kwargs)(graph)
    train, val, test = masks(result)

    assert not (train & val).any() and not (train & test).any()
    assert not (val & test).any()
    # The same number of training nodes in every class.
    assert np.bincount(labels[train], minlength=4).tolist() == [10] * 4
    expected = pyg_transforms.RandomNodeSplit(**kwargs)(
        pyg_data.Data(x=torch.ones(200, 2), y=torch.from_numpy(labels))
    )
    assert val.sum() == int(expected.val_mask.sum()) == 30
    assert test.sum() == int(expected.test_mask.sum())
    if split == "test_rest":
        assert (train | val | test).all()
        assert test.sum() == 200 - 40 - 30
    else:
        assert test.sum() == 40
        assert not (train | val | test).all()


def test_random_node_split_small_classes_and_custom_key():
    # Class 2 has fewer nodes than `num_train_per_class`, and the node with a
    # negative label is never used for training.
    labels = mx.array([0, 0, 0, 0, 1, 1, 1, 2, -1, 0])
    graph = GraphData(
        edge_index=mx.array([[0, 1], [1, 2]]), num_nodes=10, targets=labels
    )
    mx.random.seed(0)
    result = RandomNodeSplit(
        "test_rest", num_val=1, num_train_per_class=2, key="targets"
    )(graph)
    train, val, test = masks(result)

    labels_np = np.asarray(labels)
    assert np.bincount(labels_np[train][labels_np[train] >= 0]).tolist() == [2, 2, 1]
    assert not train[8]
    assert val.sum() == 1
    assert (train.astype(int) + val + test == 1).all()


def test_random_node_split_is_random_and_reproducible():
    graph = labeled_graph(0)
    for split in ("train_rest", "random"):
        transform = RandomNodeSplit(split, num_val=20, num_test=30)
        mx.random.seed(1)
        first = transform(graph)
        mx.random.seed(1)
        second = transform(graph)
        mx.random.seed(2)
        third = transform(graph)
        for name in ("train_mask", "val_mask", "test_mask"):
            assert mx.array_equal(getattr(first, name), getattr(second, name))
        assert not mx.array_equal(first.val_mask, third.val_mask)


def test_random_node_split_without_features_and_invalid():
    # Isolated nodes known through an explicit number of nodes get a mask entry.
    graph = GraphData(edge_index=mx.array([[0, 1], [1, 2]]), num_nodes=10)
    mx.random.seed(0)
    train, val, test = masks(RandomNodeSplit(num_val=2, num_test=3)(graph))
    assert (train.sum(), val.sum(), test.sum()) == (5, 2, 3)

    with pytest.raises(ValueError):
        RandomNodeSplit("other")
    with pytest.raises(ValueError):
        RandomNodeSplit("random")(graph)  # no labels
    graph.node_labels = mx.array([0, 1])
    with pytest.raises(ValueError):
        RandomNodeSplit("random")(graph)  # not one label per node


def check_link_split(graph, train, val, test, num_val, num_test, undirected=False):
    """Invariants of a link split that hold for every configuration."""
    all_edges = edge_set(graph.edge_index)
    positives, negatives = [], []
    for split in (train, val, test):
        assert split.edge_label_index.dtype == graph.edge_index.dtype
        assert split.edge_index.dtype == graph.edge_index.dtype
        assert split.edge_label.dtype == mx.float32
        assert split.edge_label.shape == (split.edge_label_index.shape[1],)
        assert split.num_nodes == graph.num_nodes
        assert split.node_features is graph.node_features

        labels = np.asarray(split.edge_label)
        label_index = np.asarray(split.edge_label_index)
        num_pos = int(labels.sum())
        # Positive edges first, then negative edges.
        assert labels[:num_pos].tolist() == [1.0] * num_pos
        assert labels[num_pos:].tolist() == [0.0] * (labels.size - num_pos)
        positives.append(edge_set(label_index[:, :num_pos]))
        negatives.append(edge_set(label_index[:, num_pos:]))
        assert len(positives[-1]) == num_pos
        assert len(negatives[-1]) == labels.size - num_pos

    train_pos, val_pos, test_pos = positives
    assert len(val_pos) == num_val and len(test_pos) == num_test
    # The positive edges of the three splits are disjoint edges of the graph.
    assert not (train_pos & val_pos or train_pos & test_pos or val_pos & test_pos)
    union = train_pos | val_pos | test_pos
    if undirected:
        assert all(i <= j for i, j in union)
        assert union | reverse(union) == all_edges
        train_pos, val_pos, test_pos = (
            edges | reverse(edges) for edges in (train_pos, val_pos, test_pos)
        )
    else:
        assert union == all_edges

    # Message passing edges: no leakage of the edges to predict.
    assert edge_set(train.edge_index) == train_pos
    assert edge_set(val.edge_index) == train_pos
    assert edge_set(test.edge_index) == train_pos | val_pos
    assert not (edge_set(val.edge_index) & val_pos)
    assert not (edge_set(val.edge_index) & test_pos)
    assert not (edge_set(test.edge_index) & test_pos)

    # Negative edges are non-edges of the full graph, without overlap.
    train_neg, val_neg, test_neg = negatives
    for sampled in negatives:
        assert not (sampled & all_edges)
        assert all(i != j for i, j in sampled)
    assert not (train_neg & val_neg or train_neg & test_neg or val_neg & test_neg)
    return negatives


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"num_val": 0.05, "num_test": 0.1, "neg_sampling_ratio": 2.0},
        {"num_val": 10, "num_test": 25, "add_negative_train_samples": False},
        {"num_val": 0.2, "num_test": 0.0, "neg_sampling_ratio": 0.5},
        {"num_val": 0, "num_test": 0},
    ],
)
def test_random_link_split_directed(seed, kwargs):
    torch = pytest.importorskip("torch")
    pyg_data = pytest.importorskip("torch_geometric.data")
    pyg_transforms = pytest.importorskip("torch_geometric.transforms")

    graph = directed_graph(seed)
    mx.random.seed(seed)
    train, val, test = RandomLinkSplit(**kwargs)(graph)

    # Same sizes as PyG for every array.
    expected = pyg_transforms.RandomLinkSplit(**kwargs)(
        pyg_data.Data(
            x=torch.ones(40, 3),
            edge_index=torch.from_numpy(np.asarray(graph.edge_index)),
            edge_attr=torch.arange(150),
        )
    )
    for ours, theirs in zip((train, val, test), expected):
        assert ours.edge_index.shape == tuple(theirs.edge_index.shape)
        assert ours.edge_label_index.shape == tuple(theirs.edge_label_index.shape)
        assert ours.edge_label.sum().item() == theirs.edge_label.sum().item()
        assert ours.edge_features.shape == tuple(theirs.edge_attr.shape)

    num_val, num_test = (int(expected[i].edge_label.sum()) for i in (1, 2))
    train_neg, val_neg, test_neg = check_link_split(
        graph, train, val, test, num_val, num_test
    )
    ratio = kwargs.get("neg_sampling_ratio", 1.0)
    num_train = 150 - num_val - num_test
    if kwargs.get("add_negative_train_samples", True):
        assert len(train_neg) == int(num_train * ratio)
    else:
        assert len(train_neg) == 0
    assert len(val_neg) == int(num_val * ratio)
    assert len(test_neg) == int(num_test * ratio)

    # Edge features follow the message passing edges.
    edges = np.asarray(graph.edge_index)
    for split in (train, val, test):
        positions = np.asarray(split.edge_features)
        assert np.array_equal(edges[:, positions], np.asarray(split.edge_index))


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize(
    "kwargs",
    [{}, {"num_val": 7, "num_test": 0.3, "neg_sampling_ratio": 1.5}],
)
def test_random_link_split_undirected(seed, kwargs):
    torch = pytest.importorskip("torch")
    pyg_data = pytest.importorskip("torch_geometric.data")
    pyg_transforms = pytest.importorskip("torch_geometric.transforms")

    graph = undirected_graph(seed)
    mx.random.seed(seed)
    train, val, test = RandomLinkSplit(is_undirected=True, **kwargs)(graph)

    expected = pyg_transforms.RandomLinkSplit(is_undirected=True, **kwargs)(
        pyg_data.Data(
            x=torch.ones(40, 3),
            edge_index=torch.from_numpy(np.asarray(graph.edge_index)),
        )
    )
    for ours, theirs in zip((train, val, test), expected):
        assert ours.edge_index.shape == tuple(theirs.edge_index.shape)
        assert ours.edge_label_index.shape == tuple(theirs.edge_label_index.shape)
        assert ours.edge_label.sum().item() == theirs.edge_label.sum().item()

    num_val, num_test = (int(expected[i].edge_label.sum()) for i in (1, 2))
    check_link_split(graph, train, val, test, num_val, num_test, undirected=True)

    # Both directions of an edge are in the same split: the message passing
    # graphs are symmetric and contain neither direction of an edge to predict.
    for split in (train, val, test):
        message_edges = edge_set(split.edge_index)
        assert reverse(message_edges) == message_edges
    for split in (val, test):
        labels = np.asarray(split.edge_label)
        positive = edge_set(np.asarray(split.edge_label_index)[:, labels == 1])
        message_edges = edge_set(split.edge_index)
        assert not (positive & message_edges)
        assert not (reverse(positive) & message_edges)


def test_random_link_split_undirected_edge_features():
    # 0 - 1 - 2 - 3 - 4 - 5, with the same feature on both directions.
    low = np.arange(5)
    edges = np.stack([np.concatenate([low, low + 1]), np.concatenate([low + 1, low])])
    graph = GraphData(
        edge_index=mx.array(edges, dtype=mx.uint32),
        edge_features=mx.array(np.concatenate([low, low]) * 10.0),
        num_nodes=8,
    )
    mx.random.seed(0)
    train, val, test = RandomLinkSplit(num_val=1, num_test=1, is_undirected=True)(graph)

    assert train.edge_index.shape == (2, 6)
    assert test.edge_index.shape == (2, 8)
    for split in (train, val, test):
        assert split.edge_index.dtype == mx.uint32
        assert split.num_nodes == 8
        split_edges = np.asarray(split.edge_index).astype(np.int64)
        features = np.asarray(split.edge_features)
        assert np.array_equal(features, split_edges.min(axis=0) * 10.0)


def test_random_link_split_is_random_and_reproducible():
    graph = directed_graph(0)
    transform = RandomLinkSplit()
    mx.random.seed(5)
    first = transform(graph)
    mx.random.seed(5)
    second = transform(graph)
    mx.random.seed(6)
    third = transform(graph)

    for a, b in zip(first, second):
        assert mx.array_equal(a.edge_index, b.edge_index)
        assert mx.array_equal(a.edge_label_index, b.edge_label_index)
    assert not mx.array_equal(first[0].edge_index, third[0].edge_index)
    # The input graph is untouched.
    assert not hasattr(graph, "edge_label_index")
    assert graph.num_edges == 150


def test_random_link_split_dense_graph_adjusts_negatives():
    # 6 nodes and 26 of the 30 possible edges: only 4 negative edges exist.
    src, dst = np.nonzero(~np.eye(6, dtype=bool))
    graph = GraphData(edge_index=mx.array(np.stack([src, dst])[:, :26]), num_nodes=6)
    mx.random.seed(0)

    with pytest.warns(UserWarning, match="not enough negative edges"):
        train, val, test = RandomLinkSplit(num_val=0.2, num_test=0.2)(graph)

    negatives = check_link_split(graph, train, val, test, 5, 5)
    assert sum(len(sampled) for sampled in negatives) == 4

    # A complete graph has no negative edge at all.
    graph = GraphData(edge_index=mx.array(np.stack([src, dst])), num_nodes=6)
    with pytest.warns(UserWarning):
        train, val, test = RandomLinkSplit()(graph)
    for split in (train, val, test):
        assert split.edge_label.min().item() == 1.0


def test_random_link_split_no_negatives_requested():
    graph = directed_graph(0)
    mx.random.seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        train, val, test = RandomLinkSplit(neg_sampling_ratio=0.0)(graph)
    assert train.edge_label_index.shape == (2, 105)
    assert val.edge_label_index.shape == (2, 15)
    assert test.edge_label_index.shape == (2, 30)
    assert train.edge_label.tolist() == [1.0] * 105


def test_random_link_split_insufficient_edges():
    graph = GraphData(edge_index=mx.array([[0, 1], [1, 2]]))
    with pytest.raises(ValueError, match="Insufficient"):
        RandomLinkSplit(num_val=1, num_test=1)(graph)
    with pytest.raises(ValueError, match="Insufficient"):
        RandomLinkSplit()(GraphData(edge_index=mx.zeros((2, 0), dtype=mx.int32)))


def test_random_link_split_graphs_can_be_batched():
    # `edge_label_index` is incremented like `edge_index` when batching.
    mx.random.seed(0)
    train_graphs = [RandomLinkSplit()(directed_graph(seed))[0] for seed in range(2)]
    batched = batch(train_graphs)

    num_labels = train_graphs[0].edge_label_index.shape[1]
    assert batched.edge_label_index.shape == (2, 2 * num_labels)
    assert batched.edge_label.shape == (2 * num_labels,)
    assert mx.array_equal(
        batched.edge_label_index[:, :num_labels], train_graphs[0].edge_label_index
    )
    assert mx.array_equal(
        batched.edge_label_index[:, num_labels:],
        train_graphs[1].edge_label_index + 40,
    )
    assert batched.num_nodes == 80


@pytest.mark.parametrize("dtype", [mx.int32, mx.uint32, mx.int64, mx.uint64])
@pytest.mark.parametrize("is_undirected", [False, True])
def test_random_link_split_preserves_edge_index_dtype(dtype, is_undirected):
    graph = undirected_graph(0)
    graph.edge_index = graph.edge_index.astype(dtype)
    mx.random.seed(0)
    splits = RandomLinkSplit(is_undirected=is_undirected)(graph)
    for split in splits:
        assert split.edge_index.dtype == dtype
        assert split.edge_label_index.dtype == dtype
    masked = RandomNodeSplit(num_val=5, num_test=5)(graph)
    assert int(masked.train_mask.sum().item()) == 30
