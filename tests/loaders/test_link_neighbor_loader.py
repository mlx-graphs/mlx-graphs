import mlx.core as mx
import numpy as np
import pytest

from mlx_graphs.data import GraphData, HeteroGraphData
from mlx_graphs.loaders import (
    ImbalancedSampler,
    LinkNeighborLoader,
    NegativeSampling,
    NeighborLoader,
)

from .helpers import check_subgraph, random_graph


def _global_label_index(batch):
    return np.asarray(batch.n_id)[np.asarray(batch.edge_label_index)]


def test_no_negative_sampling():
    data = random_graph()
    loader = LinkNeighborLoader(data, num_neighbors=[3, 2], batch_size=16)
    assert len(loader) == 8  # ceil(120 / 16)
    seen = []
    for batch in loader:
        check_subgraph(batch, data)
        input_id = np.asarray(batch.input_id)
        seen.extend(input_id.tolist())
        # Supervision edges are relabeled to the sampled nodes
        np.testing.assert_array_equal(
            _global_label_index(batch), np.asarray(data.edge_index)[:, input_id]
        )
        # The seeds (endpoints of supervision edges) come first in n_id
        seeds = np.unique(np.asarray(data.edge_index)[:, input_id])
        np.testing.assert_array_equal(np.asarray(batch.n_id)[: seeds.size], seeds)
        assert getattr(batch, "edge_label", None) is None
    assert seen == list(range(120))


def test_binary_negative_sampling():
    data = random_graph()
    label_index = data.edge_index[:, :40]
    loader = LinkNeighborLoader(
        data,
        num_neighbors=[2],
        edge_label_index=label_index,
        neg_sampling_ratio=1.5,
        batch_size=10,
    )
    batch = next(iter(loader))
    check_subgraph(batch, data)
    num_neg = 15  # ceil(1.5 * 10)
    assert batch.edge_label_index.shape == (2, 10 + num_neg)
    np.testing.assert_array_equal(
        np.asarray(batch.edge_label), [1.0] * 10 + [0.0] * num_neg
    )
    glob = _global_label_index(batch)
    np.testing.assert_array_equal(glob[:, :10], np.asarray(label_index)[:, :10])
    assert glob.min() >= 0 and glob.max() < data.num_nodes


def test_binary_with_edge_label_is_incremented():
    data = random_graph()
    edge_label = mx.array(np.arange(120) % 2)
    loader = LinkNeighborLoader(
        data,
        num_neighbors=[2],
        edge_label=edge_label,
        neg_sampling="binary",
        batch_size=8,
        shuffle=True,
    )
    for batch in loader:
        input_id = np.asarray(batch.input_id)
        labels = np.asarray(batch.edge_label)
        np.testing.assert_array_equal(labels[: input_id.size], input_id % 2 + 1)
        assert np.all(labels[input_id.size :] == 0)

    # Without negative sampling, labels are kept as they are
    loader = LinkNeighborLoader(data, [2], edge_label=edge_label, batch_size=8)
    batch = next(iter(loader))
    np.testing.assert_array_equal(
        np.asarray(batch.edge_label), np.asarray(batch.input_id) % 2
    )


def test_triplet_negative_sampling():
    data = random_graph()
    loader = LinkNeighborLoader(
        data,
        num_neighbors=[2, 2],
        neg_sampling=NegativeSampling("triplet", amount=3),
        batch_size=12,
    )
    batch = next(iter(loader))
    check_subgraph(batch, data)
    n_id = np.asarray(batch.n_id)
    input_id = np.asarray(batch.input_id)
    original = np.asarray(data.edge_index)[:, input_id]
    assert batch.dst_neg_index.shape == (12, 3)
    np.testing.assert_array_equal(n_id[np.asarray(batch.src_index)], original[0])
    np.testing.assert_array_equal(n_id[np.asarray(batch.dst_pos_index)], original[1])
    assert getattr(batch, "edge_label_index", None) is None

    loader = LinkNeighborLoader(
        data, num_neighbors=[2], neg_sampling="triplet", batch_size=5
    )
    assert next(iter(loader)).dst_neg_index.shape == (5,)

    with pytest.raises(ValueError):
        LinkNeighborLoader(
            data, [2], edge_label=mx.ones(120), neg_sampling="triplet", batch_size=4
        )
    with pytest.raises(ValueError):
        NegativeSampling("triplet", amount=1.5)


def test_weighted_negative_sampling():
    data = random_graph()
    weight = np.zeros(data.num_nodes)
    weight[[3, 7]] = 1.0
    loader = LinkNeighborLoader(
        data,
        num_neighbors=[1],
        neg_sampling=NegativeSampling(
            "binary", 2, src_weight=weight, dst_weight=weight
        ),
        batch_size=10,
    )
    batch = next(iter(loader))
    neg = _global_label_index(batch)[:, 10:]
    assert set(neg.reshape(-1).tolist()) <= {3, 7}


def test_determinism_under_seed():
    data = random_graph()

    def run():
        np.random.seed(3)
        loader = LinkNeighborLoader(
            data, [2], neg_sampling_ratio=1.0, batch_size=20, shuffle=True
        )
        return [
            (np.asarray(b.input_id).tolist(), _global_label_index(b).tolist())
            for b in loader
        ]

    assert run() == run()


def test_errors():
    data = random_graph()
    with pytest.raises(ValueError):
        LinkNeighborLoader(data, [2], edge_label=mx.ones(5))
    with pytest.raises(ValueError):
        LinkNeighborLoader(
            data, [2], neg_sampling="binary", neg_sampling_ratio=2.0, batch_size=2
        )
    with pytest.raises(ValueError):
        LinkNeighborLoader(data, {("a", "b", "c"): [2]})
    with pytest.raises(TypeError):
        LinkNeighborLoader("graph", [2])


def _hetero():
    rng = np.random.RandomState(0)
    writes = rng.randint(0, [6, 10], size=(30, 2)).T
    cites = rng.randint(0, 10, size=(2, 25))
    data = HeteroGraphData(
        edge_index_dict={
            ("author", "writes", "paper"): mx.array(writes),
            ("paper", "rev_writes", "author"): mx.array(writes[::-1].copy()),
            ("paper", "cites", "paper"): mx.array(cites),
        },
        node_features_dict={
            "author": mx.arange(6).reshape(-1, 1).astype(mx.float32),
            "paper": mx.arange(10).reshape(-1, 1).astype(mx.float32),
        },
    )
    return data, writes, cites


def test_hetero_link_neighbor_loader():
    data, writes, _ = _hetero()
    et = ("author", "writes", "paper")
    loader = LinkNeighborLoader(
        data,
        num_neighbors=[2, 2],
        edge_label_index=et,
        neg_sampling_ratio=1.0,
        batch_size=8,
    )
    assert len(loader) == 4
    seen = []
    for batch in loader:
        assert isinstance(batch, HeteroGraphData)
        assert batch.input_type == et
        input_id = np.asarray(batch.input_id)
        seen.extend(input_id.tolist())
        author_ids = np.asarray(batch.n_id_dict["author"])
        paper_ids = np.asarray(batch.n_id_dict["paper"])
        eli = np.asarray(batch.edge_label_index)
        n = input_id.size
        np.testing.assert_array_equal(author_ids[eli[0, :n]], writes[0, input_id])
        np.testing.assert_array_equal(paper_ids[eli[1, :n]], writes[1, input_id])
        assert eli.shape[1] == 2 * n
        np.testing.assert_array_equal(np.asarray(batch.edge_label), [1] * n + [0] * n)
        # Features follow the node ids
        np.testing.assert_array_equal(
            np.asarray(batch.node_features_dict["paper"]).reshape(-1), paper_ids
        )
        # Sampled edges exist in the original graph
        for edge_type, ei in batch.edge_index_dict.items():
            e_id = np.asarray(batch.e_id_dict[edge_type])
            src_t, _, dst_t = edge_type
            ei = np.asarray(ei)
            orig = np.asarray(data.edge_index_dict[edge_type])[:, e_id]
            if e_id.size:
                np.testing.assert_array_equal(
                    np.asarray(batch.n_id_dict[src_t])[ei[0]], orig[0]
                )
                np.testing.assert_array_equal(
                    np.asarray(batch.n_id_dict[dst_t])[ei[1]], orig[1]
                )
    assert seen == list(range(30))


def test_hetero_same_type_triplet():
    data, _, cites = _hetero()
    et = ("paper", "cites", "paper")
    loader = LinkNeighborLoader(
        data,
        num_neighbors=[2],
        edge_label_index=(et, mx.array(cites[:, :10])),
        neg_sampling=NegativeSampling("triplet", 2),
        batch_size=5,
    )
    batches = list(loader)
    assert len(batches) == 2
    b = batches[0]
    paper_ids = np.asarray(b.n_id_dict["paper"])
    np.testing.assert_array_equal(paper_ids[np.asarray(b.src_index)], cites[0, :5])
    np.testing.assert_array_equal(paper_ids[np.asarray(b.dst_pos_index)], cites[1, :5])
    assert b.dst_neg_index.shape == (5, 2)

    with pytest.raises(ValueError):
        LinkNeighborLoader(data, [2])
    with pytest.raises(ValueError):
        LinkNeighborLoader(data, [2], edge_label_index=("a", "b", "c"))


def test_neighbor_loader_with_imbalanced_sampler():
    data = random_graph(num_nodes=40, num_edges=100)
    data.node_labels = mx.array([0] * 36 + [1] * 4)
    input_nodes = mx.arange(40)
    sampler = ImbalancedSampler(data, input_nodes=input_nodes, num_samples=400)
    np.random.seed(0)
    loader = NeighborLoader(
        data, num_neighbors=[2], input_nodes=input_nodes, batch_size=50, sampler=sampler
    )
    assert len(loader) == 8
    labels = np.concatenate([np.asarray(b.node_labels[: b.batch_size]) for b in loader])
    assert labels.size == 400
    assert 0.35 < labels.mean() < 0.65

    # The homogeneous NeighborLoader also exposes original edge ids
    batch = next(iter(NeighborLoader(data, num_neighbors=[2], batch_size=5)))
    check_subgraph(batch, data)


def test_hetero_neighbor_loader_with_sampler():
    data, _, _ = _hetero()
    data.node_labels_dict = {"paper": mx.array([0] * 8 + [1] * 2)}
    sampler = ImbalancedSampler(data.node_labels_dict["paper"], num_samples=50)
    np.random.seed(0)
    loader = NeighborLoader(
        data,
        num_neighbors=[2],
        input_nodes="paper",
        batch_size=10,
        sampler=sampler,
    )
    assert len(loader) == 5
    batches = list(loader)
    for b in batches:
        seeds = np.asarray(b.n_id_dict["paper"][: b.batch_size])
        assert np.unique(seeds).size == seeds.size  # duplicates are merged
    labels = np.concatenate(
        [np.asarray(b.node_labels_dict["paper"][: b.batch_size]) for b in batches]
    )
    assert 0.25 < labels.mean() < 0.9


def test_graph_data_input():
    # edge_label_index may be a different set of (possibly unseen) pairs
    data = GraphData(
        edge_index=mx.array([[0, 1, 2], [1, 2, 3]]),
        node_features=mx.arange(5).reshape(-1, 1).astype(mx.float32),
    )
    loader = LinkNeighborLoader(
        data,
        num_neighbors=[-1],
        edge_label_index=mx.array([[4, 0], [3, 4]]),
        batch_size=2,
    )
    batch = next(iter(loader))
    np.testing.assert_array_equal(_global_label_index(batch), [[4, 0], [3, 4]])
    check_subgraph(batch, data)


def test_edge_features_follow_edges_when_num_nodes_equals_num_edges():
    # A directed cycle has as many edges as nodes.
    src = np.arange(6)
    data = GraphData(
        edge_index=mx.array(np.stack([src, (src + 1) % 6]).astype(np.int32)),
        node_features=mx.arange(6).reshape(-1, 1).astype(mx.float32),
        edge_features=(100 + mx.arange(6)).reshape(-1, 1).astype(mx.float32),
    )
    for b in LinkNeighborLoader(data, [2, 2], batch_size=2):
        np.testing.assert_array_equal(
            np.asarray(b.edge_features)[:, 0], 100 + np.asarray(b.e_id)
        )


def test_out_of_range_seeds_raise():
    # The C++ samplers read their buffers at the seed ids without bounds checks.
    data = random_graph()  # 30 nodes
    with pytest.raises(IndexError):
        NeighborLoader(data, [2], input_nodes=mx.array([3, 30]))
    with pytest.raises(IndexError):
        NeighborLoader(data, [2], input_nodes=mx.array([-1]))
    with pytest.raises(IndexError):
        LinkNeighborLoader(data, [2], edge_label_index=mx.array([[0, 1], [2, 30]]))

    hetero, _, _ = _hetero()  # 6 authors, 10 papers
    et = ("author", "writes", "paper")
    with pytest.raises(IndexError):
        LinkNeighborLoader(hetero, [2], edge_label_index=(et, mx.array([[6], [0]])))
    with pytest.raises(IndexError):
        LinkNeighborLoader(hetero, [2], edge_label_index=(et, mx.array([[0], [10]])))
    # In range: the largest ids of each type are accepted.
    LinkNeighborLoader(hetero, [2], edge_label_index=(et, mx.array([[5], [9]])))


@pytest.mark.parametrize("neg", [None, "binary", NegativeSampling("triplet", 2)])
def test_link_loader_index_outputs_are_int32(neg):
    data = random_graph()
    data.edge_index = data.edge_index.astype(mx.int32)
    for b in LinkNeighborLoader(data, [2], batch_size=8, neg_sampling=neg):
        assert b.edge_index.dtype == mx.int32
        for key in ("n_id", "e_id", "input_id", "edge_label_index", "src_index",
                    "dst_pos_index", "dst_neg_index"):  # fmt: skip
            if hasattr(b, key):
                assert getattr(b, key).dtype == mx.int32, key

    hetero, _, _ = _hetero()
    et = ("author", "writes", "paper")
    for b in LinkNeighborLoader(
        hetero, [2], edge_label_index=et, batch_size=8, neg_sampling="binary"
    ):
        assert b.input_id.dtype == b.edge_label_index.dtype == mx.int32
        for d in (b.n_id_dict, b.e_id_dict):
            assert all(v.dtype == mx.int32 for v in d.values())
        for key, v in b.edge_index_dict.items():
            assert v.dtype == hetero.edge_index_dict[key].dtype


def test_index_outputs_are_int32():
    data = random_graph()
    data.edge_index = data.edge_index.astype(mx.int32)
    for b in NeighborLoader(data, [2, 2], batch_size=8):
        assert b.n_id.dtype == b.e_id.dtype == mx.int32
        assert b.edge_index.dtype == mx.int32

    hetero, _, _ = _hetero()
    hetero.edge_index_dict = {
        k: v.astype(mx.int32) for k, v in hetero.edge_index_dict.items()
    }
    for b in NeighborLoader(hetero, [2, 2], input_nodes="author", batch_size=4):
        for d in (b.n_id_dict, b.e_id_dict, b.edge_index_dict):
            assert all(v.dtype == mx.int32 for v in d.values())
