import mlx.core as mx
import numpy as np
import pytest

from mlx_graphs.data import GraphData, HeteroGraphData
from mlx_graphs.loaders import NeighborLoader
from mlx_graphs.nn import SAGEConv

WRITES = ("author", "writes", "paper")
REV_WRITES = ("paper", "rev_writes", "author")
CITES = ("paper", "cites", "paper")
PUB = ("paper", "published_in", "conference")


def _make_graph():
    """4 authors, 5 papers, 2 conferences (no features).

    writes:       a0->p0, a0->p1, a1->p1, a2->p2, a3->p3, a3->p4
    cites:        p1->p0, p2->p0, p3->p1, p4->p2
    published_in: p0,p1->c0; p2,p3,p4->c1
    """
    writes = mx.array([[0, 0, 1, 2, 3, 3], [0, 1, 1, 2, 3, 4]], dtype=mx.int64)
    data = HeteroGraphData(
        edge_index_dict={
            WRITES: writes,
            REV_WRITES: writes[::-1],
            CITES: mx.array([[1, 2, 3, 4], [0, 0, 1, 2]], dtype=mx.int64),
            PUB: mx.array([[0, 1, 2, 3, 4], [0, 0, 1, 1, 1]], dtype=mx.int64),
        },
        node_features_dict={
            "author": mx.arange(4 * 3, dtype=mx.float32).reshape(4, 3),
            "paper": mx.arange(5 * 2, dtype=mx.float32).reshape(5, 2) + 100,
        },
        edge_features_dict={
            WRITES: mx.arange(6, dtype=mx.float32).reshape(6, 1) + 10,
        },
        node_labels_dict={"author": mx.array([0, 1, 0, 1])},
    )
    data.author_train_mask = mx.array([True, True, False, True])
    data.paper_year = mx.array([2001, 2002, 2003, 2004, 2005])
    data.split_dict = {"author": mx.array([0, 1, 2, 0])}
    data.unrelated_vector = mx.arange(7)  # no matching node type, not sliced
    return data


def _global_edges(batch, et):
    src_ids = np.array(batch.n_id_dict[et[0]])
    dst_ids = np.array(batch.n_id_dict[et[2]])
    ei = np.array(batch.edge_index_dict[et])
    return set(zip(src_ids[ei[0]].tolist(), dst_ids[ei[1]].tolist()))


def _original_edges(data, et):
    ei = np.array(data.edge_index_dict[et])
    return set(zip(ei[0].tolist(), ei[1].tolist()))


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------


def test_returns_hetero_graph_data():
    data = _make_graph()
    loader = NeighborLoader(data, [2], input_nodes="author", batch_size=2)
    batch = next(iter(loader))
    assert isinstance(batch, HeteroGraphData)
    assert batch.batch_size == 2
    assert batch.input_type == "author"


def test_seeds_come_first():
    data = _make_graph()
    seeds = mx.array([3, 1], dtype=mx.int64)
    loader = NeighborLoader(data, [2, 2], input_nodes=("author", seeds), batch_size=2)
    batch = next(iter(loader))
    assert batch.n_id_dict["author"][:2].tolist() == [3, 1]


def test_every_edge_type_present_even_if_empty():
    data = _make_graph()
    # 1 hop from authors only reaches papers via rev_writes
    loader = NeighborLoader(data, [2], input_nodes="author", batch_size=4)
    batch = next(iter(loader))
    assert set(batch.edge_index_dict) == set(data.edge_index_dict)
    assert batch.edge_index_dict[PUB].shape == (2, 0)
    assert batch.edge_index_dict[PUB].dtype == mx.int64


def test_featureless_node_type_has_ids_and_count():
    data = _make_graph()
    loader = NeighborLoader(data, [-1, -1], input_nodes="author", batch_size=4)
    batch = next(iter(loader))
    assert "conference" in batch.n_id_dict
    assert batch.num_nodes["conference"] == batch.n_id_dict["conference"].shape[0]
    assert batch.num_nodes["author"] == batch.n_id_dict["author"].shape[0]


# ---------------------------------------------------------------------------
# Correctness of sampled edges and sliced attributes
# ---------------------------------------------------------------------------


def test_full_two_hop_neighborhood():
    """Fan-out -1 is deterministic: author 0 -> its papers -> their citers."""
    data = _make_graph()
    seeds = mx.array([0], dtype=mx.int64)
    loader = NeighborLoader(data, [-1, -1], input_nodes=("author", seeds), batch_size=1)
    batch = next(iter(loader))

    # hop 1: a0 <- p0, p1 (rev_writes)
    # hop 2: p0 <- a0 (writes), p1 <- a0, a1 (writes); p0 <- p1, p2 and
    #        p1 <- p3 (cites)
    assert _global_edges(batch, REV_WRITES) == {(0, 0), (1, 0)}
    assert _global_edges(batch, WRITES) == {(0, 0), (0, 1), (1, 1)}
    assert _global_edges(batch, CITES) == {(1, 0), (2, 0), (3, 1)}
    assert set(batch.n_id_dict["author"].tolist()) == {0, 1}
    assert set(batch.n_id_dict["paper"].tolist()) == {0, 1, 2, 3}


def test_sampled_edges_exist_and_e_id_points_to_them():
    data = _make_graph()
    mx.random.seed(0)
    loader = NeighborLoader(
        data, [2, 2], input_nodes="author", batch_size=2, shuffle=True
    )
    for batch in loader:
        for et in data.edge_index_dict:
            assert _global_edges(batch, et) <= _original_edges(data, et)
            # e_id: column of each sampled edge in the original edge_index
            orig = np.array(data.edge_index_dict[et])[
                :, np.array(batch.e_id_dict[et], dtype=np.int64)
            ]
            src_ids = np.array(batch.n_id_dict[et[0]])
            dst_ids = np.array(batch.n_id_dict[et[2]])
            ei = np.array(batch.edge_index_dict[et])
            assert np.array_equal(orig[0], src_ids[ei[0]])
            assert np.array_equal(orig[1], dst_ids[ei[1]])


def test_node_features_and_labels_sliced():
    data = _make_graph()
    loader = NeighborLoader(data, [-1, -1], input_nodes="author", batch_size=2)
    batch = next(iter(loader))
    for t in ("author", "paper"):
        assert mx.array_equal(
            batch.node_features_dict[t],
            data.node_features_dict[t][batch.n_id_dict[t]],
        )
    assert mx.array_equal(
        batch.node_labels_dict["author"],
        data.node_labels_dict["author"][batch.n_id_dict["author"]],
    )
    assert "conference" not in batch.node_features_dict


def test_edge_features_sliced():
    data = _make_graph()
    loader = NeighborLoader(data, [-1, -1], input_nodes="author", batch_size=4)
    batch = next(iter(loader))
    assert mx.array_equal(
        batch.edge_features_dict[WRITES],
        data.edge_features_dict[WRITES][batch.e_id_dict[WRITES]],
    )


def test_custom_attributes_sliced():
    data = _make_graph()
    loader = NeighborLoader(data, [-1, -1], input_nodes="author", batch_size=2)
    batch = next(iter(loader))
    a_ids, p_ids = batch.n_id_dict["author"], batch.n_id_dict["paper"]
    assert mx.array_equal(batch.author_train_mask, data.author_train_mask[a_ids])
    assert mx.array_equal(batch.paper_year, data.paper_year[p_ids])
    assert mx.array_equal(batch.split_dict["author"], data.split_dict["author"][a_ids])
    assert not hasattr(batch, "unrelated_vector")


def test_per_hop_counts():
    data = _make_graph()
    loader = NeighborLoader(data, [-1, -1], input_nodes="author", batch_size=1)
    batch = next(iter(loader))
    for t, counts in batch.num_sampled_nodes_dict.items():
        assert sum(counts) == batch.n_id_dict[t].shape[0]
    for et, counts in batch.num_sampled_edges_dict.items():
        assert sum(counts) == batch.edge_index_dict[et].shape[1]


# ---------------------------------------------------------------------------
# Input nodes, iteration, reproducibility
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_nodes, expected",
    [
        ("author", [0, 1, 2, 3]),
        (("author", mx.array([2, 0])), [2, 0]),
        (("author", mx.array([True, True, False, True])), [0, 1, 3]),
    ],
)
def test_input_node_forms(input_nodes, expected):
    data = _make_graph()
    loader = NeighborLoader(data, [1], input_nodes=input_nodes, batch_size=1)
    seeds = [batch.n_id_dict["author"][0].item() for batch in loader]
    assert seeds == expected
    assert len(loader) == len(expected)


def test_shuffle_covers_every_seed_once():
    data = _make_graph()
    loader = NeighborLoader(data, [1], input_nodes="paper", batch_size=2, shuffle=True)
    seen = []
    for batch in loader:
        seen += batch.n_id_dict["paper"][: batch.batch_size].tolist()
    assert sorted(seen) == [0, 1, 2, 3, 4]
    assert len(loader) == 3


def test_reproducible_with_mx_random_seed():
    data = _make_graph()

    def run():
        mx.random.seed(42)
        loader = NeighborLoader(data, [1, 1], input_nodes="author", batch_size=4)
        batch = next(iter(loader))
        return {et: batch.e_id_dict[et].tolist() for et in batch.e_id_dict}

    assert run() == run()


def test_unknown_input_type():
    with pytest.raises(ValueError, match="Unknown input node type"):
        NeighborLoader(_make_graph(), [1], input_nodes="venue")


def test_mask_wrong_length():
    with pytest.raises(ValueError, match="Boolean mask"):
        NeighborLoader(
            _make_graph(), [1], input_nodes=("author", mx.array([True, False]))
        )


# ---------------------------------------------------------------------------
# Works with a bipartite SAGEConv on the sampled batch
# ---------------------------------------------------------------------------


def test_batch_feeds_bipartite_sage_conv():
    data = _make_graph()
    loader = NeighborLoader(data, [-1, -1], input_nodes="author", batch_size=2)
    batch = next(iter(loader))

    x = batch.node_features_dict
    conv = SAGEConv((2, 3), 8)  # papers (2 features) -> authors (3 features)
    h_author = conv(batch.edge_index_dict[REV_WRITES], (x["paper"], x["author"]))

    assert h_author.shape == (batch.n_id_dict["author"].shape[0], 8)


# ---------------------------------------------------------------------------
# NeighborLoader dispatches on the data type
# ---------------------------------------------------------------------------


def test_dispatches_on_data_type():
    hetero = next(iter(NeighborLoader(_make_graph(), [1], input_nodes="author")))
    assert isinstance(hetero, HeteroGraphData)

    homo_data = GraphData(edge_index=mx.array([[0, 1], [1, 2]], dtype=mx.int64))
    homo = next(iter(NeighborLoader(homo_data, [1])))
    assert isinstance(homo, GraphData)


def test_hetero_requires_input_node_type():
    with pytest.raises(ValueError, match="seed node type"):
        NeighborLoader(_make_graph(), [1])
    with pytest.raises(ValueError, match="seed node type"):
        NeighborLoader(_make_graph(), [1], input_nodes=mx.array([0, 1]))


def test_hetero_rejects_undirected():
    with pytest.raises(ValueError, match="directed"):
        NeighborLoader(_make_graph(), [1], input_nodes="author", directed=False)


def test_homogeneous_rejects_hetero_arguments():
    data = GraphData(edge_index=mx.array([[0, 1], [1, 2]], dtype=mx.int64))
    with pytest.raises(ValueError, match="Node types"):
        NeighborLoader(data, [1], input_nodes="author")
    with pytest.raises(ValueError, match="dict"):
        NeighborLoader(data, {("a", "to", "b"): [1]})


def test_rejects_unknown_data_type():
    with pytest.raises(TypeError, match="GraphData or a HeteroGraphData"):
        NeighborLoader({"edge_index": None}, [1])
