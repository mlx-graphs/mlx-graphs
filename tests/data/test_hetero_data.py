import mlx.core as mx

from mlx_graphs.data import HeteroGraphData


def test_hetero_graph_data():
    hetero_data = HeteroGraphData(
        edge_index_dict={
            ("author", "writes", "paper"): mx.array([[0, 1], [1, 2]]),
            ("paper", "cites", "paper"): mx.array([[0, 1], [1, 2]]),
        },
        node_features_dict={
            "author": mx.array([[1, 2], [3, 4], [5, 6]]),
            "paper": mx.array([[7, 8], [9, 10], [11, 12]]),
        },
        edge_features_dict={
            ("author", "writes", "paper"): mx.array([[13, 14], [15, 16]]),
            ("paper", "cites", "paper"): mx.array([[17, 18], [19, 20]]),
        },
        graph_features=mx.array([21, 22]),
        node_labels_dict={"author": mx.array([0, 1, 2]), "paper": mx.array([0, 1, 2])},
        edge_labels_dict={
            ("author", "writes", "paper"): mx.array([0, 1]),
            ("paper", "cites", "paper"): mx.array([1, 0]),
        },
        graph_labels=mx.array([1]),
    )
    expected_num_nodes = {"author": 3, "paper": 3}
    assert hetero_data.num_nodes == expected_num_nodes
    expected_num_edges_dict = {
        ("author", "writes", "paper"): 2,
        ("paper", "cites", "paper"): 2,
    }
    assert hetero_data.num_edges == expected_num_edges_dict
    expected_num_edge_classes_dict = {
        ("author", "writes", "paper"): 2,
        ("paper", "cites", "paper"): 2,
    }
    assert hetero_data.num_edge_classes == expected_num_edge_classes_dict

    expected_num_nodes = {"author": 3, "paper": 3}
    assert (
        hetero_data.num_nodes == expected_num_nodes
    ), "Expected number of nodes do not match"
    assert (
        hetero_data.num_node_classes["author"] == 3
    ), "Number of authors are different"
    assert (
        hetero_data.num_edge_features[("author", "writes", "paper")] == 2
    ), "edge_features are different"

    assert (
        hetero_data.num_edge_features[("paper", "cites", "paper")] == 2
    ), "edge_features for paper are different"

    assert (
        hetero_data.num_node_classes["author"] == 3
    ), "node classes for author are different"

    assert (
        hetero_data.num_node_classes["paper"] == 3
    ), "node classes for paper are different"

    assert hetero_data.is_directed() is True, "Graph is not directed"

    heteroGraphDict = hetero_data.to_dict()

    assert (
        heteroGraphDict["edge_index_dict"][("author", "writes", "paper")].shape[0] == 2
    ), "The edge dictionary is different"

    assert (
        heteroGraphDict["edge_index_dict"][("paper", "cites", "paper")].shape[0] == 2
    ), "The edge dictionary is different"


# ---------------------------------------------------------------------------
# num_nodes, node_types and to_csc_dict
# ---------------------------------------------------------------------------

import numpy as np  # noqa: E402
import pytest  # noqa: E402


def _graph_with_featureless_type(**kwargs):
    # 3 authors, 4 papers (features), conferences have no features
    return HeteroGraphData(
        edge_index_dict={
            ("author", "writes", "paper"): mx.array([[0, 0, 2], [1, 3, 3]]),
            ("paper", "in", "conference"): mx.array([[0, 1, 3], [1, 0, 1]]),
        },
        node_features_dict={
            "author": mx.ones((3, 2)),
            "paper": mx.ones((4, 5)),
        },
        **kwargs,
    )


def test_num_nodes_includes_featureless_node_types():
    data = _graph_with_featureless_type()
    assert data.node_types == ["author", "paper", "conference"]
    # conference: largest id used is 1 -> 2 nodes
    assert data.num_nodes == {"author": 3, "paper": 4, "conference": 2}


def test_num_nodes_uses_feature_rows_not_edge_ids():
    # Author 2 and paper 0/2 have no edges; counts must still come from features
    data = _graph_with_featureless_type()
    assert data.num_nodes["author"] == 3
    assert data.num_nodes["paper"] == 4


def test_num_nodes_explicit_override():
    # 5 conferences, even though only ids 0 and 1 appear in edges
    data = _graph_with_featureless_type(num_nodes_dict={"conference": 5})
    assert data.num_nodes == {"author": 3, "paper": 4, "conference": 5}


def test_num_nodes_without_features_uses_max_id_plus_one():
    data = HeteroGraphData(
        edge_index_dict={("a", "to", "b"): mx.array([[0, 7], [2, 2]])},
    )
    # unique ids would give a=2, b=1; correct counts are max id + 1
    assert data.num_nodes == {"a": 8, "b": 3}


def test_to_csc_dict_matches_edges():
    data = _graph_with_featureless_type()
    colptr_dict, row_dict, perm_dict = data.to_csc_dict()

    for et, edge_index in data.edge_index_dict.items():
        src, dst = np.array(edge_index[0]), np.array(edge_index[1])
        colptr = np.array(colptr_dict[et])
        row = np.array(row_dict[et])
        perm = np.array(perm_dict[et])

        assert colptr_dict[et].dtype == mx.int64 and row_dict[et].dtype == mx.int64
        assert len(colptr) == data.num_nodes[et[2]] + 1
        assert colptr[-1] == len(src)
        # perm maps CSC position -> original edge position
        assert np.array_equal(row, src[perm])
        # incoming edges of each destination are in its colptr range
        for v in range(len(colptr) - 1):
            assert np.all(dst[perm[colptr[v] : colptr[v + 1]]] == v)


def test_to_csc_dict_rejects_ids_out_of_range():
    data = HeteroGraphData(
        edge_index_dict={("a", "to", "b"): mx.array([[0, 5], [0, 1]])},
        node_features_dict={"a": mx.ones((2, 1)), "b": mx.ones((2, 1))},
    )
    with pytest.raises(ValueError, match="source ids"):
        data.to_csc_dict()
