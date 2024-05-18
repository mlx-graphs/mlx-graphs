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
