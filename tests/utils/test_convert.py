import mlx.core as mx

from mlx_graphs.data import GraphData
from mlx_graphs.datasets import KarateClubDataset
from mlx_graphs.utils.convert import from_networkx, to_networkx


def test_to_networkx():
    # GraphData with edge_index and node_features
    edge_index = mx.array([[0, 0, 1, 1, 2, 2, 3], [0, 1, 0, 2, 1, 3, 2]])
    node_features = mx.array([[1], [1], [1], [1]])

    graph = GraphData(node_features=node_features, edge_index=edge_index)

    networkx_graph = to_networkx(graph)

    assert networkx_graph.number_of_nodes() == 4
    assert networkx_graph.number_of_edges() == len(edge_index[0])

    for node in networkx_graph.nodes():
        assert (
            "features" in networkx_graph.nodes[node]
        ), f"Node {node} does not have 'features' attribute"

    for i in range(graph.num_nodes):
        assert mx.array_equal(
            mx.array(networkx_graph.nodes[i]["features"]), graph.node_features[i]
        ), f"Node {i} features do not match"

    networkx_graph_no_self_loops = to_networkx(graph, remove_self_loops=True)
    assert networkx_graph_no_self_loops.number_of_edges() == len(edge_index[0]) - 1

    # Without node features
    graph = GraphData(edge_index=mx.array([[0, 1], [1, 0]]))
    networkx_graph = to_networkx(graph)

    assert networkx_graph.number_of_nodes() == 2
    assert networkx_graph.number_of_edges() == 2


def test_from_networkx():
    karate_club_dataset = KarateClubDataset()
    karate_club_networkx_graph = to_networkx(karate_club_dataset.graphs[0])
    restored_karate_club_dataset = from_networkx(karate_club_networkx_graph)

    assert (
        restored_karate_club_dataset.num_nodes
        == karate_club_networkx_graph.number_of_nodes()
    ), "Number of nodes are not equal"

    assert (
        restored_karate_club_dataset.num_edges
        == karate_club_networkx_graph.number_of_edges()
    ), "Number of edges are not equal"

    assert mx.array_equal(
        restored_karate_club_dataset.node_features,
        karate_club_dataset[0].node_features,
    ), "Node features are not equal"
    assert (
        restored_karate_club_dataset.edge_index.shape[1]
        == karate_club_dataset[0].edge_index.shape[1]
    ), "Edge index shapes are not equal"
