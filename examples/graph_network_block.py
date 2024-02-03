import mlx.core as mx
from mlx.nn.layers.base import Module
from mlx.nn.layers.linear import Linear

from mlx_graphs.nn.graph_network_block import GraphNetworkBlock


class NodeModel(Module):
    def __init__(
        self,
        node_features_dim: int,
        edge_features_dim: int,
        graph_features_dim: int,
        output_dim: int,
    ):
        super().__init__()
        self.model = Linear(
            input_dims=node_features_dim + edge_features_dim + graph_features_dim,
            output_dims=output_dim,
        )

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_features: mx.array,
        graph_features: mx.array,
    ):
        destination_nodes = edge_index[1]
        aggregated_edges = mx.zeros([node_features.shape[0], edge_features.shape[1]])
        for i in range(node_features.shape[0]):
            aggregated_edges[i] = mx.where(
                (destination_nodes == i).reshape(edge_features.shape[0], 1),
                edge_features,
                0,
            ).mean()
        model_input = mx.concatenate(
            [
                node_features,
                aggregated_edges,
                mx.ones([node_features.shape[0], graph_features.shape[0]])
                * graph_features,
            ],
            1,
        )
        return self.model(model_input)


class EdgeModel(Module):
    def __init__(
        self,
        edge_features_dim: int,
        node_features_dim: int,
        graph_features_dim: int,
        output_dim: int,
    ):
        super().__init__()
        self.model = Linear(
            input_dims=2 * node_features_dim + edge_features_dim + graph_features_dim,
            output_dims=output_dim,
        )

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_features: mx.array,
        graph_features: mx.array,
    ):
        source_nodes = edge_index[0]
        destination_nodes = edge_index[1]
        model_input = mx.concatenate(
            [
                node_features[destination_nodes],
                node_features[source_nodes],
                edge_features,
                mx.ones([edge_features.shape[0], graph_features.shape[0]])
                * graph_features,
            ],
            1,
        )
        return self.model(model_input)


class GlobalModel(Module):
    def __init__(
        self,
        edge_features_dim: int,
        node_features_dim: int,
        graph_features_dim: int,
        output_dim: int,
    ):
        super().__init__()
        self.model = Linear(
            input_dims=node_features_dim + edge_features_dim + graph_features_dim,
            output_dims=output_dim,
        )

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_features: mx.array,
        graph_features: mx.array,
    ):
        aggregated_edges = edge_features.mean(axis=0)
        aggregated_nodes = node_features.mean(axis=0)
        model_input = mx.concatenate(
            [aggregated_nodes, aggregated_edges, graph_features], 0
        )
        return self.model(model_input)


N = 4  # number of nodes
F_N = 2  # number of node features
F_E = 1  # number of edge features
F_U = 2  # number of global features

edge_index = mx.array([[0, 0, 1, 2, 3], [1, 2, 3, 3, 0]])
node_features = mx.random.normal([N, F_N])
edge_features = mx.random.normal([edge_index.shape[1], F_E])
graph_features = mx.random.normal([F_U])


# edge model
output_edge_feature_dim = F_E
edge_model = EdgeModel(
    edge_features_dim=F_E,
    node_features_dim=F_N,
    graph_features_dim=F_U,
    output_dim=output_edge_feature_dim,
)

# node model
output_node_features_dim = F_N
node_model = NodeModel(
    node_features_dim=F_N,
    edge_features_dim=output_edge_feature_dim,
    graph_features_dim=F_U,
    output_dim=output_node_features_dim,
)

# global_model
output_graph_features_dim = F_U
graph_model = GlobalModel(
    node_features_dim=output_node_features_dim,
    edge_features_dim=output_edge_feature_dim,
    graph_features_dim=F_U,
    output_dim=output_graph_features_dim,
)

# Graph Network block
gnn = GraphNetworkBlock(
    node_model=node_model, edge_model=edge_model, graph_model=graph_model
)

# forward pass
node_features, edge_features, graph_features = gnn(
    edge_index=edge_index,
    node_features=node_features,
    edge_features=edge_features,
    graph_features=graph_features,
)
