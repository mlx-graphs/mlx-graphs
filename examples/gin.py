
from mlx_graphs.nn.conv.gin_conv import GINConv
import mlx.core as mx
import mlx.nn as nn


# default settings for the example
node_feat_dim = 16
edge_feat_dim = 10
out_feat_dim = 32

mlp = nn.Sequential(
    nn.Linear(node_feat_dim, node_feat_dim * 2),
    nn.ReLU(),
    nn.Linear(node_feat_dim * 2, out_feat_dim),
)

# original GINConv with only node features

conv = GINConv(mlp)

edge_index = mx.array([[0, 1, 2, 3, 4], [0, 0, 1, 1, 3]])
node_features = mx.random.uniform(low=0, high=1, shape=(5, 16))

h = conv(edge_index, node_features)
print('original GINConv with only node features')
print(h)


# GINConv with node and edges features
conv = GINConv(mlp, edge_features_dim=edge_feat_dim, node_features_dim=node_feat_dim)

edge_index = mx.array([[0, 1, 2, 3, 4], [0, 0, 1, 1, 3]])
node_features = mx.random.uniform(low=0, high=1, shape=(5, node_feat_dim))
edge_features = mx.random.uniform(low=0, high=1, shape=(5, edge_feat_dim))


h = conv(edge_index, node_features, edge_features)
print('advanced GINConv with node and edges features')
print(h)




