import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from mlx_graphs.datasets import PlanetoidDataset
from mlx_graphs.nn import GCNConv, Linear, GraphNetworkBlock

class GCN(nn.Module):
    def __init__(self, in_dims, hidden_dims, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_dims, hidden_dims)
        self.conv2 = GCNConv(hidden_dims, num_classes)

    def __call__(self, edge_index, node_features):
        x = nn.relu(self.conv1(edge_index, node_features))
        x = self.conv2(edge_index, x)
        return x

def train_node_classification():
    dataset = PlanetoidDataset("Cora")
    graph = dataset[0]
    model = GCN(dataset.num_node_features, 16, dataset.num_classes)
    optimizer = optim.Adam(learning_rate=0.01)
    def loss_fn(model, edge_index, x, y, mask):
        logits = model(edge_index, x)
        loss = nn.losses.cross_entropy(logits[mask], y[mask])
        return mx.mean(loss)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    for epoch in range(10):
        loss, grads = loss_and_grad_fn(model, graph.edge_index, graph.node_features, graph.node_labels, graph.train_mask)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        print(f"Epoch {epoch} | Node Loss: {loss.item():.4f}")

class MyEdgeModel(nn.Module):
    def __init__(self, node_dims, edge_dims, out_dims):
        super().__init__()
        self.linear = Linear(2 * node_dims + edge_dims, out_dims)
    def __call__(self, edge_index, node_features, edge_features, graph_features=None):
        src, dst = edge_index
        out = mx.concatenate([node_features[src], node_features[dst], edge_features], axis=-1)
        return nn.relu(self.linear(out))

class EdgeClassifier(nn.Module):
    def __init__(self, node_dims, edge_dims, num_classes):
        super().__init__()
        self.gnn = GraphNetworkBlock(edge_model=MyEdgeModel(node_dims, edge_dims, 32))
        self.classifier = Linear(32, num_classes)
    def __call__(self, edge_index, node_features, edge_features):
        _, updated_edge_features, _ = self.gnn(edge_index, node_features, edge_features)
        return self.classifier(updated_edge_features)

def train_edge_classification():
    num_nodes, num_edges, node_dims, edge_dims, num_classes = 10, 20, 16, 8, 3
    edge_index = mx.random.randint(0, num_nodes, (2, num_edges))
    node_features = mx.random.normal((num_nodes, node_dims))
    edge_features = mx.random.normal((num_edges, edge_dims))
    edge_labels = mx.random.randint(0, num_classes, (num_edges,))
    model = EdgeClassifier(node_dims, edge_dims, num_classes)
    optimizer = optim.Adam(learning_rate=0.01)
    def loss_fn(model, ei, nf, ef, y):
        logits = model(ei, nf, ef)
        return mx.mean(nn.losses.cross_entropy(logits, y))
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    for epoch in range(10):
        loss, grads = loss_and_grad_fn(model, edge_index, node_features, edge_features, edge_labels)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        print(f"Epoch {epoch} | Edge Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train_node_classification()
    train_edge_classification()
