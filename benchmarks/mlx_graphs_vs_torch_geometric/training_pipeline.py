import timeit

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import torch
import torch.nn.functional as F
from torch.nn import Linear as Linear_pyg
from torch_geometric.datasets import TUDataset as TUDataset_pyg
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv as GCNConv_pyg
from torch_geometric.nn import global_mean_pool as global_mean_pool_pyg

from mlx_graphs.datasets import TUDataset
from mlx_graphs.loaders import Dataloader
from mlx_graphs.nn import GCNConv, Linear, global_mean_pool

BATCH_SIZE = 256
DATASET = "NCI-H23H"
SPLIT = 35_000

TIMEIT_REPEAT = 10
TIMEIT_NUMBER = 1
mx.set_default_device(mx.gpu)


def pyg_pipeline():
    torch.manual_seed(42)
    print("Processing dataset ...")
    dataset = TUDataset_pyg(root="data/TUDataset", name=DATASET)
    print("Done ...")
    train_dataset = dataset[:SPLIT]
    test_dataset = dataset[SPLIT:]
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    class GCN(torch.nn.Module):
        def __init__(self, hidden_channels):
            super(GCN, self).__init__()
            self.conv1 = GCNConv_pyg(dataset.num_node_features, hidden_channels)
            self.conv2 = GCNConv_pyg(hidden_channels, hidden_channels)
            self.conv3 = GCNConv_pyg(hidden_channels, hidden_channels)
            self.lin = Linear_pyg(hidden_channels, dataset.num_classes)

        def forward(self, x, edge_index, batch):
            # 1. Obtain node embeddings
            x = self.conv1(x, edge_index)
            x = x.relu()
            x = self.conv2(x, edge_index)
            x = x.relu()
            x = self.conv3(x, edge_index)

            # 2. Readout layer
            x = global_mean_pool_pyg(x, batch)  # [batch_size, hidden_channels]

            # 3. Apply a final classifier
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.lin(x)

            return x

    model = GCN(hidden_channels=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    def train():
        model.train()

        for data in train_loader:  # Iterate in batches over the training dataset.
            out = model(
                data.x, data.edge_index, data.batch
            )  # Perform a single forward pass.
            loss = criterion(out, data.y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.

    def test(loader):
        model.eval()

        correct = 0
        for data in loader:  # Iterate in batches over the training/test dataset.
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        return correct / len(loader.dataset)  # Derive ratio of correct predictions.

    def epoch():
        train()
        test(train_loader)
        test(test_loader)

    print("Timing perfomance ...")
    times = timeit.Timer(lambda: epoch()).repeat(
        repeat=TIMEIT_REPEAT, number=TIMEIT_NUMBER
    )
    print("Done")
    return min(times) / TIMEIT_NUMBER


def mlx_pipeline():
    mx.random.seed(42)

    print("Processing dataset ...")
    dataset = TUDataset(DATASET)
    print("Done ...")
    train_dataset = dataset[:SPLIT]
    test_dataset = dataset[SPLIT:]

    train_loader = Dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = Dataloader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    class GCN(nn.Module):
        def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.5):
            super(GCN, self).__init__()

            self.conv1 = GCNConv(in_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.conv3 = GCNConv(hidden_dim, hidden_dim)
            self.linear = Linear(hidden_dim, out_dim)

            self.dropout = nn.Dropout(p=dropout)

        def __call__(self, edge_index, node_features, batch_indices):
            h = nn.relu(self.conv1(edge_index, node_features))
            h = nn.relu(self.conv2(edge_index, h))
            h = self.conv3(edge_index, h)

            h = global_mean_pool(h, batch_indices)

            h = self.dropout(h)
            h = self.linear(h)

            return h

    def loss_fn(y_hat, y, parameters=None):
        return mx.mean(nn.losses.cross_entropy(y_hat, y))

    def eval_fn(y_hat, y):
        return mx.mean(mx.argmax(y_hat, axis=1) == y)

    def forward_fn(model, graph, labels):
        y_hat = model(graph.edge_index, graph.node_features, graph.batch_indices)
        loss = loss_fn(y_hat, labels, model.parameters())
        return loss, y_hat

    model = GCN(
        in_dim=dataset.num_node_features,
        hidden_dim=64,
        out_dim=dataset.num_graph_classes,
    )

    mx.eval(model.parameters())

    optimizer = optim.Adam(learning_rate=0.01)
    loss_and_grad_fn = nn.value_and_grad(model, forward_fn)

    def train(train_loader):
        loss_sum = 0.0
        for graph in train_loader:
            (loss, y_hat), grads = loss_and_grad_fn(
                model=model,
                graph=graph,
                labels=graph.graph_labels,
            )
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            loss_sum += loss.item()
        return loss_sum / len(train_loader.dataset)

    def test(loader):
        acc = 0.0
        for graph in loader:
            y_hat = model(graph.edge_index, graph.node_features, graph.batch_indices)
            y_hat = y_hat.argmax(axis=1)
            acc += (y_hat == graph.graph_labels).sum().item()

        return acc / len(loader.dataset)

    def epoch():
        train(train_loader)
        test(train_loader)
        test(test_loader)

    print("Timing perfomance ...")
    times = timeit.Timer(lambda: epoch()).repeat(
        repeat=TIMEIT_REPEAT, number=TIMEIT_NUMBER
    )
    print("Done")
    return min(times) / TIMEIT_NUMBER


print(f"{'='*10} EVALUATING PYG {'='*10}")
pyg_min_epoch_time = pyg_pipeline()
print(f"{'='*10} EVALUATING MLX-GRAPHS {'='*10}")
mlx_min_epoch_time = mlx_pipeline()
print("\n\nResults")
print(f"{'='*20}")
print(f"PyG epoch duration: {pyg_min_epoch_time}s")
print(f"mlx-graphs epoch duration: {mlx_min_epoch_time}s")
