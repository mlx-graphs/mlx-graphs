import cProfile
import pstats

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from mlx_graphs.datasets import TUDataset
from mlx_graphs.loaders import Dataloader
from mlx_graphs.nn import GCNConv, Linear, global_mean_pool

BATCH_SIZE = 1024
DATASET = "NCI-H23H"
SPLIT = 35_000

mx.set_default_device(mx.gpu)
mx.random.seed(42)


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


print("Processing dataset ...")
dataset = TUDataset(DATASET)
print("Done ...")
train_dataset = dataset[:SPLIT]
test_dataset = dataset[SPLIT:]

train_loader = Dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = Dataloader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = GCN(
    in_dim=dataset.num_node_features,
    hidden_dim=64,
    out_dim=dataset.num_graph_classes,
)

mx.eval(model.parameters())

optimizer = optim.Adam(learning_rate=0.01)
loss_and_grad_fn = nn.value_and_grad(model, forward_fn)


def train(loader):
    loss_sum = 0.0
    for graph in loader:
        (loss, y_hat), grads = loss_and_grad_fn(
            model=model,
            graph=graph,
            labels=graph.graph_labels,
        )
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        loss_sum += loss.item()
    return loss_sum / len(loader.dataset)


def test(loader):
    acc = 0.0
    for graph in loader:
        y_hat = model(graph.edge_index, graph.node_features, graph.batch_indices)
        y_hat = y_hat.argmax(axis=1)
        acc += (y_hat == graph.graph_labels).sum().item()

    return acc / len(loader.dataset)


print("Start profiling training loop")
profiler = cProfile.Profile()
profiler.enable()
train(train_loader)
profiler.disable()
print("Profiling completed ...")
stats = pstats.Stats(profiler).sort_stats("tottime")
stats.strip_dirs()
# stats.print_stats()
profiler.dump_stats("program.prof")
print("Results saved in program.prof")
