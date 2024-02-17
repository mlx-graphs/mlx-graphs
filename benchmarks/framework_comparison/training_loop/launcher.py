import timeit
from functools import partial

import dgl
import dgl.data as dgl_datasets
import dgl.dataloading as dgl_loaders
import dgl.nn.pytorch as dgl_nn
import mlx.core as mx
import mlx.nn as mlx_nn
import mlx.optimizers as mlx_optim
import torch
import torch._dynamo
import torch.nn as torch_nn
import torch.nn.functional as F
import torch.optim
import torch_geometric.datasets as pyg_datasets
import torch_geometric.loader as pyg_loaders
import torch_geometric.nn as pyg_nn

import mlx_graphs.datasets as mxg_datasets
import mlx_graphs.loaders as mxg_loaders
import mlx_graphs.nn as mxg_nn

torch._dynamo.config.suppress_errors = True


# mlx-graphs
class MXG_model(mlx_nn.Module):
    def __init__(self, layer, in_dim, hidden_dim, out_dim, dropout=0.5):
        super(MXG_model, self).__init__()

        self.conv1 = layer(in_dim, hidden_dim)
        self.conv2 = layer(hidden_dim, hidden_dim)
        self.conv3 = layer(hidden_dim, hidden_dim)
        self.linear = mxg_nn.Linear(hidden_dim, out_dim)

        self.dropout = mlx_nn.Dropout(p=dropout)

    def __call__(self, edge_index, node_features, batch_indices):
        h = mlx_nn.relu(self.conv1(edge_index, node_features))
        h = mlx_nn.relu(self.conv2(edge_index, h))
        h = self.conv3(edge_index, h)

        h = mxg_nn.global_mean_pool(h, batch_indices)

        h = self.dropout(h)
        h = self.linear(h)

        return h


def loss_fn(y_hat, y, parameters=None):
    return mlx_nn.losses.cross_entropy(y_hat, y, reduction="mean")


def forward_fn(model, graph):
    y_hat = model(graph.edge_index, graph.node_features, graph.batch_indices)
    labels = graph.graph_labels
    loss = loss_fn(y_hat, labels, model.parameters())
    return loss, y_hat


def setup_training_mxg(dataset, layer, batch_size, hid_size, compile=True):
    loader = mxg_loaders.Dataloader(dataset, batch_size=batch_size, shuffle=True)

    model = MXG_model(
        layer=layer,
        in_dim=dataset.num_node_features,
        hidden_dim=hid_size,
        out_dim=dataset.num_graph_classes,
    )
    mx.eval(model.parameters())

    optimizer = mlx_optim.Adam(learning_rate=0.01)
    loss_and_grad_fn = mlx_nn.value_and_grad(model, forward_fn)

    state = [model.state, optimizer.state, mx.random.state]

    if compile:

        @partial(mx.compile, inputs=state, outputs=state)
        def step(graph):
            (loss, y_hat), grads = loss_and_grad_fn(model=model, graph=graph)
            optimizer.update(model, grads)
    else:

        def step(graph):
            (loss, y_hat), grads = loss_and_grad_fn(model=model, graph=graph)
            optimizer.update(model, grads)

    return loader, step, state


def train_mxg(loader, step, state=None, epochs=2):
    for _ in range(epochs):
        for graph in loader:
            step(graph)
            mx.eval(state)


# PyG
class PyG_model(torch.nn.Module):
    def __init__(self, layer, in_dim, hidden_dim, out_dim):
        super(PyG_model, self).__init__()

        self.conv1 = layer(in_dim, hidden_dim)
        self.conv2 = layer(hidden_dim, hidden_dim)
        self.conv3 = layer(hidden_dim, hidden_dim)
        self.lin = torch_nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        x = pyg_nn.global_mean_pool(x, batch)

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


def setup_training_pyg(dataset, layer, batch_size, hid_size, compile=True):
    loader = pyg_loaders.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = PyG_model(
        layer=layer,
        in_dim=dataset.num_node_features,
        hidden_dim=hid_size,
        out_dim=dataset.num_classes,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch_nn.CrossEntropyLoss()

    model.train()

    def step(data):
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if compile:
        step = torch.compile(step, dynamic=True)

    return loader, step, None


def train_pyg(loader, step, state=None, epochs=2):
    for _ in range(epochs):
        for data in loader:
            step(data)


# dgl
class DGL_model(torch_nn.Module):
    def __init__(self, layer, in_dim, hidden_dim, out_dim):
        super(DGL_model, self).__init__()

        if "GATConv" in str(layer):
            self.conv1 = layer(
                in_dim, hidden_dim, num_heads=1, allow_zero_in_degree=True
            )
            self.conv2 = layer(
                hidden_dim, hidden_dim, num_heads=1, allow_zero_in_degree=True
            )
            self.conv3 = layer(
                hidden_dim, hidden_dim, num_heads=1, allow_zero_in_degree=True
            )
        else:
            self.conv1 = layer(in_dim, hidden_dim, allow_zero_in_degree=True)
            self.conv2 = layer(hidden_dim, hidden_dim, allow_zero_in_degree=True)
            self.conv3 = layer(hidden_dim, hidden_dim, allow_zero_in_degree=True)

        self.classify = torch_nn.Linear(hidden_dim, out_dim)

    def forward(self, g, h):
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        h = F.relu(self.conv3(g, h))
        with g.local_scope():
            g.ndata["h"] = h
            hg = dgl.mean_nodes(g, "h")
            return self.classify(hg.squeeze())


def setup_training_dgl(dataset, layer, batch_size, hid_size, compile=True):
    loader = dgl_loaders.GraphDataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )

    model = DGL_model(
        layer=layer,
        in_dim=dataset[0][0].ndata["x"].shape[1],
        hidden_dim=hid_size,
        out_dim=dataset.num_classes,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch_nn.CrossEntropyLoss()

    model.train()

    def step(data, labels):
        out = model(data, data.ndata["x"])
        loss = criterion(out, labels.squeeze())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if compile:
        step = torch.compile(step, dynamic=True)

    return loader, step, None


def train_dgl(loader, step, state=None, epochs=2):
    for _ in range(epochs):
        for data, labels in loader:
            step(data, labels)


# Benchmark

framework_to_setup = {
    "mxg": setup_training_mxg,
    "pyg": setup_training_pyg,
    "dgl": setup_training_dgl,
}

framework_to_train = {
    "mxg": train_mxg,
    "pyg": train_pyg,
    "dgl": train_dgl,
}


def dgl_dataset(name):
    pyg_dataset = pyg_datasets.TUDataset(f".mlx_graphs_data/{name}", name)
    dgl_dataset = dgl_datasets.TUDataset(dataset_name)

    for i, (pyg_p, dgl_p) in enumerate(zip(pyg_dataset, dgl_dataset.graph_lists)):
        dgl_dataset.graph_lists[i].ndata["x"] = pyg_p.x

    return dgl_dataset


framework_to_datasets = {
    "mxg": lambda name: mxg_datasets.TUDataset(name),
    "pyg": lambda name: pyg_datasets.TUDataset(f".mlx_graphs_data/{name}", name),
    "dgl": lambda name: dgl_dataset(name),
}
layer_classes = {
    "mxg": {
        "GCNConv": mxg_nn.GCNConv,
        "GATConv": mxg_nn.GATConv,
    },
    "pyg": {
        "GCNConv": pyg_nn.GCNConv,
        "GATConv": pyg_nn.GATConv,
    },
    "dgl": {
        "GCNConv": dgl_nn.GraphConv,
        "GATConv": dgl_nn.GATConv,
    },
}

frameworks = ["dgl", "pyg", "mxg"]
datasets = ["BZR_MD"]  # ["BZR_MD", "MUTAG", "DD"]#, "NCI-H23"]
layers = ["GCNConv", "GATConv"]

batch_size = 64
hid_size = 128

TIMEIT_REPEAT = 5
TIMEIT_NUMBER = 1
COMPILE = False

torch.manual_seed(42)
mx.random.seed(42)
dgl.seed(42)

mx.set_default_device(mx.gpu)


def benchmark(framework, loader, step, state):
    train_fn = framework_to_train[framework]
    train_fn(loader, step, state)


for dataset_name in datasets:
    print(dataset_name)
    print("=" * 10)

    for framework in frameworks:
        dataset = framework_to_datasets[framework](dataset_name)

        for i, layer_name in enumerate(layers):
            layer = layer_classes[framework][layer_name]
            loader, step, state = framework_to_setup[framework](
                dataset, layer, batch_size, hid_size, compile=COMPILE
            )

            times = timeit.Timer(
                lambda: benchmark(framework, loader, step, state)
            ).repeat(repeat=TIMEIT_REPEAT, number=TIMEIT_NUMBER)

            time = min(times) / TIMEIT_NUMBER

            print(
                " | ".join(
                    [
                        f"{framework}",
                        f"{layer_name}",
                        f"{time:.3f}s",
                    ]
                )
            )
        print("")
