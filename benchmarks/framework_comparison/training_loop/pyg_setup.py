import torch
import torch.nn as torch_nn
import torch.nn.functional as F
import torch.optim
import torch_geometric.loader as pyg_loaders
import torch_geometric.nn as pyg_nn


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
