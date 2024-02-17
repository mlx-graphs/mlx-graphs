import dgl
import dgl.dataloading as dgl_loaders
import torch
import torch._dynamo
import torch.nn as torch_nn
import torch.nn.functional as F
import torch.optim


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
