from argparse import ArgumentParser
from time import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.nn.losses import nll_loss
from mlx.utils import tree_flatten

try:
    import torch_geometric.transforms as T
    from torch_geometric.datasets import Planetoid
except ImportError:
    raise ImportError("Run `pip install torch_geometric` to run this example.")

from mlx_graphs.nn.conv.gat_conv import GATConv


class GAT(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()

        self.conv1 = GATConv(in_channels, 8, heads=8, dropout=dropout)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(
            8 * 8, out_channels, heads=1, concat=False, dropout=dropout
        )
        self.dropout = nn.Dropout(p=dropout)

    def __call__(self, x, edge_index):
        x = self.dropout(x)
        x = nn.elu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return nn.log_softmax(x, axis=-1)


def loss_fn(y_hat, y, weight_decay=0.0, parameters=None):
    loss = nll_loss(y_hat, y, reduction="mean")

    if weight_decay != 0.0:
        assert parameters is not None, "Model parameters missing for L2 reg."

        l2_reg = sum(mx.sum(p[1] ** 2) for p in tree_flatten(parameters))
        return loss + weight_decay * l2_reg

    return loss


def eval_fn(x, y):
    return mx.mean(mx.argmax(x, axis=1) == y)


def forward_fn(gat, x, adj, y, train_mask, weight_decay):
    y_hat = gat(x, adj)
    loss = loss_fn(y_hat[train_mask], y[train_mask], weight_decay, gat.parameters())
    return loss, y_hat


def to_mlx(x, y, adj, train_mask, val_mask, test_mask):
    x = mx.array(x.tolist(), mx.float32)
    y = mx.array(y.tolist(), mx.int32)
    adj = mx.array(adj.tolist())
    train_mask = mx.array(train_mask.tolist())
    val_mask = mx.array(val_mask.tolist())
    test_mask = mx.array(test_mask.tolist())
    return x, y, adj, train_mask, val_mask, test_mask


def get_masks(train_mask, val_mask, test_mask):
    train_mask = mx.array([i for i, e in enumerate(train_mask) if e])
    val_mask = mx.array([i for i, e in enumerate(val_mask) if e])
    test_mask = mx.array([i for i, e in enumerate(test_mask) if e])

    return (train_mask, val_mask, test_mask)


def main(args):
    # Data loading
    dataset = Planetoid(root="data/Cora", name="Cora", transform=T.NormalizeFeatures())
    data = dataset[0]

    x, y, adj = data.x, data.y, data.edge_index
    train_mask, val_mask, test_mask = get_masks(
        data.train_mask, data.val_mask, data.test_mask
    )

    x, y, adj, train_mask, val_mask, test_mask = to_mlx(
        x, y, adj, train_mask, val_mask, test_mask
    )

    gat = GAT(
        in_channels=x.shape[-1],
        out_channels=args.nb_classes,
        dropout=args.dropout,
    )
    mx.eval(gat.parameters())

    optimizer = optim.Adam(learning_rate=args.lr)
    loss_and_grad_fn = nn.value_and_grad(gat, forward_fn)

    best_val_loss = float("inf")
    cnt = 0
    times = []

    # Training loop
    for epoch in range(args.epochs):
        start = time()

        # Loss
        (loss, y_hat), grads = loss_and_grad_fn(
            gat, x, adj, y, train_mask, args.weight_decay
        )
        optimizer.update(gat, grads)
        mx.eval(gat.parameters(), optimizer.state)

        # Validation
        val_loss = loss_fn(y_hat[val_mask], y[val_mask])
        val_acc = eval_fn(y_hat[val_mask], y[val_mask])

        times.append(time() - start)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            cnt = 0
        else:
            cnt += 1
            if cnt == args.patience:
                break

        print(
            " | ".join(
                [
                    f"Epoch: {epoch:3d}",
                    f"Train loss: {loss.item():.3f}",
                    f"Val loss: {val_loss.item():.3f}",
                    f"Val acc: {val_acc.item():.2f}",
                ]
            )
        )

    # Test
    test_y_hat = gat(x, adj)
    test_loss = loss_fn(test_y_hat[test_mask], y[test_mask])
    test_acc = eval_fn(test_y_hat[test_mask], y[test_mask])
    mean_time = sum(times) / len(times)

    print(f"Test loss: {test_loss.item():.3f}  |  Test acc: {test_acc.item():.2f}")
    print(f"Mean time: {mean_time:.5f}")
    return mean_time


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--nb_classes", type=int, default=7)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    main(args)
