from argparse import ArgumentParser
from time import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

try:
    from torch_geometric.datasets import Planetoid
except ImportError:
    raise ImportError("Run `pip install torch_geometric` to run this example.")

from mlx_graphs.nn import GCNConv


class GCN(nn.Module):
    """Graph Convolutional Network implementation [1]

    Args:
        node_features_dim (int): Size of input node features
        hidden_features_dim (int): Size of hidden node embeddings
        out_features_dim (int): Size of output node embeddings
        num_layers (int): Number of GCN layers
        dropout (float): Probability p for dropout
        bias (bool): Whether to use bias in the node projection

    References:
        [1] [Kipf et al. Semi-Supervised Classification with
            Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)
    """

    def __init__(
        self,
        node_features_dim: int,
        hidden_features_dim: int,
        out_features_dim: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        bias: bool = True,
    ):
        super(GCN, self).__init__()

        layer_sizes = (
            [node_features_dim]
            + [hidden_features_dim] * num_layers
            + [out_features_dim]
        )
        self.gcn_layers = [
            GCNConv(in_dim, out_features_dim, bias)
            for in_dim, out_features_dim in zip(layer_sizes[:-1], layer_sizes[1:])
        ]
        self.dropout = nn.Dropout(p=dropout)

    def __call__(self, edge_index: mx.array, node_features: mx.array) -> mx.array:
        for layer in self.gcn_layers[:-1]:
            node_features = nn.relu(layer(edge_index, node_features))
            node_features = self.dropout(node_features)

        node_features = self.gcn_layers[-1](edge_index, node_features)
        return node_features


def loss_fn(y_hat, y, weight_decay=0.0, parameters=None):
    loss = mx.mean(nn.losses.cross_entropy(y_hat, y))

    if weight_decay != 0.0:
        assert parameters is not None, "Model parameters missing for L2 reg."

        l2_reg = sum(mx.sum(p[1] ** 2) for p in tree_flatten(parameters)).sqrt()
        return loss + weight_decay * l2_reg

    return loss


def eval_fn(node_features, y):
    return mx.mean(mx.argmax(node_features, axis=1) == y)


def forward_fn(gcn, node_features, adj, y, train_mask, weight_decay):
    y_hat = gcn(adj, node_features)
    loss = loss_fn(y_hat[train_mask], y[train_mask], weight_decay, gcn.parameters())
    return loss, y_hat


def to_mlx(node_features, y, adj, train_mask, val_mask, test_mask):
    node_features = mx.array(node_features.tolist(), mx.float32)
    y = mx.array(y.tolist(), mx.int32)
    adj = mx.array(adj.tolist())
    train_mask = mx.array(train_mask.tolist())
    val_mask = mx.array(val_mask.tolist())
    test_mask = mx.array(test_mask.tolist())
    return node_features, y, adj, train_mask, val_mask, test_mask


def get_masks(train_mask, val_mask, test_mask):
    train_mask = mx.array([i for i, e in enumerate(train_mask) if e])
    val_mask = mx.array([i for i, e in enumerate(val_mask) if e])
    test_mask = mx.array([i for i, e in enumerate(test_mask) if e])

    return (train_mask, val_mask, test_mask)


def main(args):
    # Data loading
    dataset = Planetoid(root=".local_data/Cora", name="Cora")
    data = dataset[0]

    node_features, y, adj = data.x, data.y, data.edge_index
    train_mask, val_mask, test_mask = get_masks(
        data.train_mask, data.val_mask, data.test_mask
    )

    node_features, y, adj, train_mask, val_mask, test_mask = to_mlx(
        node_features, y, adj, train_mask, val_mask, test_mask
    )

    gcn = GCN(
        node_features_dim=node_features.shape[-1],
        hidden_features_dim=args.hidden_dim,
        out_features_dim=args.nb_classes,
        num_layers=args.nb_layers,
        dropout=args.dropout,
        bias=args.bias,
    )
    mx.eval(gcn.parameters())

    optimizer = optim.Adam(learning_rate=args.lr)
    loss_and_grad_fn = nn.value_and_grad(gcn, forward_fn)

    best_val_loss = float("inf")
    cnt = 0
    times = []

    # Training loop
    for epoch in range(args.epochs):
        start = time()

        # Loss
        (loss, y_hat), grads = loss_and_grad_fn(
            gcn, node_features, adj, y, train_mask, args.weight_decay
        )
        optimizer.update(gcn, grads)
        mx.eval(gcn.parameters(), optimizer.state)

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
    test_y_hat = gcn(adj, node_features)
    test_loss = loss_fn(test_y_hat[test_mask], y[test_mask])
    test_acc = eval_fn(test_y_hat[test_mask], y[test_mask])
    mean_time = sum(times) / len(times)

    print(f"Test loss: {test_loss.item():.3f}  |  Test acc: {test_acc.item():.2f}")
    print(f"Mean time: {mean_time:.5f}")
    return mean_time


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hidden_dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--nb_layers", type=int, default=2)
    parser.add_argument("--nb_classes", type=int, default=7)
    parser.add_argument("--bias", type=bool, default=True)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    main(args)
