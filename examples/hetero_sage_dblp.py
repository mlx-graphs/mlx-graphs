"""Heterogeneous GraphSAGE on DBLP with mini-batch neighbor sampling.

Classifies DBLP authors into 4 research areas. DBLP has four node types
(author, paper, term, conference) and edges between authors and papers,
papers and terms, and papers and conferences, in both directions.

The model:

1. ``HeteroDictLinear`` projects every node type to ``hidden_dim`` features
   (node types have different feature sizes; conferences have none, so each
   gets a single constant feature ``1.0``, as in PyG's DBLP examples).
2. ``num_layers`` ``HeteroConv`` layers run one ``SAGEConv`` per edge type and
   sum the results per node type.
3. A linear layer predicts the research area of each author.

Training uses ``NeighborLoader`` (backed by
``mlx_cluster.hetero_neighbor_sample``): each mini-batch is the sampled
``num_layers``-hop neighborhood of ``batch_size`` training authors, and the
loss is computed on those seed authors only.

Run::

    python examples/hetero_sage_dblp.py
    python examples/hetero_sage_dblp.py --hidden_dim 128 --num_neighbors 10,5
"""

from argparse import ArgumentParser
from time import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from mlx_graphs.data import HeteroGraphData
from mlx_graphs.datasets import DBLP
from mlx_graphs.loaders import NeighborLoader
from mlx_graphs.nn import HeteroConv, HeteroDictLinear, SAGEConv


class HeteroSAGE(nn.Module):
    """Project-first heterogeneous GraphSAGE.

    Args:
        node_features_dims: Number of input features per node type.
        edge_types: The ``(src, rel, dst)`` edge types of the graph.
        hidden_dim: Size of the node embeddings after projection.
        out_dim: Number of classes.
        target: Node type to classify.
        num_layers: Number of ``HeteroConv`` layers.
        dropout: Dropout probability applied after every layer.
    """

    def __init__(
        self,
        node_features_dims: dict[str, int],
        edge_types: list[tuple[str, str, str]],
        hidden_dim: int,
        out_dim: int,
        target: str,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.target = target
        self.proj = HeteroDictLinear(node_features_dims, hidden_dim)
        self.convs = [
            HeteroConv(
                {et: SAGEConv(hidden_dim, hidden_dim) for et in edge_types},
                aggr="sum",
            )
            for _ in range(num_layers)
        ]
        self.dropout = nn.Dropout(p=dropout)
        self.head = nn.Linear(hidden_dim, out_dim)

    def __call__(
        self,
        x_dict: dict[str, mx.array],
        edge_index_dict: dict[tuple[str, str, str], mx.array],
    ) -> mx.array:
        h = {k: self.dropout(nn.relu(v)) for k, v in self.proj(x_dict).items()}
        for conv in self.convs:
            h = {
                k: self.dropout(nn.relu(v)) for k, v in conv(h, edge_index_dict).items()
            }
        return self.head(h[self.target])


def add_constant_features(data: HeteroGraphData) -> HeteroGraphData:
    """Gives every node type without features a single constant feature 1.0."""
    for node_type, num_nodes in data.num_nodes.items():
        if node_type not in data.node_features_dict:
            data.node_features_dict[node_type] = mx.ones((num_nodes, 1))
    return data


def loss_fn(model, batch, target):
    n = batch.batch_size
    logits = model(batch.node_features_dict, batch.edge_index_dict)
    labels = batch.node_labels_dict[target][:n]
    return nn.losses.cross_entropy(logits[:n], labels, reduction="mean")


def evaluate(model, loader, target):
    """Accuracy and loss on the seed nodes of every batch in ``loader``."""
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    for batch in loader:
        n = batch.batch_size
        logits = model(batch.node_features_dict, batch.edge_index_dict)[:n]
        labels = batch.node_labels_dict[target][:n]
        loss_sum += nn.losses.cross_entropy(logits, labels, reduction="sum").item()
        correct += (logits.argmax(axis=-1) == labels).sum().item()
        total += n
    model.train()
    return correct / total, loss_sum / total


def main(args, data: HeteroGraphData = None):
    mx.random.seed(args.seed)
    target = "author"

    if data is None:
        data = DBLP(base_dir=args.data_dir)[0]
    data = add_constant_features(data)

    num_neighbors = [int(k) for k in args.num_neighbors.split(",")]
    eval_num_neighbors = [-1] * len(num_neighbors)

    def loader(mask, shuffle, fanout, batch_size):
        return NeighborLoader(
            data,
            num_neighbors=fanout,
            input_nodes=(target, mask),
            batch_size=batch_size,
            shuffle=shuffle,
        )

    train_loader = loader(data.author_train_mask, True, num_neighbors, args.batch_size)
    val_loader = loader(data.author_val_mask, False, eval_num_neighbors, 1024)
    test_loader = loader(data.author_test_mask, False, eval_num_neighbors, 1024)

    model = HeteroSAGE(
        node_features_dims=data.num_node_features,
        edge_types=list(data.edge_index_dict),
        hidden_dim=args.hidden_dim,
        out_dim=data.num_node_classes[target],
        target=target,
        num_layers=len(num_neighbors),
        dropout=args.dropout,
    )
    mx.eval(model.parameters())

    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=args.weight_decay)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    print(f"Node counts: {data.num_nodes}")
    print(f"Edge types: {list(data.edge_index_dict)}")
    print(
        f"Training on {len(train_loader.input_nodes)} authors, "
        f"{len(train_loader)} batches per epoch\n"
    )

    best_val_acc, best_state, times = 0.0, None, []
    for epoch in range(args.epochs):
        start = time()
        losses = []
        for batch in train_loader:
            loss, grads = loss_and_grad_fn(model, batch, target)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            losses.append(loss.item())
        times.append(time() - start)

        val_acc, val_loss = evaluate(model, val_loader, target)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = [(k, mx.array(v)) for k, v in tree_flatten(model.parameters())]

        print(
            " | ".join(
                [
                    f"Epoch: {epoch:3d}",
                    f"Train loss: {sum(losses) / len(losses):.3f}",
                    f"Val loss: {val_loss:.3f}",
                    f"Val acc: {val_acc:.3f}",
                    f"Time: {times[-1]:.2f}s",
                ]
            )
        )

    # Test with the parameters of the best validation epoch
    model.load_weights(best_state)
    test_acc, test_loss = evaluate(model, test_loader, target)
    print(f"\nBest val acc: {best_val_acc:.3f}")
    print(f"Test loss: {test_loss:.3f}  |  Test acc: {test_acc:.3f}")
    print(f"Mean epoch time: {sum(times) / len(times):.2f}s")
    return test_acc


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument(
        "--num_neighbors",
        type=str,
        default="10,10",
        help="Neighbors sampled per hop for training; also sets the number "
        "of layers. Evaluation uses all neighbors.",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    main(args)
