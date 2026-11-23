"""PointNet++ classification on ModelNet10, as PyG's
``examples/pointnet2_classification.py``.

    python examples/pointnet2_modelnet.py --epochs 20

Each mesh is normalized once (``NormalizeScale``) and 1024 points are sampled
from its faces every time it is loaded (``SamplePoints``). Each set abstraction
level samples centers with farthest point sampling (``fps``), groups their
neighbors within a radius (``radius``) and applies a ``PointNetConv``.

References:
    [1] `Qi et al. PointNet++: Deep Hierarchical Feature Learning on Point Sets
    in a Metric Space. <https://arxiv.org/abs/1706.02413>`_
"""

from argparse import ArgumentParser
from time import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from mlx_graphs.datasets import ModelNet
from mlx_graphs.loaders import Dataloader
from mlx_graphs.nn import MLP, PointNetConv, global_max_pool
from mlx_graphs.transforms import NormalizeScale, SamplePoints
from mlx_graphs.utils import fps, radius


class SAModule(nn.Module):
    """Set abstraction: farthest point sampling, radius grouping, PointNet."""

    def __init__(self, ratio: float, r: float, mlp: nn.Module):
        super().__init__()
        self.ratio, self.r = ratio, r
        self.conv = PointNetConv(mlp, add_self_loops=False)

    def __call__(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(
            pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64
        )
        x_dst = None if x is None else x[idx]
        x = self.conv(mx.stack([col, row]), (x, x_dst), (pos, pos[idx]))
        return x, pos[idx], batch[idx]


class PointNet2(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.sa1 = SAModule(0.5, 0.2, MLP([3, 64, 64, 128], norm="batch"))
        self.sa2 = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256], norm="batch"))
        self.sa3 = MLP([256 + 3, 256, 512, 1024], norm="batch")
        self.mlp = MLP([1024, 512, 256, num_classes], dropout=0.5)

    def __call__(self, pos, batch, num_graphs):
        x, pos, batch = self.sa1(None, pos, batch)
        x, pos, batch = self.sa2(x, pos, batch)
        x = self.sa3(mx.concatenate([x, pos], axis=1))
        return self.mlp(global_max_pool(x, batch, num_graphs))


def main(args):
    mx.random.seed(args.seed)
    transform, pre_transform = SamplePoints(1024), NormalizeScale()
    train = ModelNet("10", True, pre_transform=pre_transform, transform=transform)
    test = ModelNet("10", False, pre_transform=pre_transform, transform=transform)
    model = PointNet2(train.num_classes)
    optimizer = optim.Adam(learning_rate=args.lr)

    def loss_fn(model, batch):
        logits = model(batch.pos, batch.batch_indices, batch.num_graphs)
        return nn.losses.cross_entropy(logits, batch.graph_labels, reduction="mean")

    step = nn.value_and_grad(model, loss_fn)
    for epoch in range(1, args.epochs + 1):
        model.train()
        start, total = time(), 0.0
        for batch in Dataloader(train, batch_size=args.batch_size, shuffle=True):
            loss, grads = step(model, batch)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            total += loss.item() * batch.num_graphs

        model.eval()
        correct = 0
        for batch in Dataloader(test, batch_size=args.batch_size):
            logits = model(batch.pos, batch.batch_indices, batch.num_graphs)
            correct += (mx.argmax(logits, axis=1) == batch.graph_labels).sum().item()
        print(
            f"Epoch {epoch:03d}, loss {total / len(train):.4f}, "
            f"test accuracy {correct / len(test):.4f} ({time() - start:.1f} s)"
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
