import mlx.core as mx
import numpy as np

from mlx_graphs.data import GraphData
from mlx_graphs.loaders import Dataloader


def test_dataloader():
    data = [GraphData(edge_index=mx.array([[0], [0]]))] * 4

    # test batch 1
    dl = Dataloader(data, batch_size=1)
    i = 0
    for d in dl:
        i += 1
    assert i == 4

    # test batch 2
    dl = Dataloader(data, batch_size=2)
    i = 0
    for d in dl:
        i += 1
    assert i == 2

    # test batch 3 with 1 remaining data
    dl = Dataloader(data, batch_size=3)
    i = 0
    for d in dl:
        i += 1
    assert i == 2
    assert d.num_graphs == 1


def test_dataloader_shuffles_first_epoch_and_restarts():
    graphs = [
        GraphData(edge_index=mx.array([[0], [0]]), node_features=mx.full((1, 1), i))
        for i in range(50)
    ]

    np.random.seed(0)
    loader = Dataloader(graphs, batch_size=10, shuffle=True)
    assert len(loader) == 5
    first = [int(v) for b in loader for v in b.node_features.reshape(-1).tolist()]
    assert sorted(first) == list(range(50))
    assert first != list(range(50))  # the first epoch is shuffled too

    # breaking out of an epoch does not leak into the next one
    for _ in loader:
        break
    assert sum(1 for _ in loader) == 5

    ordered = Dataloader(graphs, batch_size=10)
    values = [int(v) for b in ordered for v in b.node_features.reshape(-1).tolist()]
    assert values == list(range(50))
