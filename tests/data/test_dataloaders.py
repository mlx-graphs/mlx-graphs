import mlx.core as mx

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
