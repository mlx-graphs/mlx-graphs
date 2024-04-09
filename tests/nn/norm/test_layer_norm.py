import mlx.core as mx
import pytest
import torch

from mlx_graphs.data import GraphData, GraphDataBatch
from mlx_graphs.nn import LayerNormalization


@pytest.mark.parametrize("mode", ["node", "graph"])
def test_layer_norm(mode):
    from torch_geometric.data import Batch, Data
    from torch_geometric.nn import LayerNorm as torch_LayerNormalization

    # Graph 1
    x = torch.tensor(
        [[0.8, 0.1], [0.1, 0.9], [0.5, 0.5], [0.7, 0.3]], dtype=torch.float
    )
    edge_index = torch.tensor([[0, 1, 2, 2, 3], [1, 0, 2, 3, 0]], dtype=torch.long)

    # Graph 2
    x2 = torch.tensor(
        [[0.9, 0.4], [0.7, 1.0], [0.8, 0.9], [0.7, 0.3]], dtype=torch.float
    )
    edge_index2 = torch.tensor([[0, 1, 2, 2, 3], [1, 0, 2, 3, 0]], dtype=torch.long)

    graph_batch = Batch.from_data_list([Data(x, edge_index), Data(x2, edge_index2)])

    torch_layer_norm = torch_LayerNormalization(2, affine=False, mode=mode)

    graphs = [
        GraphData(
            edge_index=mx.array([[0, 1, 2, 2, 3], [1, 0, 2, 3, 0]]),
            node_features=mx.array([[0.8, 0.1], [0.1, 0.9], [0.5, 0.5], [0.7, 0.3]]),
        ),
        GraphData(
            edge_index=mx.array([[0, 1, 2, 2, 3], [1, 0, 2, 3, 0]]),
            node_features=mx.array([[0.9, 0.4], [0.7, 1.0], [0.8, 0.9], [0.7, 0.3]]),
        ),
    ]
    batch = GraphDataBatch(graphs)
    layer_norm = LayerNormalization(2, affine=False, mode=mode)
    assert mx.allclose(
        mx.array(torch_layer_norm(graph_batch.x, graph_batch.batch).numpy()),
        layer_norm(batch.node_features, batch.batch_indices),
    )
