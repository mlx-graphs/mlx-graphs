import mlx.core as mx
from torch_geometric.datasets import TUDataset as TUDataset_torch
from torch_geometric.loader import DataLoader

from mlx_graphs.data import Dataloader
from mlx_graphs.datasets import TUDataset
from mlx_graphs.datasets.dataset import DEFAULT_BASE_DIR


def test_tu_dataset():
    dataset_name = "IMDB-BINARY"
    dataset = TUDataset(dataset_name)
    dataset_torch = TUDataset_torch(
        DEFAULT_BASE_DIR + "tests/" + dataset_name, dataset_name
    )

    train_loader = Dataloader(dataset, 3, shuffle=False)
    train_loader_torch = DataLoader(dataset_torch, 3, shuffle=False)

    for batch_mxg, batch_pyg in zip(train_loader, train_loader_torch):
        assert mx.array_equal(
            mx.array(batch_pyg.edge_index.tolist()), batch_mxg.edge_index
        ), "Two arrays between PyG and mxg are different"
        # assert mx.array_equal(
        #     mx.array(batch_pyg.x.tolist()), batch_mxg.node_features
        # ), "Two arrays between PyG and mxg are different"
        assert mx.array_equal(
            mx.array(batch_pyg.y.tolist()), batch_mxg.graph_labels
        ), "Two arrays between PyG and mxg are different"
        # assert mx.array_equal(
        #     mx.array(batch_pyg.edge_attr.tolist()), batch_mxg.edge_features + 1
        # ), "Two arrays between PyG and mxg are different"


test_tu_dataset()
