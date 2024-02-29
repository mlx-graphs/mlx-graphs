import os
import shutil

import mlx.core as mx
import pytest

from mlx_graphs.datasets import TUDataset
from mlx_graphs.loaders import Dataloader


@pytest.mark.slow
def test_tu_dataset():
    from torch_geometric.datasets import TUDataset as TUDataset_torch
    from torch_geometric.loader import DataLoader

    dataset_name = "ENZYMES"
    path = os.path.join("/".join(__file__.split("/")[:-1]), ".tests/")
    shutil.rmtree(path, ignore_errors=True)

    dataset = TUDataset(dataset_name, base_dir=path)
    dataset_torch = TUDataset_torch(path, dataset_name)

    train_loader = Dataloader(dataset, 10, shuffle=False)
    train_loader_torch = DataLoader(dataset_torch, 10, shuffle=False)

    for batch_mxg, batch_pyg in zip(train_loader, train_loader_torch):
        assert mx.array_equal(
            mx.array(batch_pyg.edge_index.tolist()), batch_mxg.edge_index
        ), "Two arrays between PyG and mxg are different"

        if batch_mxg.node_features is not None:
            assert mx.array_equal(
                mx.array(batch_pyg.x.tolist()), batch_mxg.node_features
            ), "Two arrays between PyG and mxg are different"

        if batch_mxg.graph_labels is not None:
            assert mx.array_equal(
                mx.array(batch_pyg.y.tolist()), batch_mxg.graph_labels
            ), "Two arrays between PyG and mxg are different"

        if batch_mxg.graph_labels is not None:
            assert mx.array_equal(
                mx.array(batch_pyg.y.tolist()), batch_mxg.graph_labels
            ), "Two arrays between PyG and mxg are different"

    shutil.rmtree(path)
