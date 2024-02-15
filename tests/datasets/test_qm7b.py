import os
import shutil

import mlx.core as mx
import pytest
from torch_geometric.datasets import QM7b as QM7b_torch
from torch_geometric.loader import DataLoader

from mlx_graphs.datasets import QM7bDataset
from mlx_graphs.loaders import Dataloader


@pytest.mark.slow
def test_tu_dataset():
    path = os.path.join("/".join(__file__.split("/")[:-1]), ".tests/")
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass

    dataset = QM7bDataset(base_dir=path)
    dataset_torch = QM7b_torch(path)

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

    shutil.rmtree(path)
