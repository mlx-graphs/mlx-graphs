import os
import shutil

import mlx.core as mx
import pytest

from mlx_graphs.datasets import PlanetoidDataset
from mlx_graphs.loaders import Dataloader


@pytest.mark.slow
@pytest.mark.parametrize(
    "dataset_name, split",
    [
        ("cora", "public"),
        ("cora", "full"),
        ("cora", "geom-gcn"),
        ("citeseer", "public"),
        ("pubmed", "public"),
    ],
)
def test_planetoid_cora_dataset(dataset_name, split):
    from torch_geometric.datasets import Planetoid as Planetoid_torch
    from torch_geometric.loader import DataLoader

    path = os.path.join("/".join(__file__.split("/")[:-1]), ".tests/")
    shutil.rmtree(path, ignore_errors=True)

    dataset = PlanetoidDataset(dataset_name, base_dir=path, split=split)
    dataset_torch = Planetoid_torch(path, dataset_name, split=split)

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

        if batch_mxg.node_labels is not None:
            assert mx.array_equal(
                mx.array(batch_pyg.y.tolist()), batch_mxg.node_labels
            ), "Two arrays between PyG and mxg are different"

        if batch_mxg.train_mask is not None:
            assert mx.array_equal(
                mx.array(batch_pyg.train_mask.tolist()), batch_mxg.train_mask
            ), "Two arrays between PyG and mxg are different"

        if batch_mxg.test_mask is not None:
            assert mx.array_equal(
                mx.array(batch_pyg.test_mask.tolist()), batch_mxg.test_mask
            ), "Two arrays between PyG and mxg are different"

        if batch_mxg.val_mask is not None:
            assert mx.array_equal(
                mx.array(batch_pyg.val_mask.tolist()), batch_mxg.val_mask
            ), "Two arrays between PyG and mxg are different"

    shutil.rmtree(path)
