import mlx.core as mx
import pytest

from mlx_graphs.data import GraphData
from mlx_graphs.datasets import EllipticBitcoinDataset
from mlx_graphs.loaders import Dataloader
from mlx_graphs.transforms import FeaturesNormalizedTransform


def test_normalize_features():
    transform = FeaturesNormalizedTransform()
    assert str(transform) == "FeaturesNormalizedTransform()"

    node_features = mx.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    data = GraphData(
        edge_index=mx.array([[0, 1, 2], [1, 2, 0]]), node_features=node_features
    )

    norm_data = transform(data)
    assert norm_data.node_features.tolist() == [[0.5, 0, 0.5], [0, 1, 0], [0, 0, 0]]


@pytest.mark.slow
def test_transform(tmp_path):
    from torch_geometric.datasets import (
        EllipticBitcoinDataset as EllipticBitcoinDataset_torch,
    )
    from torch_geometric.loader import DataLoader as DataLoader_torch
    from torch_geometric.transforms import NormalizeFeatures as NormalizeFeatures_torch

    dataset = EllipticBitcoinDataset(
        base_dir=tmp_path, pre_transform=FeaturesNormalizedTransform()
    )
    dataset_torch = EllipticBitcoinDataset_torch(
        tmp_path, pre_transform=NormalizeFeatures_torch()
    )

    train_loader = Dataloader(dataset, 10, shuffle=False)
    train_loader_torch = DataLoader_torch(dataset_torch, 10, shuffle=False)

    for batch_mxg, batch_pyg in zip(train_loader, train_loader_torch):
        assert mx.array_equal(
            mx.array(batch_pyg.edge_index.tolist()), batch_mxg.edge_index
        ), "Two arrays (edge_indexes) between PyG and mxg are different"

        if batch_mxg.node_features is not None:
            assert mx.allclose(
                mx.array(batch_pyg.x.tolist()), batch_mxg.node_features
            ), "Two arrays(node_features) between PyG and mxg are different"

        if batch_mxg.node_labels is not None:
            assert mx.array_equal(
                mx.array(batch_pyg.y.tolist()), batch_mxg.node_labels
            ), "Two arrays(labels) between PyG and mxg are different"
