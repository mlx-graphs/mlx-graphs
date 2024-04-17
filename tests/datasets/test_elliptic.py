import mlx.core as mx
import pytest

from mlx_graphs.datasets import EllipticBitcoinDataset
from mlx_graphs.loaders import Dataloader


@pytest.mark.slow
def test_elliptic_bitcoin_dataset(tmp_path):
    from torch_geometric.datasets import (
        EllipticBitcoinDataset as EllipticBitcoinDataset_torch,
    )
    from torch_geometric.loader import DataLoader

    dataset = EllipticBitcoinDataset(base_dir=tmp_path)
    dataset_torch = EllipticBitcoinDataset_torch(tmp_path)

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
