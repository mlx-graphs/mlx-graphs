import mlx.core as mx
import pytest

from mlx_graphs.datasets import SuperPixelDataset
from mlx_graphs.loaders import Dataloader


@pytest.mark.slow
def test_superpixel_dataset(tmp_path):
    from torch_geometric.datasets import GNNBenchmarkDataset
    from torch_geometric.loader import DataLoader

    dataset = SuperPixelDataset("MNIST", "test", base_dir=tmp_path)  # , base_dir=path)
    dataset_torch = GNNBenchmarkDataset(root=tmp_path, name="MNIST", split="test")

    train_loader = Dataloader(dataset, 10, shuffle=False)
    train_loader_torch = DataLoader(dataset_torch, 10, shuffle=False)

    for batch_mxg, batch_pyg in zip(train_loader, train_loader_torch):
        assert (
            mx.array(batch_pyg.edge_index.tolist()).shape == batch_mxg.edge_index.shape
        ), "Edge indices have different shapes"

        assert mx.array_equal(
            mx.array(batch_pyg.y.tolist()), batch_mxg.graph_labels
        ), "Graph labels are different"
