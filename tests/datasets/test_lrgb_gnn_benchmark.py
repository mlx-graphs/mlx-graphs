import os

import mlx.core as mx
import numpy as np
import pytest

from mlx_graphs.datasets import GNNBenchmarkDataset, LRGBDataset
from mlx_graphs.loaders import Dataloader


def assert_equal(mlx_array, torch_tensor, dtype):
    assert mlx_array is not None
    assert mlx_array.dtype == dtype
    np.testing.assert_array_equal(np.array(mlx_array), torch_tensor.numpy())


def test_invalid_names():
    with pytest.raises(ValueError, match="not available"):
        LRGBDataset("PascalVOC-SP")
    with pytest.raises(ValueError, match="Unknown"):
        LRGBDataset("peptides")
    with pytest.raises(ValueError, match="SuperPixelDataset"):
        GNNBenchmarkDataset("MNIST")
    with pytest.raises(ValueError, match="not available"):
        GNNBenchmarkDataset("TSP")
    with pytest.raises(ValueError, match="Split"):
        GNNBenchmarkDataset("PATTERN", split="validation")


@pytest.mark.slow
@pytest.mark.parametrize("name", ["peptides-func", "peptides-struct"])
def test_lrgb_matches_pyg(tmp_path, name):
    from torch_geometric.datasets import LRGBDataset as LRGBDataset_torch

    sizes = {"train": 10873, "val": 2331, "test": 2331}
    num_tasks = 10 if name == "peptides-func" else 11
    for split in ["train", "val", "test"]:
        dataset = LRGBDataset(name, split, base_dir=os.path.join(tmp_path, "mxg"))
        dataset_torch = LRGBDataset_torch(os.path.join(tmp_path, "pyg"), name, split)
        assert len(dataset) == len(dataset_torch) == sizes[split]

        for graph, graph_torch in zip(dataset, dataset_torch):
            assert_equal(graph.edge_index, graph_torch.edge_index, mx.int32)
            assert_equal(graph.node_features, graph_torch.x, mx.int32)
            assert_equal(graph.edge_features, graph_torch.edge_attr, mx.int32)
            assert_equal(graph.graph_labels, graph_torch.y, mx.float32)
            assert graph.graph_labels.shape == (1, num_tasks)

        # batched labels have the PyG shape
        batch = next(iter(Dataloader(dataset, 4)))
        assert batch.graph_labels.shape == (4, num_tasks)

    # the processed data is reloaded
    reloaded = LRGBDataset(name, "val", base_dir=os.path.join(tmp_path, "mxg"))
    assert len(reloaded) == sizes["val"]


@pytest.mark.slow
@pytest.mark.parametrize("name", ["CSL", "CLUSTER", "PATTERN"])
def test_gnn_benchmark_matches_pyg(tmp_path, name):
    from torch_geometric.datasets import GNNBenchmarkDataset as GNNBenchmark_torch

    splits = ["train"] if name == "CSL" else ["train", "val", "test"]
    for split in splits:
        dataset = GNNBenchmarkDataset(
            name, split, base_dir=os.path.join(tmp_path, "mxg")
        )
        dataset_torch = GNNBenchmark_torch(os.path.join(tmp_path, "pyg"), name, split)
        assert len(dataset) == len(dataset_torch)

        for graph, graph_torch in zip(dataset, dataset_torch):
            assert_equal(graph.edge_index, graph_torch.edge_index, mx.int32)
            assert graph.num_nodes == graph_torch.num_nodes
            if name == "CSL":
                assert graph.node_features is None
                assert graph.graph_labels.tolist() == [graph_torch.y]
            else:
                assert_equal(graph.node_features, graph_torch.x, mx.float32)
                assert_equal(graph.node_labels, graph_torch.y, mx.int32)
        del dataset_torch

    if name == "CSL":
        assert len(dataset) == 150
        assert dataset.num_graph_classes == 10
        with pytest.warns(UserWarning, match="CSL"):
            GNNBenchmarkDataset("CSL", "test", base_dir=os.path.join(tmp_path, "mxg"))
