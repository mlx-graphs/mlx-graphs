import mlx.core as mx
import pytest

from mlx_graphs.data import batch
from mlx_graphs.datasets import TUDataset
from mlx_graphs.datasets.tu_dataset import read_tu_data
from mlx_graphs.loaders import Dataloader


@pytest.mark.slow
def test_tu_dataset(tmp_path):
    from torch_geometric.datasets import TUDataset as TUDataset_torch
    from torch_geometric.loader import DataLoader

    dataset_name = "ENZYMES"

    dataset = TUDataset(dataset_name, base_dir=tmp_path)
    dataset_torch = TUDataset_torch(tmp_path, dataset_name, use_node_attr=True)

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


def test_read_tu_data_without_node_features_keeps_isolated_nodes(tmp_path):
    # Two graphs, no node labels/attributes (like REDDIT-BINARY).
    # Graph 0 has nodes 1-4 (1-indexed); node 4 is isolated.
    # Graph 1 has nodes 5-6.
    prefix = "TOY"
    folder = tmp_path / prefix
    folder.mkdir()
    (folder / f"{prefix}_A.txt").write_text("1, 2\n2, 1\n2, 3\n3, 2\n5, 6\n6, 5\n")
    (folder / f"{prefix}_graph_indicator.txt").write_text("1\n1\n1\n1\n2\n2\n")
    (folder / f"{prefix}_graph_labels.txt").write_text("1\n-1\n")

    graphs = read_tu_data(str(tmp_path), prefix)

    assert graphs[0].node_features is None
    assert graphs[0].num_nodes == 4
    assert graphs[1].num_nodes == 2

    batched = batch(graphs)
    expected_edge_index = mx.array([[0, 1, 1, 2, 4, 5], [1, 0, 2, 1, 5, 4]])
    assert mx.array_equal(batched.edge_index, expected_edge_index)
    assert mx.array_equal(batched.batch_indices, mx.array([0, 0, 0, 0, 1, 1]))
