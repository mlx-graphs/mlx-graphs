import mlx.core as mx
import numpy as np
import pytest

from mlx_graphs.datasets import AmazonDataset, CoauthorDataset


@pytest.mark.slow
@pytest.mark.parametrize(
    "cls, name, pyg_cls, pyg_name",
    [
        (AmazonDataset, "photo", "Amazon", "Photo"),
        (AmazonDataset, "computers", "Amazon", "Computers"),
        (CoauthorDataset, "cs", "Coauthor", "CS"),
    ],
)
def test_npz_dataset_matches_pyg(tmp_path, cls, name, pyg_cls, pyg_name):
    import torch_geometric.datasets as pyg_datasets

    graph = cls(name, base_dir=str(tmp_path / "mxg"))[0]
    ref = getattr(pyg_datasets, pyg_cls)(str(tmp_path / "pyg"), pyg_name)[0]

    assert graph.num_nodes == ref.num_nodes
    # both are sorted by (source, destination) without duplicates
    assert np.array_equal(np.array(graph.edge_index), ref.edge_index.numpy())
    assert np.array_equal(np.array(graph.node_features), ref.x.numpy())
    assert np.array_equal(np.array(graph.node_labels), ref.y.numpy())
    assert graph.is_undirected()
    assert not graph.has_self_loops()
    assert graph.edge_index.dtype == mx.int32


def test_npz_dataset_invalid_name(tmp_path):
    with pytest.raises((AssertionError, ValueError)):
        AmazonDataset("books", base_dir=str(tmp_path))
