import mlx.core as mx
import numpy as np
import pytest

from mlx_graphs.datasets import HeterophilousGraphDataset


@pytest.mark.slow
@pytest.mark.parametrize(
    "name",
    ["roman_empire", "amazon-ratings", "minesweeper", "tolokers", "questions"],
)
def test_heterophilous_matches_pyg(tmp_path, name):
    from torch_geometric.datasets import (
        HeterophilousGraphDataset as HeterophilousGraphDataset_torch,
    )

    dataset = HeterophilousGraphDataset(name, base_dir=str(tmp_path / "mxg"))
    ref = HeterophilousGraphDataset_torch(str(tmp_path / "pyg"), name)[0]
    graph = dataset[0]

    assert graph.num_nodes == ref.num_nodes
    assert np.array_equal(np.array(graph.edge_index), ref.edge_index.numpy())
    assert np.array_equal(np.array(graph.node_features), ref.x.numpy())
    assert np.array_equal(np.array(graph.node_labels), ref.y.numpy())
    for split in ["train", "val", "test"]:
        mask = getattr(graph, f"{split}_mask")
        assert mask.shape == (graph.num_nodes, 10)
        assert mask.dtype == mx.bool_
        assert np.array_equal(np.array(mask), getattr(ref, f"{split}_mask").numpy())
    assert graph.is_undirected()


def test_heterophilous_invalid_name(tmp_path):
    with pytest.raises(AssertionError):
        HeterophilousGraphDataset("chameleon", base_dir=str(tmp_path))
