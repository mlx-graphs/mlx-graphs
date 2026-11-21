import numpy as np
import pytest

from mlx_graphs.datasets import WikiCS


@pytest.mark.slow
@pytest.mark.parametrize("is_undirected", [True, False])
def test_wikics_matches_pyg(tmp_path, is_undirected):
    from torch_geometric.datasets import WikiCS as WikiCS_torch

    graph = WikiCS(is_undirected=is_undirected, base_dir=str(tmp_path / "mxg"))[0]
    ref = WikiCS_torch(str(tmp_path / "pyg"), is_undirected=is_undirected)[0]

    assert graph.num_nodes == ref.num_nodes
    assert np.array_equal(np.array(graph.edge_index), ref.edge_index.numpy())
    assert np.array_equal(np.array(graph.node_features), ref.x.numpy())
    assert np.array_equal(np.array(graph.node_labels), ref.y.numpy())
    for mask in ["train_mask", "val_mask", "test_mask", "stopping_mask"]:
        assert np.array_equal(
            np.array(getattr(graph, mask)), getattr(ref, mask).numpy()
        ), mask
    assert graph.train_mask.shape == (graph.num_nodes, 20)
    assert graph.is_undirected() == is_undirected
