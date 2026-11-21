import mlx.core as mx
import numpy as np
import pytest

from mlx_graphs.datasets import FB15k_237, WordNet18, WordNet18RR


@pytest.mark.slow
@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_fb15k_237_matches_pyg(tmp_path, split):
    from torch_geometric.datasets import FB15k_237 as FB15k_237_torch

    dataset = FB15k_237(split, base_dir=str(tmp_path / "mxg"))
    ref = FB15k_237_torch(str(tmp_path / "pyg"), split=split)[0]
    assert len(dataset) == 1
    graph = dataset[0]

    assert graph.num_nodes == ref.num_nodes == 14541
    assert np.array_equal(np.array(graph.edge_index), ref.edge_index.numpy())
    assert np.array_equal(np.array(graph.edge_type), ref.edge_type.numpy())
    assert graph.edge_index.dtype == mx.int32


@pytest.mark.slow
@pytest.mark.parametrize(
    "cls, pyg_name", [(WordNet18, "WordNet18"), (WordNet18RR, "WordNet18RR")]
)
def test_wordnet_matches_pyg(tmp_path, cls, pyg_name):
    import torch_geometric.datasets as pyg_datasets

    graph = cls(base_dir=str(tmp_path / "mxg"))[0]
    ref = getattr(pyg_datasets, pyg_name)(str(tmp_path / "pyg"))[0]

    assert graph.num_nodes == ref.num_nodes
    assert np.array_equal(np.array(graph.edge_index), ref.edge_index.numpy())
    # PyG sorts the triplets by (head, tail) with an unstable sort, so
    # triplets that only differ by their relation may be ordered differently
    keys = graph.num_nodes * np.array(graph.edge_index[0]).astype(np.int64)
    keys = keys + np.array(graph.edge_index[1])
    attrs = ["edge_type", "train_mask", "val_mask", "test_mask"]
    ours = np.stack([np.array(getattr(graph, a)).astype(np.int64) for a in attrs])
    theirs = np.stack([getattr(ref, a).numpy().astype(np.int64) for a in attrs])
    order_ours = np.lexsort([*ours[::-1], keys])
    order_theirs = np.lexsort([*theirs[::-1], keys])
    assert np.array_equal(ours[:, order_ours], theirs[:, order_theirs])
    assert int(
        graph.train_mask.sum() + graph.val_mask.sum() + graph.test_mask.sum()
    ) == (graph.num_edges)
