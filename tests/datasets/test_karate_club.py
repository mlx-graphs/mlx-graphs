import pytest

from mlx_graphs.datasets import KarateClubDataset


def test_karate_club_dataset():
    d = KarateClubDataset()
    assert len(d) == 1, "Wrong dataset lenght"
    assert d.num_node_classes == 2, "Wrong number of classes"

    g = d[0]
    assert g.edge_index.shape == (2, 156), "Wrong edge_index shape"
    assert g.node_features.shape == (34, 1), "Wrong node_features shape"
    assert g.node_labels.shape == (34, 1), "Wrong node_labels shape"

    # only one graph in the dataset
    with pytest.raises(IndexError):
        _ = d[1]
