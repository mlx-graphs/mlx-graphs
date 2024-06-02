import mlx.core as mx
import numpy as np

from mlx_graphs.datasets import LANLDataset
from mlx_graphs.loaders import LANLDataLoader


@pytest.mark.slow
def test_lanl_dataset(tmp_path):
    dataset = LANLDataset(base_dir=tmp_path)

    # Attributes
    assert dataset[0].edge_index.shape == (2, 158)
    assert dataset[0].edge_features.shape == (158, 4)
    assert dataset[0].edge_labels.shape == (158,)
    assert dataset[0].edge_timestamps.shape == (158,)
    assert dataset[0].node_features.shape == (17685, 17685)

    # Indexing
    assert dataset[range(10)].edge_index.shape == (2, 1975)
    assert dataset[[0, 1, 2]].edge_index.shape == (2, 552)
    assert dataset[np.array([0, 1, 2])].edge_index.shape == (2, 552)
    assert dataset[mx.array([0, 1, 2])].edge_index.shape == (2, 552)
    assert dataset[:3].edge_index.shape == (2, 552)

    # With loader + graph compression
    for split in ["train", "all"]:
        loader = LANLDataLoader(
            dataset,
            split=split,
            remove_self_loops=False,
            force_processing=True,
            compress_edges=True,
        )
        graph = next(loader)
        assert graph.edge_index.shape == (2, 2708)
        assert graph.edge_features.shape == (2708, 13)
        assert graph.edge_labels.shape == (2708,)
        assert graph.node_features.shape == (17685, 17685)

    # Without force_processing (should store the results in a new folder)
    loader = LANLDataLoader(
        dataset,
        split="valid",
        remove_self_loops=False,
        compress_edges=True,
        batch_size=60,
    )
    graph = next(loader)
    assert graph.edge_index.shape == (2, 9917)
    assert graph.edge_features.shape == (9917, 13)
    assert graph.edge_labels.shape == (9917,)
    assert graph.node_features.shape == (17685, 17685)

    loader = LANLDataLoader(
        dataset,
        split="test",
        remove_self_loops=False,
        force_processing=True,
        compress_edges=True,
    )
    graph = next(loader)
    assert graph.edge_index.shape == (2, 7662)
    assert graph.edge_features.shape == (7662, 13)
    assert graph.edge_labels.shape == (7662,)
    assert graph.node_features.shape == (17685, 17685)

    # With loader - graph compression
    loader = LANLDataLoader(
        dataset,
        split="train",
        remove_self_loops=False,
        force_processing=True,
        compress_edges=False,
    )
    graph = next(loader)
    assert graph.edge_index.shape == (2, 11624)
    assert graph.edge_features.shape == (11624, 4)
    assert graph.edge_labels.shape == (11624,)
    assert graph.edge_timestamps.shape == (11624,)
    assert graph.node_features.shape == (17685, 17685)

    # With loader + remove self loops
    loader = LANLDataLoader(
        dataset,
        split="train",
        remove_self_loops=True,
        force_processing=True,
        compress_edges=False,
    )
    graph = next(loader)
    assert graph.edge_index.shape == (2, 11624)
    assert graph.edge_features.shape == (11624, 4)
    assert graph.edge_labels.shape == (11624,)
    assert graph.edge_timestamps.shape == (11624,)
    assert graph.node_features.shape == (17685, 17685)

    # With loader + nb_processes
    loader = LANLDataLoader(
        dataset,
        split="train",
        remove_self_loops=False,
        force_processing=True,
        compress_edges=False,
        nb_processes=4,
    )
    graph = next(loader)
    assert graph.edge_index.shape == (2, 11624)
    assert graph.edge_features.shape == (11624, 4)
    assert graph.edge_labels.shape == (11624,)
    assert graph.edge_timestamps.shape == (11624,)
    assert graph.node_features.shape == (17685, 17685)

    # With loader + batch_size
    loader = LANLDataLoader(
        dataset,
        split="train",
        remove_self_loops=False,
        force_processing=True,
        compress_edges=False,
        batch_size=120,
    )
    graph = next(loader)
    truth = dataset[:120]
    assert graph.edge_index.shape == truth.edge_index.shape
    assert graph.edge_features.shape == truth.edge_features.shape
    assert graph.edge_labels.shape == truth.edge_labels.shape
    assert graph.edge_timestamps.shape == truth.edge_timestamps.shape
    assert graph.node_features.shape == truth.node_features.shape

    # With loader + without force_processing
    loader = LANLDataLoader(
        dataset,
        split="train",
        remove_self_loops=False,
        force_processing=False,
        compress_edges=False,
        batch_size=120,
    )
    graph = next(loader)
    truth = dataset[:120]
    assert graph.edge_index.shape == truth.edge_index.shape
    assert graph.edge_features.shape == truth.edge_features.shape
    assert graph.edge_labels.shape == truth.edge_labels.shape
    assert graph.edge_timestamps.shape == truth.edge_timestamps.shape
    assert graph.node_features.shape == truth.node_features.shape
