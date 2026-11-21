import mlx.core as mx
import pytest

from mlx_graphs.datasets import IMDB


@pytest.mark.slow
def test_imdb_dataset(tmp_path):
    from torch_geometric.datasets import IMDB as IMDB_torch

    dataset = IMDB(base_dir=tmp_path)
    dataset_torch = IMDB_torch(tmp_path)
    graph = dataset.graphs[0]
    data = dataset_torch.data
    assert mx.array_equal(
        graph.edge_index_dict[("actor", "to", "movie")],
        mx.array(data["actor", "to", "movie"].edge_index.tolist()),
    ), "Actor movie edge index is not equal"
    assert mx.array_equal(
        graph.edge_index_dict[("director", "to", "movie")],
        mx.array(data["director", "to", "movie"].edge_index.tolist()),
    ), "Director movie edge index is not equal"
    assert mx.array_equal(
        graph.edge_index_dict[("movie", "to", "actor")],
        mx.array(data["movie", "to", "actor"].edge_index.tolist()),
    ), "movie actor edge index is not equal"
    assert mx.array_equal(
        graph.edge_index_dict[("movie", "to", "director")],
        mx.array(data["movie", "to", "director"].edge_index.tolist()),
    ), "movie director edge index is not equal"

    assert mx.array_equal(
        graph.node_features_dict["actor"],
        mx.array(data["actor"].x),
    ), "Actor features are not equal"

    assert mx.array_equal(
        graph.node_features_dict["movie"],
        mx.array(data["movie"].x),
    ), "movie features are not equal"

    assert mx.array_equal(
        graph.node_features_dict["director"],
        mx.array(data["director"].x),
    ), "director features are not equal"


def test_imdb_labels_are_keyed_by_the_movie_node_type(tmp_path):
    # Tiny raw files with the layout of the real dataset: 3 movies,
    # 2 directors, 2 actors.
    import numpy as np

    sp = pytest.importorskip("scipy.sparse")
    raw = tmp_path / "IMDB" / "raw"
    raw.mkdir(parents=True)
    for i, n in enumerate((3, 2, 2)):
        sp.save_npz(raw / f"features_{i}.npz", sp.csr_matrix(np.eye(n, 4)))
    np.save(raw / "labels.npy", np.array([2, 0, 1]))
    np.savez(
        raw / "train_val_test_idx.npz",
        train_idx=np.array([0]),
        val_idx=np.array([1]),
        test_idx=np.array([2]),
    )
    adj = np.zeros((7, 7))
    for movie, other in ((0, 3), (1, 4), (2, 5), (2, 6)):  # director / actor
        adj[movie, other] = adj[other, movie] = 1
    sp.save_npz(raw / "adjM.npz", sp.csr_matrix(adj))

    graph = IMDB(base_dir=str(tmp_path))[0]

    assert set(graph.node_labels_dict) == {"movie"}
    assert mx.array_equal(graph.node_labels_dict["movie"], mx.array([2, 0, 1]))
    homogeneous = graph.to_homogeneous()
    assert mx.array_equal(homogeneous.node_labels[:3], mx.array([2, 0, 1]))
