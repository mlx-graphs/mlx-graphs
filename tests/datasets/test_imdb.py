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
