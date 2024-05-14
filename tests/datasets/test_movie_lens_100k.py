import mlx.core as mx
import pytest

from mlx_graphs.datasets import MovieLens100K


@pytest.mark.slow
def test_movie_lens_100k(tmp_path):
    from torch_geometric.datasets import (
        MovieLens100K as MovieLens100K_torch,
    )

    dataset = MovieLens100K(base_dir=tmp_path)
    dataset_torch = MovieLens100K_torch(tmp_path)
    graph = dataset.graphs[0]
    data = dataset_torch.data
    assert mx.array_equal(
        graph.edge_index_dict[("user", "rates", "movie")],
        mx.array(data["user", "rates", "movie"].edge_index.tolist()),
    ), "Edge index is not equal"
