import mlx.core as mx
import pytest

from mlx_graphs.datasets import DBLP


@pytest.mark.slow
def test_dblp_dataset(tmp_path):
    from torch_geometric.datasets import DBLP as DBLP_torch

    dataset = DBLP(base_dir=tmp_path)
    dataset_torch = DBLP_torch(tmp_path)
    graph = dataset.graphs[0]
    data = dataset_torch.data
    assert mx.array_equal(
        graph.edge_index_dict[("author", "to", "paper")],
        mx.array(data["author", "to", "paper"].edge_index.tolist()),
    ), "Author paper edge index is not equal"
    assert mx.array_equal(
        graph.edge_index_dict[("paper", "to", "author")],
        mx.array(data["paper", "to", "author"].edge_index.tolist()),
    ), "Paper Author edge index is not equal"

    assert mx.array_equal(
        graph.edge_index_dict[("paper", "to", "term")],
        mx.array(data["paper", "to", "term"].edge_index.tolist()),
    ), "paper term edge index is not equal"

    assert mx.array_equal(
        graph.edge_index_dict[("paper", "to", "conference")],
        mx.array(data["paper", "to", "conference"].edge_index.tolist()),
    ), "paper conference edge index is not equal"

    assert mx.array_equal(
        graph.edge_index_dict[("term", "to", "paper")],
        mx.array(data["term", "to", "paper"].edge_index.tolist()),
    ), "term paper edge index is not equal"

    assert mx.array_equal(
        graph.edge_index_dict[("conference", "to", "paper")],
        mx.array(data["conference", "to", "paper"].edge_index.tolist()),
    ), "conference paper edge index is not equal"

    assert mx.array_equal(
        graph.node_features_dict["paper"],
        mx.array(data["paper"].x),
    ), "paper features are not equal"

    assert mx.array_equal(
        graph.node_features_dict["author"],
        mx.array(data["author"].x),
    ), "author features are not equal"

    assert mx.array_equal(
        graph.node_features_dict["term"],
        mx.array(data["term"].x),
    ), "term features are not equal"
