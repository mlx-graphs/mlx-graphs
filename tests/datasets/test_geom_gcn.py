import mlx.core as mx
import numpy as np
import pytest

from mlx_graphs.datasets import Actor, WebKB, WikipediaNetwork


def _assert_same_graph(graph, ref):
    assert graph.num_nodes == ref.num_nodes
    assert np.array_equal(np.array(graph.edge_index), ref.edge_index.numpy())
    assert np.array_equal(np.array(graph.node_features), ref.x.numpy())
    assert np.array_equal(np.array(graph.node_labels), ref.y.numpy())
    for split in ["train", "val", "test"]:
        mask = getattr(graph, f"{split}_mask")
        assert mask.shape == (graph.num_nodes, 10)
        assert mask.dtype == mx.bool_
        assert np.array_equal(np.array(mask), getattr(ref, f"{split}_mask").numpy())


@pytest.mark.slow
@pytest.mark.parametrize("name", ["cornell", "texas", "wisconsin"])
def test_webkb_matches_pyg(tmp_path, name):
    from torch_geometric.datasets import WebKB as WebKB_torch

    graph = WebKB(name, base_dir=str(tmp_path / "mxg"))[0]
    _assert_same_graph(graph, WebKB_torch(str(tmp_path / "pyg"), name)[0])


@pytest.mark.slow
@pytest.mark.parametrize("name", ["chameleon", "squirrel"])
def test_wikipedia_network_matches_pyg(tmp_path, name):
    from torch_geometric.datasets import WikipediaNetwork as WikipediaNetwork_torch

    graph = WikipediaNetwork(name, base_dir=str(tmp_path / "mxg"))[0]
    ref = WikipediaNetwork_torch(str(tmp_path / "pyg"), name, geom_gcn_preprocess=True)
    _assert_same_graph(graph, ref[0])


@pytest.mark.slow
def test_actor_matches_pyg(tmp_path):
    from torch_geometric.datasets import Actor as Actor_torch

    graph = Actor(base_dir=str(tmp_path / "mxg"))[0]
    _assert_same_graph(graph, Actor_torch(str(tmp_path / "pyg"))[0])


def test_geom_gcn_invalid_names(tmp_path):
    with pytest.raises(AssertionError):
        WebKB("cora", base_dir=str(tmp_path))
    with pytest.raises(AssertionError):
        WikipediaNetwork("crocodile", base_dir=str(tmp_path))
