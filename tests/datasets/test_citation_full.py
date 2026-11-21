import mlx.core as mx
import numpy as np
import pytest

from mlx_graphs.datasets import CitationFull, CoraFull


def _assert_same_graph(graph, ref):
    assert graph.num_nodes == ref.num_nodes
    assert np.array_equal(np.array(graph.edge_index), ref.edge_index.numpy())
    assert np.array_equal(np.array(graph.node_features), ref.x.numpy())
    assert np.array_equal(np.array(graph.node_labels), ref.y.numpy())
    assert graph.edge_index.dtype == mx.int32


@pytest.mark.slow
@pytest.mark.parametrize(
    "name, to_undirected",
    [
        ("cora_ml", True),
        ("cora_ml", False),
        ("citeseer", True),
        ("dblp", True),
        ("pubmed", True),
    ],
)
def test_citation_full_matches_pyg(tmp_path, name, to_undirected):
    from torch_geometric.datasets import CitationFull as CitationFull_torch

    dataset = CitationFull(
        name, to_undirected=to_undirected, base_dir=str(tmp_path / "mxg")
    )
    ref = CitationFull_torch(str(tmp_path / "pyg"), name, to_undirected=to_undirected)
    assert len(dataset) == len(ref) == 1
    graph = dataset[0]
    _assert_same_graph(graph, ref[0])
    assert graph.is_undirected() == to_undirected
    assert not graph.has_self_loops()


@pytest.mark.slow
def test_cora_full_matches_pyg(tmp_path):
    from torch_geometric.datasets import CoraFull as CoraFull_torch

    graph = CoraFull(base_dir=str(tmp_path / "mxg"))[0]
    _assert_same_graph(graph, CoraFull_torch(str(tmp_path / "pyg"))[0])
    assert graph.num_nodes == 19793
    assert graph.num_edges == 126842


def test_citation_full_invalid_name(tmp_path):
    with pytest.raises(AssertionError):
        CitationFull("cora_full", base_dir=str(tmp_path))
