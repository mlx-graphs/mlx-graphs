import os

import mlx.core as mx
import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
pytest.importorskip("networkx")

from mlx_graphs.explain.explanation import Explanation  # noqa: E402
from mlx_graphs.explain.visualization import has_graphviz, visualize_graph  # noqa: E402

matplotlib.use("Agg")  # headless


@pytest.fixture
def explanation():
    one_way = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    edge_index = mx.array(np.concatenate([one_way, one_way[::-1]], axis=1))
    return Explanation(
        node_mask=mx.array([[0.9], [0.5], [0.1], [0.0], [0.0]]),
        edge_mask=mx.array([0.9, 0.5, 0.2, 0.0, 0.8, 0.4, 0.1, 0.0]),
        edge_index=edge_index,
        index=1,
    )


def test_visualize_graph_networkx(explanation, tmp_path):
    path = str(tmp_path / "graph.png")
    ax = explanation.visualize_graph(path, backend="networkx")
    assert os.path.getsize(path) > 0
    # zero-weight edges (3->4 and 4->3) are not drawn, so node 4 is dropped
    drawn = {t.get_text() for t in ax.texts}
    assert drawn == {"0", "1", "2", "3"}


def test_visualize_graph_custom_labels(explanation, tmp_path):
    labels = ["a", "b", "c", "d", "e"]
    ax = explanation.visualize_graph(
        str(tmp_path / "g.png"), backend="networkx", node_labels=labels
    )
    assert {t.get_text() for t in ax.texts} == {"a", "b", "c", "d"}


@pytest.mark.skipif(not has_graphviz(), reason="graphviz not installed")
def test_visualize_graph_graphviz(explanation, tmp_path):
    path = str(tmp_path / "graph.svg")
    g = explanation.visualize_graph(path, backend="graphviz")
    assert os.path.getsize(path) > 0
    assert "#d62728" in g.source  # explained node highlighted
    assert '"4"' not in g.source  # isolated after dropping zero-weight edges


def test_visualize_graph_bad_backend(explanation):
    with pytest.raises(ValueError, match="backend"):
        explanation.visualize_graph(backend="plotly")


def test_visualize_graph_without_weights(tmp_path):
    path = str(tmp_path / "plain.png")
    visualize_graph(mx.array([[0, 1], [1, 2]]), path=path, backend="networkx")
    assert os.path.getsize(path) > 0


def test_feature_importance(tmp_path):
    exp = Explanation(node_mask=mx.array([[0.1, 0.9, 0.3], [0.2, 0.8, 0.0]]))
    path = str(tmp_path / "feat.png")
    ax = exp.visualize_feature_importance(path, feat_labels=["x", "y", "z"], top_k=2)
    assert os.path.getsize(path) > 0
    assert [t.get_text() for t in ax.get_yticklabels()] == ["x", "y"]  # top: y
    assert ax.get_title() == "Feature importance for top 2 features"


def test_feature_importance_needs_feature_mask(explanation):
    with pytest.raises(ValueError, match="one value per feature"):
        explanation.visualize_feature_importance()


def test_feature_importance_label_count():
    exp = Explanation(node_mask=mx.ones((2, 3)))
    with pytest.raises(ValueError, match="labels"):
        exp.visualize_feature_importance(feat_labels=["a"])


def test_edge_labels_networkx(explanation, tmp_path):
    ax = explanation.visualize_graph(
        str(tmp_path / "labels.png"), backend="networkx", edge_labels=True
    )
    texts = {t.get_text() for t in ax.texts}
    # original (unscaled) mask values, only for drawn edges
    assert {"0.90", "0.50", "0.20", "0.80", "0.40", "0.10"} <= texts
    assert "0.00" not in texts


@pytest.mark.skipif(not has_graphviz(), reason="graphviz not installed")
def test_edge_labels_graphviz(explanation, tmp_path):
    g = explanation.visualize_graph(
        str(tmp_path / "labels.svg"), backend="graphviz", edge_labels=True
    )
    assert "label=0.90" in g.source


def test_no_edge_labels_by_default(explanation, tmp_path):
    ax = explanation.visualize_graph(str(tmp_path / "plain.png"), backend="networkx")
    assert not any(t.get_text().startswith("0.") for t in ax.texts)
