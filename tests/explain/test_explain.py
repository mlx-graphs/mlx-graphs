import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlx_graphs.explain import Explainer, ModelConfig
from mlx_graphs.explain.algorithm import GNNExplainer, IntegratedGradients, Saliency
from mlx_graphs.explain.datasets import ba_shapes
from mlx_graphs.explain.explanation import Explanation
from mlx_graphs.explain.metric import characterization_score, fidelity, groundtruth_auc
from mlx_graphs.nn import GCNConv

NODE_CFG = dict(mode="multiclass_classification", task_level="node", return_type="raw")
GRAPH_CFG = dict(
    mode="multiclass_classification", task_level="graph", return_type="raw"
)


class GCN(nn.Module):
    def __init__(self, in_dim=4, hidden=8, out_dim=3, layers=2, pool=False):
        super().__init__()
        dims = [in_dim] + [hidden] * layers
        self.convs = [GCNConv(a, b) for a, b in zip(dims[:-1], dims[1:], strict=True)]
        self.head = nn.Linear(hidden, out_dim)
        self.pool = pool

    def __call__(self, edge_index, node_features, edge_weights=None):
        h = node_features
        for conv in self.convs:
            h = nn.relu(conv(edge_index, h, edge_weights))
        if self.pool:
            h = h.mean(axis=0, keepdims=True)
        return self.head(h)


class NoEdgeWeightsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = GCNConv(4, 3)

    def __call__(self, edge_index, node_features):
        return self.conv(edge_index, node_features)


@pytest.fixture
def graph():
    """Two components: a path 0-1-2-3-4 and an isolated edge 5-6."""
    mx.random.seed(0)
    one_way = np.array([[0, 1, 2, 3, 5], [1, 2, 3, 4, 6]])
    edge_index = mx.array(np.concatenate([one_way, one_way[::-1]], axis=1))
    node_features = mx.random.normal((7, 4))
    return edge_index, node_features


def make(
    algorithm,
    model=None,
    node_mask_type="attributes",
    edge_mask_type="object",
    model_config=NODE_CFG,
    **kw,
):
    mx.random.seed(1)
    return Explainer(
        model=model or GCN(),
        algorithm=algorithm,
        explanation_type=kw.pop("explanation_type", "model"),
        node_mask_type=node_mask_type,
        edge_mask_type=edge_mask_type,
        model_config=model_config,
        **kw,
    )


# ---------------------------------------------------------------- config


def test_config_validation():
    with pytest.raises(ValueError):
        ModelConfig(mode="clustering")
    with pytest.raises(ValueError):
        make(Saliency(), node_mask_type=None, edge_mask_type=None)
    with pytest.raises(ValueError):
        make(Saliency(), node_mask_type="nodes")


def test_gradient_methods_reject_phenomenon():
    with pytest.raises(ValueError, match="does not support"):
        make(Saliency(), explanation_type="phenomenon")


def test_phenomenon_requires_target(graph):
    explainer = make(GNNExplainer(epochs=2), explanation_type="phenomenon")
    with pytest.raises(ValueError, match="target"):
        explainer(*graph, index=0)


def test_model_without_edge_weights_gives_clear_error(graph):
    explainer = make(Saliency(), model=NoEdgeWeightsModel())
    with pytest.raises(TypeError, match="edge_weights"):
        explainer(*graph, index=0)


# ---------------------------------------------------------------- shapes


@pytest.mark.parametrize("algorithm", [Saliency, IntegratedGradients, GNNExplainer])
@pytest.mark.parametrize(
    "node_mask_type,shape",
    [("object", (7, 1)), ("attributes", (7, 4)), ("common_attributes", (1, 4))],
)
def test_mask_shapes(graph, algorithm, node_mask_type, shape):
    algo = algorithm(epochs=5) if algorithm is GNNExplainer else algorithm()
    exp = make(algo, node_mask_type=node_mask_type)(*graph, index=2)
    assert isinstance(exp, Explanation)
    assert exp.node_mask.shape == shape
    assert exp.edge_mask.shape == (graph[0].shape[1],)
    assert exp.prediction.shape == (7, 3)


@pytest.mark.parametrize("algorithm", [Saliency, IntegratedGradients, GNNExplainer])
def test_nothing_outside_receptive_field(graph, algorithm):
    """Node 0 cannot see the other component (nodes 5, 6) nor nodes > 2 hops.

    Edges pointing into node 2 (e.g. 3->2) still matter: they change node 2's
    degree, hence the GCN normalization of the 2->1 message. Edges pointing
    into nodes 3..6 cannot affect node 0.
    """
    algo = algorithm(epochs=10) if algorithm is GNNExplainer else algorithm()
    exp = make(algo)(*graph, index=0)
    node_mask = np.asarray(exp.node_mask)
    assert np.all(node_mask[3:] == 0)
    edge_index = np.asarray(graph[0])
    far = np.isin(edge_index[1], [3, 4, 5, 6])
    assert np.all(np.asarray(exp.edge_mask)[far] == 0)


def test_gnn_explainer_masks_in_unit_interval(graph):
    exp = make(GNNExplainer(epochs=30, seed=0))(*graph, index=1)
    for m in (exp.node_mask, exp.edge_mask):
        m = np.asarray(m)
        assert m.min() >= 0 and m.max() <= 1
    assert np.isfinite(exp.extra["final_loss"])


def test_graph_level_task(graph):
    model = GCN(pool=True)
    for algo in (Saliency(), IntegratedGradients(n_steps=5), GNNExplainer(epochs=5)):
        exp = make(algo, model=model, model_config=GRAPH_CFG)(*graph)
        assert exp.node_mask.shape == (7, 4)
        assert exp.target.shape == (1,)


# ---------------------------------------------------------------- correctness


def test_integrated_gradients_completeness(graph):
    """Attributions sum to score(input) - score(baseline)."""
    edge_index, x = graph
    explainer = make(IntegratedGradients(n_steps=300))
    exp = explainer(edge_index, x, index=2)
    model, c = explainer.model, int(exp.target[2])
    full = model(edge_index, x, mx.ones((edge_index.shape[1],)))[2, c]
    base = model(edge_index, mx.zeros_like(x), mx.zeros((edge_index.shape[1],)))[2, c]
    total = exp.node_mask.sum() + exp.edge_mask.sum()
    assert abs(float(total) - float(full - base)) < 1e-2


def test_saliency_matches_finite_differences(graph):
    edge_index, x = graph
    explainer = make(Saliency(), edge_mask_type=None)
    exp = explainer(edge_index, x, index=1)
    c, eps = int(exp.target[1]), 1e-3
    model = explainer.model
    bump = mx.zeros_like(x).at[1, 0].add(eps)
    fd = (model(edge_index, x + bump)[1, c] - model(edge_index, x - bump)[1, c]) / (
        2 * eps
    )
    assert abs(abs(float(fd)) - float(exp.node_mask[1, 0])) < 1e-2


# ---------------------------------------------------------------- explanation utils


def test_thresholds():
    exp = Explanation(
        edge_mask=mx.array([0.1, 0.9, 0.5, 0.7]),
        edge_index=mx.array([[0, 1, 2, 3], [1, 2, 3, 0]]),
    )
    assert np.allclose(
        np.asarray(exp.threshold("hard", 0.6).edge_mask), [0, 0.9, 0, 0.7]
    )
    assert np.allclose(np.asarray(exp.threshold("topk", 2).edge_mask), [0, 0.9, 0, 0.7])
    assert np.allclose(
        np.asarray(exp.threshold("topk_hard", 3).edge_mask), [0, 1, 1, 1]
    )
    sub = exp.threshold("topk_hard", 2).get_explanation_subgraph()
    assert np.asarray(sub).tolist() == [[1, 3], [2, 0]]


def test_threshold_config_in_explainer(graph):
    explainer = make(
        Saliency(), threshold_config=dict(threshold_type="topk_hard", value=3)
    )
    exp = explainer(*graph, index=2)
    assert int(exp.edge_mask.sum()) == 3


# ---------------------------------------------------------------- metrics & data


def test_groundtruth_auc():
    gt = mx.array([0, 0, 1, 1])
    assert groundtruth_auc(mx.array([0.1, 0.2, 0.8, 0.9]), gt) == 1.0
    assert groundtruth_auc(mx.array([0.9, 0.8, 0.2, 0.1]), gt) == 0.0
    assert groundtruth_auc(mx.array([0.5, 0.5, 0.5, 0.5]), gt) == 0.5
    assert groundtruth_auc(mx.array([0.5, 0.5]), mx.array([1, 1])) is None


def test_fidelity_and_characterization(graph):
    explainer = make(Saliency())
    exp = explainer(*graph, index=2).threshold("topk_hard", 4)
    pos, neg = fidelity(explainer, exp)
    assert 0 <= pos <= 1 and 0 <= neg <= 1
    assert characterization_score(1.0, 0.0) == pytest.approx(1.0)


def test_ba_shapes():
    g = ba_shapes(num_base_nodes=50, num_motifs=10, seed=0)
    assert g.node_features.shape == (100, 10)
    assert g.edge_index.shape[1] == g.edge_groundtruth.shape[0]
    assert int(g.edge_groundtruth.sum()) == 10 * 12  # 6 undirected edges per house
    assert set(np.asarray(g.node_labels).tolist()) == {0, 1, 2, 3}


def test_gnn_explainer_edge_mask_starts_near_half(graph):
    """As in PyG, masks start around sigmoid(0) = 0.5 (regression test)."""
    exp = make(GNNExplainer(epochs=0, seed=0), node_mask_type=None)(*graph)
    m = np.asarray(exp.edge_mask)
    assert abs(m.mean() - 0.5) < 0.1
