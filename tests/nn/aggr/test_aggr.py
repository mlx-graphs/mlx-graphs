import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch
from mlx.utils import tree_flatten
from torch_geometric.nn import aggr as pyg_aggr

import mlx_graphs.nn as gnn
from mlx_graphs.nn import aggr

DIM = 4
# 5 sets (set 3 is empty), index not sorted
INDEX = np.array([0, 2, 1, 0, 4, 2, 2, 0, 1, 4, 4, 4, 0], dtype=np.int64)
NUM_SETS = 5
TOL = dict(rtol=1e-4, atol=1e-5)
# MLX's GPU matmul is only accurate to ~1e-3 relative on some machines, while a
# wrong formula gives O(1) errors with these O(1) inputs.
MATMUL_TOL = dict(rtol=2e-2, atol=2e-2)


def make_x(seed=0, positive=False):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((INDEX.size, DIM)).astype(np.float32)
    return np.abs(x) + 0.1 if positive else x


def run(mlx_aggr, pyg_module, x=None, index=INDEX, dim_size=NUM_SETS):
    x = make_x() if x is None else x
    out = mlx_aggr(mx.array(x), mx.array(index), dim_size)
    expected = pyg_module(torch.tensor(x), torch.tensor(index), dim_size=dim_size)
    return np.array(out), expected.detach().numpy()


SIMPLE = [
    (aggr.SumAggregation, pyg_aggr.SumAggregation, {}),
    (aggr.MeanAggregation, pyg_aggr.MeanAggregation, {}),
    (aggr.MaxAggregation, pyg_aggr.MaxAggregation, {}),
    (aggr.MinAggregation, pyg_aggr.MinAggregation, {}),
    (aggr.MulAggregation, pyg_aggr.MulAggregation, {}),
    (aggr.VarAggregation, pyg_aggr.VarAggregation, {}),
    (aggr.VarAggregation, pyg_aggr.VarAggregation, {"semi_grad": True}),
    (aggr.StdAggregation, pyg_aggr.StdAggregation, {}),
    (aggr.SoftmaxAggregation, pyg_aggr.SoftmaxAggregation, {}),
    (aggr.SoftmaxAggregation, pyg_aggr.SoftmaxAggregation, {"t": 0.3}),
    (aggr.SoftmaxAggregation, pyg_aggr.SoftmaxAggregation, {"semi_grad": True}),
    (aggr.SoftmaxAggregation, pyg_aggr.SoftmaxAggregation, {"learn": True}),
    (aggr.PowerMeanAggregation, pyg_aggr.PowerMeanAggregation, {}),
    (aggr.PowerMeanAggregation, pyg_aggr.PowerMeanAggregation, {"p": 2.5}),
    (aggr.PowerMeanAggregation, pyg_aggr.PowerMeanAggregation, {"learn": True}),
    (aggr.MedianAggregation, pyg_aggr.MedianAggregation, {}),
]


@pytest.mark.parametrize("mlx_cls, pyg_cls, kwargs", SIMPLE)
def test_matches_pyg(mlx_cls, pyg_cls, kwargs):
    positive = mlx_cls is aggr.PowerMeanAggregation
    out, expected = run(
        mlx_cls(**kwargs), pyg_cls(**kwargs), x=make_x(positive=positive)
    )
    assert out.shape == (NUM_SETS, DIM)
    np.testing.assert_allclose(out, expected, **TOL)


@pytest.mark.parametrize("cls", [aggr.SoftmaxAggregation, aggr.PowerMeanAggregation])
def test_learnable_channels(cls):
    pyg_cls = getattr(pyg_aggr, cls.__name__)
    module = cls(learn=True, channels=DIM)
    pyg_module = pyg_cls(learn=True, channels=DIM)
    values = np.linspace(0.5, 2.0, DIM).astype(np.float32)
    name = "t" if cls is aggr.SoftmaxAggregation else "p"
    setattr(module, name, mx.array(values))
    getattr(pyg_module, name).data = torch.tensor(values)

    out, expected = run(module, pyg_module, x=make_x(positive=True))
    np.testing.assert_allclose(out, expected, **TOL)
    assert [k for k, _ in tree_flatten(module.trainable_parameters())] == [name]

    # the parameter gets a gradient
    def loss(params):
        module.update(params)
        return module(mx.array(make_x(positive=True)), mx.array(INDEX)).sum()

    grads = mx.grad(loss)(module.trainable_parameters())
    assert np.abs(np.array(grads[name])).sum() > 0


def test_learn_options_validation():
    with pytest.raises(ValueError):
        aggr.SoftmaxAggregation(learn=True, semi_grad=True)
    with pytest.raises(ValueError):
        aggr.SoftmaxAggregation(channels=2)
    with pytest.raises(ValueError):
        aggr.PowerMeanAggregation(channels=2)


@pytest.mark.parametrize("q", [0.1, [0.25, 0.5, 0.75]])
@pytest.mark.parametrize("interpolation", ["linear", "lower", "midpoint"])
def test_quantile(q, interpolation):
    out, expected = run(
        aggr.QuantileAggregation(q, interpolation),
        pyg_aggr.QuantileAggregation(q, interpolation),
    )
    np.testing.assert_allclose(out, expected, **TOL)


def test_defaults_without_index_or_dim_size():
    x = mx.array(make_x())
    out = aggr.SumAggregation()(x)
    np.testing.assert_allclose(np.array(out), np.array(x.sum(0, keepdims=True)), **TOL)
    # dim_size inferred as index.max() + 1
    assert aggr.MeanAggregation()(x, mx.array(INDEX)).shape == (NUM_SETS, DIM)


@pytest.mark.parametrize(
    "mode", ["cat", "sum", "mean", "max", "min", "logsumexp", "std", "var"]
)
def test_multi_aggregation(mode):
    names = ["mean", "max", "std"]
    module = aggr.MultiAggregation(names, mode=mode)
    pyg_module = pyg_aggr.MultiAggregation(names, mode=mode)
    out, expected = run(module, pyg_module)
    np.testing.assert_allclose(out, expected, **TOL)
    assert out.shape[1] == module.get_out_features_dim(DIM)


def test_multi_aggregation_std_gradient_of_equal_aggregations():
    # A set with a single element (or of zeros) gives equal aggregations, whose
    # std has a zero gradient in torch, not NaN.
    names = ["mean", "max", "sum"]
    x = np.array([[0.0, 2.0], [3.0, 1.0], [5.0, 0.0]], dtype=np.float32)
    index = np.array([0, 1, 1])

    module = aggr.MultiAggregation(names, mode="std")
    grad = mx.grad(lambda v: module(v, mx.array(index), 2).sum())(mx.array(x))

    x_torch = torch.tensor(x, requires_grad=True)
    pyg_aggr.MultiAggregation(names, mode="std")(
        x_torch, torch.tensor(index), dim_size=2
    ).sum().backward()
    np.testing.assert_allclose(np.array(grad), x_torch.grad.numpy(), **TOL)


def test_multi_aggregation_proj_and_kwargs():
    module = aggr.MultiAggregation(
        ["mean", aggr.SoftmaxAggregation(t=0.5), "powermean"],
        aggrs_kwargs=[{}, {}, {"p": 2.0}],
        mode="proj",
        mode_kwargs={"node_features_dim": DIM, "out_features_dim": 6},
    )
    pyg_module = pyg_aggr.MultiAggregation(
        ["mean", pyg_aggr.SoftmaxAggregation(t=0.5), "powermean"],
        aggrs_kwargs=[{}, {}, {"p": 2.0}],
        mode="proj",
        mode_kwargs={"in_channels": DIM, "out_channels": 6},
    )
    module.lin.weight = mx.array(pyg_module.lin.weight.detach().numpy())
    module.lin.bias = mx.array(pyg_module.lin.bias.detach().numpy())
    out, expected = run(module, pyg_module, x=make_x(positive=True))
    np.testing.assert_allclose(out, expected, **MATMUL_TOL)
    assert module.get_out_features_dim(DIM) == 6

    with pytest.raises(ValueError):
        aggr.MultiAggregation(["mean", "max"], mode="proj")
    with pytest.raises(ValueError):
        aggr.MultiAggregation([])
    with pytest.raises(ValueError):
        aggr.MultiAggregation(["mean"], mode="attn")


DEG = np.array([1, 3, 5, 2, 1])


@pytest.mark.parametrize(
    "aggregators, scalers",
    [
        ("mean", "identity"),
        (["mean", "min", "max", "std"], ["identity", "amplification", "attenuation"]),
        (["sum", "var"], ["linear", "inverse_linear"]),
    ],
)
@pytest.mark.parametrize("train_norm", [False, True])
def test_degree_scaler(aggregators, scalers, train_norm):
    module = aggr.DegreeScalerAggregation(aggregators, scalers, DEG, train_norm)
    pyg_module = pyg_aggr.DegreeScalerAggregation(
        aggregators, scalers, torch.tensor(DEG), train_norm
    )
    out, expected = run(module, pyg_module)
    np.testing.assert_allclose(out, expected, **TOL)
    num_params = len(tree_flatten(module.trainable_parameters()))
    assert num_params == (2 if train_norm else 0)


def test_degree_scaler_three_dimensional():
    x = np.random.default_rng(0).standard_normal((INDEX.size, 2, 3)).astype(np.float32)
    module = aggr.DegreeScalerAggregation(["mean", "max"], ["identity", "linear"], DEG)
    out = module(mx.array(x), mx.array(INDEX), NUM_SETS)
    assert out.shape == (NUM_SETS, 2, 12)
    with pytest.raises(ValueError):
        aggr.DegreeScalerAggregation("mean", "square", DEG)


@pytest.mark.parametrize("k", [1, 3, 6])
def test_sort_aggregation(k):
    # sorted index, as PyG requires; distinct values so the order is unique
    index = np.sort(INDEX)
    x = make_x(seed=3)
    out, expected = run(aggr.SortAggregation(k), pyg_aggr.SortAggregation(k), x, index)
    assert out.shape == (NUM_SETS, k * DIM)
    np.testing.assert_allclose(out, expected, **TOL)


def test_sort_aggregation_unsorted_index_and_gradient():
    x = make_x(seed=4)
    perm = np.random.default_rng(0).permutation(INDEX.size)
    pool = aggr.SortAggregation(3)
    sorted_order = np.argsort(INDEX, kind="stable")
    expected = pool(mx.array(x[sorted_order]), mx.array(INDEX[sorted_order]), NUM_SETS)
    out = pool(mx.array(x[perm]), mx.array(INDEX[perm]), NUM_SETS)
    np.testing.assert_allclose(np.array(out), np.array(expected), **TOL)

    grad = mx.grad(lambda v: pool(v, mx.array(INDEX), NUM_SETS).sum())(mx.array(x))
    # every set has at most 3 elements kept
    assert int((np.array(grad).sum(1) > 0).sum()) <= 3 * NUM_SETS

    out = gnn.global_sort_pool(mx.array(x), mx.array(INDEX), k=2)
    assert out.shape == (NUM_SETS, 2 * DIM)


def test_deep_sets():
    local_nn = nn.Linear(DIM, 8)
    global_nn = nn.Linear(8, 3)
    module = aggr.DeepSetsAggregation(local_nn, global_nn)
    pyg_module = pyg_aggr.DeepSetsAggregation(
        torch.nn.Linear(DIM, 8), torch.nn.Linear(8, 3)
    )
    for mlx_lin, torch_lin in [
        (local_nn, pyg_module.local_nn),
        (global_nn, pyg_module.global_nn),
    ]:
        mlx_lin.weight = mx.array(torch_lin.weight.detach().numpy())
        mlx_lin.bias = mx.array(torch_lin.bias.detach().numpy())
    out, expected = run(module, pyg_module)
    np.testing.assert_allclose(out, expected, **MATMUL_TOL)
    assert len(tree_flatten(module.trainable_parameters())) == 4
    assert aggr.DeepSetsAggregation()(mx.ones((3, 2))).shape == (1, 2)


@pytest.mark.parametrize(
    "name, cls, kwargs",
    [
        ("lstm", "LSTMAggregation", dict(in_channels=DIM, out_channels=3)),
        ("GRU", "GRUAggregation", dict(node_features_dim=DIM, out_features_dim=3)),
        ("sort", "SortAggregation", dict(k=2)),
        ("set_transformer", "SetTransformerAggregation", dict(channels=DIM)),
        (
            "graph_multiset_transformer",
            "GraphMultisetTransformer",
            dict(channels=DIM, k=2),
        ),
        (
            "mlp",
            "MLPAggregation",
            dict(in_channels=DIM, out_channels=3, max_num_elements=4),
        ),
        ("lcm", "LCMAggregation", dict(in_channels=DIM, out_channels=DIM)),
        ("deep_sets", "DeepSetsAggregation", dict()),
        ("variance_preserving", "VariancePreservingAggregation", dict()),
        ("MedianAggregation", "MedianAggregation", dict()),
    ],
)
def test_resolver_knows_every_aggregation(name, cls, kwargs):
    # As PyG's resolver: every aggregation class by (normalized) name, with
    # PyG's in_channels / out_channels accepted.
    module = aggr.aggregation_resolver(name, **kwargs)
    assert type(module) is getattr(aggr, cls)
    assert module(mx.ones((6, DIM)), mx.array([0, 0, 1, 1, 1, 2]), 3).shape[0] == 3


def test_resolver_list_is_multi_aggregation():
    module = aggr.aggregation_resolver(["mean", "max"], mode="sum")
    assert isinstance(module, aggr.MultiAggregation) and module.mode == "sum"


def test_resolver_and_reexports():
    assert isinstance(aggr.aggregation_resolver("sum"), aggr.SumAggregation)
    assert isinstance(aggr.aggregation_resolver("add"), aggr.SumAggregation)
    resolved = aggr.aggregation_resolver("SoftmaxAggregation", t=2.0)
    assert isinstance(resolved, aggr.SoftmaxAggregation)
    instance = aggr.MaxAggregation()
    assert aggr.aggregation_resolver(instance) is instance
    with pytest.raises(ValueError):
        aggr.aggregation_resolver("unknown")

    from mlx_graphs.nn.pooling.attentional_aggregation import AttentionalAggregation
    from mlx_graphs.nn.pooling.set2set import Set2Set

    assert aggr.Set2Set is Set2Set
    assert aggr.AttentionalAggregation is AttentionalAggregation
    assert gnn.MultiAggregation is aggr.MultiAggregation
