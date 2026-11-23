import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pytest

from mlx_graphs.algorithms import MetaPath2Vec

USE_GPU = mx.metal.is_available()

# Toy bibliographic graph: 4 authors, 5 papers, 2 venues.
# Author 3 writes nothing, so walks from it end right away.
WRITES = [[0, 0, 1, 2, 2], [0, 1, 2, 3, 4]]  # author -> paper
PUBLISHED_IN = [[0, 1, 2, 3, 4], [0, 0, 1, 1, 1]]  # paper -> venue
NUM_NODES = {"author": 4, "paper": 5, "venue": 2}

METAPATH = [
    ("author", "writes", "paper"),
    ("paper", "published_in", "venue"),
    ("venue", "publishes", "paper"),
    ("paper", "written_by", "author"),
]


def _edge_index_dict():
    writes = mx.array(WRITES)
    published_in = mx.array(PUBLISHED_IN)
    return {
        ("author", "writes", "paper"): writes,
        ("paper", "written_by", "author"): writes[::-1],
        ("paper", "published_in", "venue"): published_in,
        ("venue", "publishes", "paper"): published_in[::-1],
    }


def _model(**kwargs):
    args = dict(
        edge_index_dict=_edge_index_dict(),
        embedding_dim=8,
        metapath=METAPATH,
        walk_length=8,
        context_size=3,
        num_nodes_dict=NUM_NODES,
        use_gpu=USE_GPU,
    )
    args.update(kwargs)
    return MetaPath2Vec(**args)


def _to_local(model, global_ids):
    """Global embedding row -> (node_type, local id)."""
    for t in model.node_types:
        start = model._start[t]
        if start <= global_ids < model._end[t]:
            return t, global_ids - start
    raise AssertionError(f"{global_ids} is not an embedding row")


def test_embedding_shapes():
    model = _model()
    assert model.node_types == ["author", "paper", "venue"]
    assert model.num_nodes_dict == NUM_NODES
    assert model.embedding.weight.shape == (11, 8)
    assert model("author").shape == (4, 8)
    assert model("paper").shape == (5, 8)
    assert model("venue", mx.array([1])).shape == (1, 8)
    with pytest.raises(KeyError):
        model("term")


def test_only_embedding_is_trainable():
    from mlx.utils import tree_flatten

    names = [k for k, _ in tree_flatten(_model().trainable_parameters())]
    assert names == ["embedding.weight"]


def test_positive_windows_follow_metapath():
    mx.random.seed(0)
    model = _model(walks_per_node=4)
    edge_sets = {
        et: set(zip(*np.asarray(ei).tolist())) for et, ei in _edge_index_dict().items()
    }
    pos = model.pos_sample(mx.arange(4))
    assert pos.ndim == 2 and pos.shape[1] == 3
    assert pos.shape[0] > 0

    expected_types = model._walk_types
    for window in pos.tolist():
        local = [_to_local(model, g) for g in window]
        types = [t for t, _ in local]
        # A window starting at walk position j must have the types of
        # positions j .. j + context_size - 1.
        assert any(
            types == expected_types[j : j + 3] for j in range(len(expected_types) - 2)
        ), types
        for (t_u, u), (t_v, v) in zip(local[:-1], local[1:]):
            ets = [et for et in METAPATH if et[0] == t_u and et[2] == t_v]
            assert any((u, v) in edge_sets[et] for et in ets), (t_u, u, t_v, v)


def test_dead_end_windows_are_dropped():
    model = _model()
    # Author 3 has no papers: every window of its walk hits padding.
    pos = model.pos_sample(mx.array([3]))
    assert pos.shape == (0, 3)
    # Negative samples don't depend on the graph, so they are still there.
    neg = model.neg_sample(mx.array([3]))
    assert neg.shape == (8 + 2 - 3, 3)
    # The loss stays finite with no positive windows.
    assert np.isfinite(model.loss(pos, neg).item())


def test_negative_windows_have_metapath_types():
    mx.random.seed(1)
    model = _model(walks_per_node=2, num_negative_samples=3)
    batch = mx.array([0, 1, 2])
    neg = model.neg_sample(batch)
    num_windows = 8 + 2 - 3
    assert neg.shape == (3 * 2 * 3 * num_windows, 3)

    # The first block of windows starts at the batch nodes themselves.
    first = neg[: 3 * 2 * 3, 0].tolist()
    assert sorted(set(first)) == [0, 1, 2]
    for j in range(num_windows):
        block = neg[j * 18 : (j + 1) * 18].tolist()
        for window in block:
            types = [_to_local(model, g)[0] for g in window]
            assert types == model._walk_types[j : j + 3]


def test_dataloader_covers_start_nodes():
    model = _model()
    batches = list(model.dataloader(batch_size=3, shuffle=False))
    assert len(batches) == 2
    # Negative walks start at each start node exactly once per window.
    starts = mx.concatenate([neg[: neg.shape[0] // 7, 0] for _, neg in batches])
    assert sorted(starts.tolist()) == [0, 1, 2, 3]


def test_loss_decreases():
    mx.random.seed(2)
    np.random.seed(2)
    model = _model(walks_per_node=5, num_negative_samples=2)
    optimizer = optim.Adam(learning_rate=0.05)
    loss_and_grad = nn.value_and_grad(model, model.loss)

    def epoch():
        total = 0.0
        for pos, neg in model.dataloader(batch_size=4):
            loss, grads = loss_and_grad(pos, neg)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            total += loss.item()
        return total

    first = epoch()
    for _ in range(30):
        last = epoch()
    assert last < first


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"context_size": 1}, "context_size"),
        ({"context_size": 10}, "context_size"),
        ({"walk_length": 0}, "walk_length"),
        ({"metapath": METAPATH[:2], "walk_length": 3}, "cyclic"),
        ({"metapath": [METAPATH[0], METAPATH[2]]}, "Invalid metapath"),
    ],
)
def test_invalid_arguments(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _model(**kwargs)
