import mlx.core as mx
import numpy as np

from mlx_graphs.data import GraphData
from mlx_graphs.utils.train_test_split_edges import train_test_split_edges
from mlx_graphs.utils.transformations import to_undirected


def edge_set(edges) -> set:
    return set(map(tuple, np.asarray(edges).T.tolist()))


def test_train_test_split_edges():
    num_nodes = 30
    rng = np.random.default_rng(0)
    upper = np.unique(rng.integers(0, num_nodes, size=(2, 120)), axis=1)
    upper = upper[:, upper[0] < upper[1]]
    num_upper = upper.shape[1]
    edge_index = to_undirected(mx.array(upper.astype(np.int32)), coalesce=True)
    # Features identify each undirected edge.
    src, dst = np.asarray(edge_index)
    keys = np.minimum(src, dst) * num_nodes + np.maximum(src, dst)
    edge_features = mx.array(keys.astype(np.float32)).reshape(-1, 1)
    data = GraphData(
        edge_index=edge_index,
        edge_features=edge_features,
        node_features=mx.ones((num_nodes, 2)),
    )

    mx.random.seed(0)
    out = train_test_split_edges(data, val_ratio=0.1, test_ratio=0.2)
    # The input is not modified.
    assert data.edge_index is edge_index and data.edge_features is edge_features
    assert out.edge_index is None and out.edge_features is None
    assert out.num_nodes == num_nodes

    n_v, n_t = int(0.1 * num_upper), int(0.2 * num_upper)
    val, test = np.asarray(out.val_pos_edge_index), np.asarray(out.test_pos_edge_index)
    train = np.asarray(out.train_pos_edge_index)
    assert val.shape == (2, n_v) and test.shape == (2, n_t)
    assert train.shape == (2, 2 * (num_upper - n_v - n_t))
    assert np.all(val[0] < val[1]) and np.all(test[0] < test[1])
    train_upper = train[:, train[0] < train[1]]
    assert edge_set(train) == edge_set(train_upper) | edge_set(train_upper[::-1])
    assert edge_set(val) | edge_set(test) | edge_set(train_upper) == edge_set(upper)
    assert not (edge_set(val) & edge_set(test))

    for split, idx in (("val", val), ("test", test), ("train", train)):
        feats = np.asarray(getattr(out, f"{split}_pos_edge_features")).reshape(-1)
        expected = np.minimum(idx[0], idx[1]) * num_nodes + np.maximum(idx[0], idx[1])
        assert np.array_equal(feats, expected)

    val_neg = np.asarray(out.val_neg_edge_index)
    test_neg = np.asarray(out.test_neg_edge_index)
    assert val_neg.shape == (2, n_v) and test_neg.shape == (2, n_t)
    negatives = edge_set(val_neg) | edge_set(test_neg)
    assert len(negatives) == n_v + n_t
    assert not (negatives & edge_set(upper))
    assert np.all(val_neg[0] < val_neg[1])

    neg_mask = np.asarray(out.train_neg_adj_mask)
    assert neg_mask.shape == (num_nodes, num_nodes) and neg_mask.dtype == bool
    assert not np.any(np.tril(neg_mask))
    assert not neg_mask[upper[0], upper[1]].any()
    assert not neg_mask[val_neg[0], val_neg[1]].any()
    total = num_nodes * (num_nodes - 1) // 2
    assert neg_mask.sum() == total - num_upper - n_v - n_t


def test_train_test_split_edges_without_features():
    data = GraphData(
        edge_index=mx.array([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]), num_nodes=4
    )
    out = train_test_split_edges(data, val_ratio=0.34, test_ratio=0.34)
    assert out.val_pos_edge_index.shape == (2, 1)
    assert out.test_pos_edge_index.shape == (2, 1)
    assert out.train_pos_edge_index.shape == (2, 2)
    assert not hasattr(out, "train_pos_edge_features")
