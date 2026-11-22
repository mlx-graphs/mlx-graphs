import mlx.core as mx
import numpy as np

from mlx_graphs.data import GraphData


def random_graph(num_nodes: int = 30, num_edges: int = 120, seed: int = 0):
    """Random directed graph without self-loops nor duplicate edges, with
    node ids as features, node labels, a train mask and edge features equal
    to the edge ids."""
    rng = np.random.RandomState(seed)
    keys = set()
    while len(keys) < num_edges:
        u, v = rng.randint(0, num_nodes, 2)
        if u != v:
            keys.add((int(u), int(v)))
    edges = np.array(sorted(keys)).T
    rng.shuffle(edges.T)
    data = GraphData(
        edge_index=mx.array(edges),
        node_features=mx.arange(num_nodes).reshape(-1, 1).astype(mx.float32),
        edge_features=mx.arange(num_edges).astype(mx.float32),
        node_labels=mx.array(rng.randint(0, 3, num_nodes)),
    )
    data.train_mask = mx.array(rng.rand(num_nodes) < 0.5)
    return data


def edge_set(edge_index) -> set:
    e = np.asarray(edge_index)
    return set(zip(e[0].tolist(), e[1].tolist()))


def cliques_ring(num_cliques: int = 4, size: int = 8) -> GraphData:
    """``num_cliques`` cliques connected in a ring by one (undirected) edge."""
    edges = []
    n = num_cliques * size
    for c in range(num_cliques):
        nodes = range(size * c, size * (c + 1))
        edges += [(u, v) for u in nodes for v in nodes if u != v]
        a, b = size * c + size - 1, (size * (c + 1)) % n
        edges += [(a, b), (b, a)]
    return GraphData(
        edge_index=mx.array(np.array(edges).T),
        node_features=mx.arange(n).reshape(-1, 1).astype(mx.float32),
    )


def check_subgraph(batch, data, induced: bool = False):
    """Checks that ``batch`` is a consistent subgraph of ``data`` through its
    ``n_id`` / ``e_id``: features, labels, masks and edges map back to the
    original graph."""
    n_id = np.asarray(batch.n_id)
    e_id = np.asarray(batch.e_id)
    assert batch.num_nodes == n_id.size
    assert batch.edge_index.shape == (2, e_id.size)
    original = np.asarray(data.edge_index)
    local = np.asarray(batch.edge_index)
    if e_id.size:
        assert local.min() >= 0 and local.max() < n_id.size
        # Every edge maps back to its original edge.
        np.testing.assert_array_equal(n_id[local], original[:, e_id])
    np.testing.assert_array_equal(
        np.asarray(batch.node_features).reshape(-1), n_id.astype(np.float32)
    )
    if data.edge_features is not None:
        np.testing.assert_array_equal(np.asarray(batch.edge_features), e_id)
    if getattr(data, "train_mask", None) is not None:
        np.testing.assert_array_equal(
            np.asarray(batch.train_mask), np.asarray(data.train_mask)[n_id]
        )
    if induced:
        # All original edges between the nodes are present.
        in_sub = np.zeros(data.num_nodes, dtype=bool)
        in_sub[n_id] = True
        expected = np.nonzero(in_sub[original[0]] & in_sub[original[1]])[0]
        assert sorted(e_id.tolist()) == sorted(expected.tolist())
