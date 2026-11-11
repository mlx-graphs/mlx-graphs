import mlx.core as mx
import pytest

from mlx_graphs.algorithms import Node2Vec

mx.random.seed(42)


def test_node2vec():
    edge_index = mx.array([[0, 1, 2, 3], [0, 0, 1, 1]])

    embedding_size = 5
    model = Node2Vec(
        edge_index=edge_index,
        embedding_dim=embedding_size,
        walk_length=2,
        context_size=1,
        num_nodes=4,
    )
    embeddings = model(mx.arange(4).astype(mx.int64))
    assert embeddings.shape == (4, 5), "Embedding dimensions are not equal"


def _assert_walks_follow_edges(walks, edge_index):
    edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    has_out_edge = {u for u, _ in edges}
    for walk in walks.tolist():
        for u, v in zip(walk[:-1], walk[1:]):
            if u in has_out_edge:
                assert (u, v) in edges, f"step {u} -> {v} is not an edge"
            else:
                # The walk stays in place on a node without outgoing edges.
                assert u == v, f"node {u} has no outgoing edge but walked to {v}"


@pytest.mark.parametrize("p, q", [(1.0, 1.0), (2.0, 0.5)])
def test_node2vec_rowptr_with_zero_out_degree_nodes(p, q):
    # Node 1 (middle) and node 4 (last) have no outgoing edges.
    edge_index = mx.array([[0, 2, 2, 3], [1, 3, 0, 2]])
    num_nodes = 5
    model = Node2Vec(
        edge_index=edge_index,
        embedding_dim=4,
        walk_length=5,
        context_size=5,
        num_nodes=num_nodes,
        p=p,
        q=q,
    )
    assert model.rowptr.tolist() == [0, 1, 1, 3, 4, 4]
    assert sorted(model.col.tolist()[1:3]) == [0, 3]

    walks = model.pos_sample(mx.arange(num_nodes))
    _assert_walks_follow_edges(walks, edge_index)
