import mlx.core as mx

from mlx_graphs.nn.algorithms import Node2Vec

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
