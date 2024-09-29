import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.utils.sorting import sort_edge_index

try:
    from mlx_cluster import random_walk, rejection_sampling
except ImportError:
    raise ImportError(
        "mlx_cluster is required for performing random walks",
        "run `pip install mlx_cluster`",
    )
import numpy as np


class Node2Vec(nn.Module):
    r"""The Node2Vec model from the
    `"node2vec: Scalable Feature Learning for Networks"
    <https://arxiv.org/abs/1607.00653>`_ paper where random walks of
    length :obj:`walk_length` are sampled in a given graph, and node embeddings
    are learned via negative sampling optimization

    Args:
        edge_index (mx.array): The edge indices.
        embedding_dim (int): The size of each embedding vector.
        walk_length (int): The walk length.
        context_size (int): The actual context size which is considered for
            positive samples. This parameter increases the effective sampling
            rate by reusing samples across different source nodes.
        num_nodes (int): Number of nodes in a graph
        walks_per_node (int, optional): The number of walks to sample for each
            node. (default: :obj:`1`)
        p (float, optional): Likelihood of immediately revisiting a node in the
            walk. (default: :obj:`1`)
        q (float, optional): Control parameter to interpolate between
            breadth-first strategy and depth-first strategy (default: :obj:`1`)
        num_negative_samples (int, optional): The number of negative samples to
            use for each positive sample. (default: :obj:`1`)
    """

    def __init__(
        self,
        edge_index: mx.array,
        embedding_dim: int,
        walk_length: int,
        context_size: int,
        num_nodes: int,
        walks_per_node: int = 1,
        p: float = 1.0,
        q: float = 1.0,
        num_negative_samples: int = 1,
    ):
        super().__init__()
        self.edge_index = edge_index.astype(mx.int64)
        self.num_nodes = num_nodes
        self.p = p
        self.q = q
        self.walk_length = walk_length - 1
        self.num_negative_samples = num_negative_samples
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(self.num_nodes, embedding_dim)
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.EPS = 1e-15
        assert walk_length >= context_size

        # Converting a CSC matrix to a CSR matrix
        sorted_edge_index = sort_edge_index(edge_index=self.edge_index)
        row = sorted_edge_index[0][0]
        col = sorted_edge_index[0][1]
        _, counts_mlx = np.unique(np.array(row, copy=False), return_counts=True)
        cum_sum = counts_mlx.cumsum()
        self.rowptr = mx.concatenate([mx.array([0]), mx.array(cum_sum)])
        self.col = col

    def __call__(self, batch):
        """Returns the embeddings for the nodes in :obj:`batch`."""
        emb = self.embedding.weight
        return emb if batch is None else emb[batch]

    def pos_sample(self, batch: mx.array):
        batch = mx.repeat(batch, self.walks_per_node)

        rand_data = mx.random.uniform(shape=[self.num_nodes, self.walk_length])
        if self.p == 1.0 and self.q == 1.0:
            rw = random_walk(
                self.rowptr.astype(mx.int64),
                self.col.astype(mx.int64),
                batch,
                rand_data,
                stream=mx.cpu,
            )
        else:
            rw = rejection_sampling(
                self.rowptr.astype(mx.int64),
                self.col.astype(mx.int64),
                batch,
                self.walk_length,
                self.p,
                self.q,
                stream=mx.cpu,
            )

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size

        for j in range(num_walks_per_rw):
            walks.append(rw[:, j : j + self.context_size])
        walks = mx.concatenate(walks, 0)
        return walks

    def neg_sample(
        self,
        batch: mx.array,
    ):
        batch = mx.repeat(batch, self.walks_per_node * self.num_negative_samples)

        rw = mx.random.randint(0, self.num_nodes, (batch.shape[0], self.walk_length))
        rw = mx.concatenate([batch.reshape(-1, 1), rw], axis=-1)
        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size

        for j in range(num_walks_per_rw):
            walks.append(rw[:, j : j + self.context_size])
        walks = mx.concatenate(walks, 0)
        return walks

    def dataloader(self, batch_size):
        r"""Dataloader for nodes"""
        data_array = mx.arange(self.num_nodes).astype(mx.int64)
        perm = mx.array(np.random.permutation(self.num_nodes))

        for s in range(0, self.num_nodes, batch_size):
            ids = perm[s : s + batch_size]
            yield (
                self.pos_sample(data_array[ids]),
                self.neg_sample(data_array[ids]),
            )

    def loss(self, pos_array, neg_array):
        r"""Computes the loss given positive and negative random walks."""

        # Positive loss
        start, rest = pos_array[:, 0], pos_array[:, 1:]
        h_start = self.embedding(start).reshape(pos_array.shape[0], 1, -1)

        h_rest = self.embedding(rest.reshape(-1)).reshape(
            pos_array.shape[0], -1, self.embedding_dim
        )

        out = (h_start * h_rest).sum(axis=-1)
        pos_loss = -mx.mean(mx.log(mx.sigmoid(out) + self.EPS))

        # Negative loss
        start, rest = neg_array[:, 0], neg_array[:, 1:]
        h_start = self.embedding(start).reshape(neg_array.shape[0], 1, -1)
        h_rest = self.embedding(rest.reshape(-1)).reshape(
            neg_array.shape[0], -1, self.embedding_dim
        )
        out = (h_start * h_rest).sum(axis=-1)
        neg_loss = -mx.mean(mx.log(1 - mx.sigmoid(out) + self.EPS))

        return pos_loss + neg_loss
