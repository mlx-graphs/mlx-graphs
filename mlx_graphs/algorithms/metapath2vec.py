from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_graphs.sampling import (
    build_metapath_csr,
    metapath_node_types,
    metapath_random_walk,
)

NodeType = str
EdgeType = Tuple[str, str, str]


class MetaPath2Vec(nn.Module):
    r"""The MetaPath2Vec model from the `"metapath2vec: Scalable Representation
    Learning for Heterogeneous Networks"
    <https://ericdongyx.github.io/papers/KDD17-dong-chawla-swami-metapath2vec.pdf>`_
    paper.

    Random walks that follow a given meta-path (a sequence of edge types) are
    sampled in a heterogeneous graph, and node embeddings are learned with a
    skip-gram objective and negative sampling. Negative samples respect node
    types: each position of a negative walk gets a random node of the type
    the meta-path puts at that position.

    All node types visited by the meta-path share one embedding table, so
    they are embedded in the same space. Use ``model(node_type)`` to get the
    embeddings of one node type.

    .. note::
        This feature is experimental. Walks use ``mlx_cluster>=0.1.2`` when
        it is installed and MLX operations otherwise (e.g. on Linux).

    Args:
        edge_index_dict: ``(src, rel, dst) -> [2, num_edges]`` edge indices,
            with node ids local to each node type, e.g.
            ``HeteroGraphData.edge_index_dict``.
        embedding_dim: The size of each embedding vector.
        metapath: The meta-path, a list of ``(src, rel, dst)`` edge types. Each
            edge type must start at the node type the previous one ends at.
        walk_length: Number of steps per random walk, so every walk has
            ``walk_length + 1`` nodes. If it is longer than the meta-path, the
            meta-path must be cyclic (end at the node type it starts from).
        context_size: Size of the context windows cut out of each walk; the
            first node of a window is paired with the other
            ``context_size - 1`` nodes as positive samples.
        walks_per_node: Number of walks to start from each node of type
            ``metapath[0][0]``. (default: :obj:`1`)
        num_negative_samples: Number of negative walks per positive walk.
            (default: :obj:`1`)
        num_nodes_dict: Optional number of nodes per node type, e.g.
            ``HeteroGraphData.num_nodes``. Types not given are inferred from
            the largest node id in ``edge_index_dict``. (default: :obj:`None`)
        use_gpu: Whether to sample the random walks on the GPU.
            (default: :obj:`True`)

    Example:

    .. code-block:: python

        metapath = [
            ("author", "writes", "paper"),
            ("paper", "published_in", "venue"),
            ("venue", "publishes", "paper"),
            ("paper", "written_by", "author"),
        ]
        model = MetaPath2Vec(
            edge_index_dict,
            embedding_dim=128,
            metapath=metapath,
            walk_length=20,
            context_size=7,
        )
        optimizer = optim.Adam(learning_rate=0.01)
        loss_and_grad = nn.value_and_grad(model, model.loss)
        for pos, neg in model.dataloader(batch_size=128):
            loss, grads = loss_and_grad(pos, neg)
            optimizer.update(model, grads)

        author_embeddings = model("author")
    """

    def __init__(
        self,
        edge_index_dict: Mapping[EdgeType, mx.array],
        embedding_dim: int,
        metapath: Sequence[EdgeType],
        walk_length: int,
        context_size: int,
        walks_per_node: int = 1,
        num_negative_samples: int = 1,
        num_nodes_dict: Optional[Mapping[NodeType, int]] = None,
        use_gpu: bool = True,
    ):
        super().__init__()
        if walk_length < 1:
            raise ValueError("walk_length must be >= 1")
        if not 2 <= context_size <= walk_length + 1:
            raise ValueError(
                f"context_size ({context_size}) must be between 2 and "
                f"walk_length + 1 ({walk_length + 1})"
            )
        if walks_per_node < 1 or num_negative_samples < 1:
            raise ValueError("walks_per_node and num_negative_samples must be >= 1")

        # Validates the meta-path and sorts every relation into a CSR once.
        # Attributes starting with "_" are not treated as model parameters.
        self._csr = build_metapath_csr(edge_index_dict, metapath, num_nodes_dict)
        metapath = self._csr.metapath
        m = len(metapath)
        if walk_length > m and metapath[-1][2] != metapath[0][0]:
            raise ValueError(
                f"walk_length ({walk_length}) is longer than the metapath ({m}), "
                "so the metapath must be cyclic: it ends at "
                f"'{metapath[-1][2]}' but starts at '{metapath[0][0]}'"
            )
        num_nodes = self._csr.num_nodes_dict

        # One embedding table for all node types on the meta-path; node type t
        # owns rows [start[t], end[t]).
        node_types: List[NodeType] = []
        for t in [metapath[0][0]] + [et[2] for et in metapath]:
            if t not in node_types:
                node_types.append(t)
        self._start: Dict[NodeType, int] = {}
        self._end: Dict[NodeType, int] = {}
        offset = 0
        for t in node_types:
            n = int(num_nodes.get(t, 0))
            if n == 0:
                raise ValueError(f"Node type '{t}' on the metapath has no nodes")
            self._start[t], self._end[t] = offset, offset + n
            offset += n

        # Node type of every walk position, as embedding offsets and sizes.
        self._walk_types = metapath_node_types(metapath, walk_length)
        self._walk_offset = mx.array(
            [self._start[t] for t in self._walk_types], dtype=mx.int64
        )
        self._walk_count = mx.array(
            [self._end[t] - self._start[t] for t in self._walk_types],
            dtype=mx.int64,
        )

        self.metapath = metapath
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.num_negative_samples = num_negative_samples
        self.use_gpu = use_gpu
        self.embedding = nn.Embedding(offset, embedding_dim)

    @property
    def node_types(self) -> List[NodeType]:
        """Node types with an embedding, in embedding-table order."""
        return list(self._start)

    @property
    def num_nodes_dict(self) -> Dict[NodeType, int]:
        """Number of embedded nodes per node type."""
        return {t: self._end[t] - self._start[t] for t in self._start}

    def __call__(
        self, node_type: NodeType, batch: Optional[mx.array] = None
    ) -> mx.array:
        """Returns the embeddings of the nodes of type ``node_type``.

        Args:
            node_type: The node type.
            batch: Optional local node ids of that type. All nodes of the type
                are returned if omitted.
        """
        if node_type not in self._start:
            raise KeyError(
                f"Node type '{node_type}' is not on the metapath; "
                f"embedded types are {self.node_types}"
            )
        emb = self.embedding.weight[self._start[node_type] : self._end[node_type]]
        return emb if batch is None else emb[batch]

    def _windows(self, walks: mx.array) -> mx.array:
        """Cuts ``[B, walk_length + 1]`` walks into ``context_size`` windows."""
        num_windows = self.walk_length + 2 - self.context_size
        return mx.concatenate(
            [walks[:, j : j + self.context_size] for j in range(num_windows)], axis=0
        )

    def pos_sample(self, batch: mx.array) -> mx.array:
        """Samples positive context windows from meta-path random walks.

        Args:
            batch: Local ids of start nodes, of type ``metapath[0][0]``.

        Returns:
            ``[num_windows, context_size]`` global embedding indices. Windows
            that run past the end of a walk that hit a dead end (a node with
            no neighbor of the next edge type) are dropped.
        """
        batch = mx.repeat(batch.astype(mx.int64).reshape(-1), self.walks_per_node)
        nodes, _ = metapath_random_walk(
            self._csr,
            batch,
            self.walk_length,
            stream=mx.gpu if self.use_gpu else mx.cpu,
        )
        # Local ids -> embedding rows; dead-end padding (-1) stays -1.
        walks = mx.where(nodes >= 0, nodes + self._walk_offset, -1)
        windows = self._windows(walks)
        keep = np.asarray(mx.all(windows >= 0, axis=1))
        if keep.all():
            return windows
        return windows[mx.array(np.flatnonzero(keep))]

    def neg_sample(self, batch: mx.array) -> mx.array:
        """Samples negative context windows.

        Each negative walk starts at a node of ``batch`` and continues with
        random nodes of the type the meta-path puts at each position.

        Args:
            batch: Local ids of start nodes, of type ``metapath[0][0]``.

        Returns:
            ``[num_windows, context_size]`` global embedding indices.
        """
        batch = batch.astype(mx.int64).reshape(-1)
        batch = mx.repeat(batch, self.walks_per_node * self.num_negative_samples)
        u = mx.random.uniform(shape=(batch.shape[0], self.walk_length))
        counts = self._walk_count[1:]
        rest = mx.minimum((u * counts).astype(mx.int64), counts - 1)
        walks = mx.concatenate([batch[:, None], rest], axis=1) + self._walk_offset
        return self._windows(walks)

    def dataloader(
        self, batch_size: int, shuffle: bool = True
    ) -> Iterator[Tuple[mx.array, mx.array]]:
        """Yields ``(pos, neg)`` samples for mini-batches of start nodes.

        Every node of type ``metapath[0][0]`` is used as a start node once per
        pass.

        Args:
            batch_size: Number of start nodes per mini-batch.
            shuffle: Whether to shuffle the start nodes. (default: :obj:`True`)
        """
        start_type = self.metapath[0][0]
        n = self._end[start_type] - self._start[start_type]
        order = np.random.permutation(n) if shuffle else np.arange(n)
        for s in range(0, n, batch_size):
            batch = mx.array(order[s : s + batch_size], dtype=mx.int64)
            yield self.pos_sample(batch), self.neg_sample(batch)

    def _skipgram_scores(self, windows: mx.array) -> mx.array:
        h_start = self.embedding(windows[:, :1])  # [N, 1, D]
        h_rest = self.embedding(windows[:, 1:])  # [N, C - 1, D]
        return (h_start * h_rest).sum(axis=-1)

    def loss(self, pos_rw: mx.array, neg_rw: mx.array) -> mx.array:
        """Skip-gram loss with negative sampling.

        Args:
            pos_rw: Positive windows from :meth:`pos_sample`.
            neg_rw: Negative windows from :meth:`neg_sample`.
        """
        # log_sigmoid is the numerically stable log(sigmoid(x)).
        loss = -nn.log_sigmoid(-self._skipgram_scores(neg_rw)).mean()
        if pos_rw.shape[0] > 0:
            loss = loss - nn.log_sigmoid(self._skipgram_scores(pos_rw)).mean()
        return loss
