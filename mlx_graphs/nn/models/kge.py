import math
from typing import Iterator, Tuple

import mlx.core as mx
import mlx.nn as nn
from tqdm import tqdm

Triplets = Tuple[mx.array, mx.array, mx.array]


def _glorot_embedding(num: int, dim: int) -> nn.Embedding:
    emb = nn.Embedding(num, dim)
    emb.weight = nn.init.glorot_uniform()(emb.weight)
    return emb


def _p_norm(x: mx.array, p: float, axis: int = -1) -> mx.array:
    if p == 1:
        return mx.abs(x).sum(axis=axis)
    if p == 2:
        return mx.sqrt((x * x).sum(axis=axis))
    return (mx.abs(x) ** p).sum(axis=axis) ** (1.0 / p)


def _normalize(x: mx.array, p: float, eps: float = 1e-12) -> mx.array:
    """``x / max(||x||_p, eps)`` along the last axis, like
    ``torch.nn.functional.normalize``."""
    return x / mx.maximum(_p_norm(x, p)[..., None], eps)


class KGTripletLoader:
    """Iterates over mini-batches of ``(head_index, rel_type, tail_index)``
    triplets.

    Args:
        head_index: The head indices, shape `[num_triplets]`
        rel_type: The relation types, shape `[num_triplets]`
        tail_index: The tail indices, shape `[num_triplets]`
        batch_size: Number of triplets per batch. Default: ``1``
        shuffle: Whether to shuffle the triplets at every epoch (with
            ``mx.random``). Default: ``False``
        drop_last: Whether to drop the last incomplete batch. Default: ``False``

    Example:

    .. code-block:: python

        loader = model.loader(
            head_index, rel_type, tail_index, batch_size=1000, shuffle=True
        )
        for head, rel, tail in loader:
            ...
    """

    def __init__(
        self,
        head_index: mx.array,
        rel_type: mx.array,
        tail_index: mx.array,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
    ):
        if not (head_index.size == rel_type.size == tail_index.size):
            raise ValueError("head_index, rel_type and tail_index differ in size")
        self.head_index = head_index.reshape(-1)
        self.rel_type = rel_type.reshape(-1)
        self.tail_index = tail_index.reshape(-1)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __len__(self) -> int:
        n = self.head_index.size
        if self.drop_last:
            return n // self.batch_size
        return math.ceil(n / self.batch_size)

    def __iter__(self) -> Iterator[Triplets]:
        n = self.head_index.size
        if self.shuffle:
            perm = mx.random.permutation(n)
        else:
            perm = mx.arange(n)
        for i in range(len(self)):
            index = perm[i * self.batch_size : (i + 1) * self.batch_size]
            yield self.head_index[index], self.rel_type[index], self.tail_index[index]


class KGEModel(nn.Module):
    """Base class of the knowledge graph embedding models.

    Subclasses implement the scoring function :meth:`__call__` (higher is
    more plausible) and the training objective :meth:`loss`. Scoring
    functions broadcast over their inputs, so :meth:`test` scores many
    candidate tails at once.

    Args:
        num_nodes: The number of nodes (entities) in the graph.
        num_relations: The number of relations in the graph.
        hidden_features_dim: The size of the embeddings.
    """

    def __init__(self, num_nodes: int, num_relations: int, hidden_features_dim: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.hidden_features_dim = hidden_features_dim
        self.node_emb = nn.Embedding(num_nodes, hidden_features_dim)
        self.rel_emb = nn.Embedding(num_relations, hidden_features_dim)

    def __call__(
        self, head_index: mx.array, rel_type: mx.array, tail_index: mx.array
    ) -> mx.array:
        """Returns the score of the given triplets.

        Args:
            head_index: The head indices
            rel_type: The relation types
            tail_index: The tail indices

        Returns:
            The scores, with the broadcast shape of the inputs.
        """
        raise NotImplementedError

    def loss(
        self, head_index: mx.array, rel_type: mx.array, tail_index: mx.array
    ) -> mx.array:
        """Returns the loss of the given (positive) triplets, against negative
        triplets drawn with :meth:`random_sample`.

        Args:
            head_index: The head indices
            rel_type: The relation types
            tail_index: The tail indices
        """
        raise NotImplementedError

    def loader(
        self,
        head_index: mx.array,
        rel_type: mx.array,
        tail_index: mx.array,
        **kwargs,
    ) -> KGTripletLoader:
        """Returns a mini-batch loader of triplets.

        Args:
            head_index: The head indices
            rel_type: The relation types
            tail_index: The tail indices
            **kwargs: Arguments of :class:`KGTripletLoader`: ``batch_size``,
                ``shuffle`` and ``drop_last``.
        """
        return KGTripletLoader(head_index, rel_type, tail_index, **kwargs)

    def test(
        self,
        head_index: mx.array,
        rel_type: mx.array,
        tail_index: mx.array,
        batch_size: int,
        k: int = 10,
        log: bool = True,
    ) -> Tuple[float, float, float]:
        """Evaluates the model with the Mean Rank, the MRR and the Hits@k of
        the true tail among all the possible tail entities.

        The rank of a triplet is the position (0 for the best) of the true
        tail among all entities sorted by decreasing score, as in PyG
        (``argsort(descending=True)``). Entities whose score ties with the
        true tail's are ordered by index, as a stable sort would: PyG's
        unstable sort orders them arbitrarily, so its ranks may differ from
        these when scores tie.

        Args:
            head_index: The head indices
            rel_type: The relation types
            tail_index: The tail indices
            batch_size: Maximum number of scores computed at once. Triplets are
                evaluated by chunks of ``max(1, batch_size // num_nodes)``.
            k: The :math:`k` of Hits@:math:`k`. Default: ``10``
            log: Whether to show a progress bar. Default: ``True``

        Returns:
            A tuple ``(mean_rank, mrr, hits_at_k)``.
        """
        head_index = head_index.reshape(-1)
        rel_type = rel_type.reshape(-1)
        tail_index = tail_index.reshape(-1)
        num_triplets = head_index.size
        chunk = max(1, batch_size // self.num_nodes)
        candidates = mx.arange(self.num_nodes)[None, :]

        ranks = []
        starts = range(0, num_triplets, chunk)
        for start in tqdm(starts) if log else starts:
            h = head_index[start : start + chunk, None]
            r = rel_type[start : start + chunk, None]
            t = tail_index[start : start + chunk, None]
            scores = mx.stop_gradient(self(h, r, candidates))  # [chunk, num_nodes]
            true_scores = mx.take_along_axis(scores, t, axis=1)
            # position of the true tail in a stable descending argsort
            ahead = (scores > true_scores) | (
                (scores == true_scores) & (candidates < t)
            )
            rank = ahead.sum(axis=1)
            mx.eval(rank)
            ranks.append(rank)

        ranks = mx.concatenate(ranks).astype(mx.float32)
        mean_rank = float(ranks.mean().item())
        mrr = float((1.0 / (ranks + 1)).mean().item())
        hits_at_k = float((ranks < k).astype(mx.float32).mean().item())
        return mean_rank, mrr, hits_at_k

    def random_sample(
        self, head_index: mx.array, rel_type: mx.array, tail_index: mx.array
    ) -> Triplets:
        """Samples negative triplets by replacing the head of the first half
        of the triplets, and the tail of the second half, with random nodes.

        Args:
            head_index: The head indices
            rel_type: The relation types
            tail_index: The tail indices
        """
        num_negatives = head_index.size // 2
        rnd_index = mx.random.randint(0, self.num_nodes, head_index.shape).astype(
            head_index.dtype
        )
        first_half = mx.arange(head_index.size).reshape(head_index.shape) < (
            num_negatives
        )
        head_index = mx.where(first_half, rnd_index, head_index)
        tail_index = mx.where(first_half, tail_index, rnd_index)
        return head_index, rel_type, tail_index


def _margin_ranking_loss(
    pos_score: mx.array, neg_score: mx.array, margin: float
) -> mx.array:
    return mx.maximum(0.0, margin - (pos_score - neg_score)).mean()


def _bce_loss(pos_score: mx.array, neg_score: mx.array) -> mx.array:
    scores = mx.concatenate([pos_score, neg_score])
    target = mx.concatenate([mx.ones_like(pos_score), mx.zeros_like(neg_score)])
    return nn.losses.binary_cross_entropy(scores, target, with_logits=True)


class TransE(KGEModel):
    r"""The TransE model from the `"Translating Embeddings for Modeling
    Multi-Relational Data" <https://proceedings.neurips.cc/paper/2013/file/
    1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf>`_ paper.

    Relations are translations from head to tail entities,
    :math:`\mathbf{e}_h + \mathbf{e}_r \approx \mathbf{e}_t`, with the score

    .. math::
        d(h, r, t) = - {\| \mathbf{e}_h + \mathbf{e}_r - \mathbf{e}_t \|}_p

    where the entity embeddings are normalized to unit :math:`p`-norm. It is
    trained with a margin ranking loss.

    Args:
        num_nodes: The number of nodes (entities) in the graph.
        num_relations: The number of relations in the graph.
        hidden_features_dim: The size of the embeddings.
        margin: The margin of the ranking loss. Default: ``1.0``
        p_norm: The order of the embedding and distance norms. Default:
            ``1.0``

    Example:

    .. code-block:: python

        import mlx.nn as nn
        import mlx.optimizers as optim
        from mlx_graphs.nn import TransE

        model = TransE(num_nodes, num_relations, hidden_features_dim=50)
        optimizer = optim.Adam(learning_rate=0.01)
        loss_and_grad = nn.value_and_grad(model, lambda m, h, r, t: m.loss(h, r, t))

        for head, rel, tail in model.loader(
            head_index, rel_type, tail_index, batch_size=1000, shuffle=True
        ):
            loss, grads = loss_and_grad(model, head, rel, tail)
            optimizer.update(model, grads)

        mean_rank, mrr, hits_at_10 = model.test(
            test_head, test_rel, test_tail, batch_size=20000, k=10
        )

    References:
        [1] `Bordes et al. Translating Embeddings for Modeling Multi-Relational
        Data <https://proceedings.neurips.cc/paper/2013/file/
        1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf>`_
    """

    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        hidden_features_dim: int,
        margin: float = 1.0,
        p_norm: float = 1.0,
    ):
        super().__init__(num_nodes, num_relations, hidden_features_dim)
        self.p_norm = p_norm
        self.margin = margin
        bound = 6.0 / math.sqrt(hidden_features_dim)
        self.node_emb.weight = mx.random.uniform(
            -bound, bound, self.node_emb.weight.shape
        )
        self.rel_emb.weight = _normalize(
            mx.random.uniform(-bound, bound, self.rel_emb.weight.shape), p_norm
        )

    def __call__(
        self, head_index: mx.array, rel_type: mx.array, tail_index: mx.array
    ) -> mx.array:
        head = _normalize(self.node_emb(head_index), self.p_norm)
        rel = self.rel_emb(rel_type)
        tail = _normalize(self.node_emb(tail_index), self.p_norm)
        return -_p_norm(head + rel - tail, self.p_norm)

    def loss(
        self, head_index: mx.array, rel_type: mx.array, tail_index: mx.array
    ) -> mx.array:
        pos_score = self(head_index, rel_type, tail_index)
        neg_score = self(*self.random_sample(head_index, rel_type, tail_index))
        return _margin_ranking_loss(pos_score, neg_score, self.margin)


class DistMult(KGEModel):
    r"""The DistMult model from the `"Embedding Entities and Relations for
    Learning and Inference in Knowledge Bases" <https://arxiv.org/abs/1412.6575>`_
    paper.

    Relations are diagonal matrices, which gives the score

    .. math::
        d(h, r, t) = \langle \mathbf{e}_h, \mathbf{e}_r, \mathbf{e}_t \rangle

    It is trained with a margin ranking loss.

    Args:
        num_nodes: The number of nodes (entities) in the graph.
        num_relations: The number of relations in the graph.
        hidden_features_dim: The size of the embeddings.
        margin: The margin of the ranking loss. Default: ``1.0``

    References:
        [1] `Yang et al. Embedding Entities and Relations for Learning and
        Inference in Knowledge Bases <https://arxiv.org/abs/1412.6575>`_
    """

    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        hidden_features_dim: int,
        margin: float = 1.0,
    ):
        super().__init__(num_nodes, num_relations, hidden_features_dim)
        self.margin = margin
        self.node_emb = _glorot_embedding(num_nodes, hidden_features_dim)
        self.rel_emb = _glorot_embedding(num_relations, hidden_features_dim)

    def __call__(
        self, head_index: mx.array, rel_type: mx.array, tail_index: mx.array
    ) -> mx.array:
        head = self.node_emb(head_index)
        rel = self.rel_emb(rel_type)
        tail = self.node_emb(tail_index)
        return (head * rel * tail).sum(axis=-1)

    def loss(
        self, head_index: mx.array, rel_type: mx.array, tail_index: mx.array
    ) -> mx.array:
        pos_score = self(head_index, rel_type, tail_index)
        neg_score = self(*self.random_sample(head_index, rel_type, tail_index))
        return _margin_ranking_loss(pos_score, neg_score, self.margin)


class ComplEx(KGEModel):
    r"""The ComplEx model from the `"Complex Embeddings for Simple Link
    Prediction" <https://arxiv.org/abs/1606.06357>`_ paper.

    Entities and relations have complex embeddings, and the score is the real
    part of the Hermitian triple product

    .. math::
        d(h, r, t) = \textrm{Re}(\langle \mathbf{e}_h, \mathbf{e}_r,
        \overline{\mathbf{e}_t} \rangle)

    It is trained with a binary cross entropy loss.

    Args:
        num_nodes: The number of nodes (entities) in the graph.
        num_relations: The number of relations in the graph.
        hidden_features_dim: The size of the embeddings (of both their real
            and imaginary parts).

    References:
        [1] `Trouillon et al. Complex Embeddings for Simple Link Prediction
        <https://arxiv.org/abs/1606.06357>`_
    """

    def __init__(self, num_nodes: int, num_relations: int, hidden_features_dim: int):
        super().__init__(num_nodes, num_relations, hidden_features_dim)
        self.node_emb = _glorot_embedding(num_nodes, hidden_features_dim)
        self.node_emb_im = _glorot_embedding(num_nodes, hidden_features_dim)
        self.rel_emb = _glorot_embedding(num_relations, hidden_features_dim)
        self.rel_emb_im = _glorot_embedding(num_relations, hidden_features_dim)

    def __call__(
        self, head_index: mx.array, rel_type: mx.array, tail_index: mx.array
    ) -> mx.array:
        head_re, head_im = self.node_emb(head_index), self.node_emb_im(head_index)
        rel_re, rel_im = self.rel_emb(rel_type), self.rel_emb_im(rel_type)
        tail_re, tail_im = self.node_emb(tail_index), self.node_emb_im(tail_index)
        return (
            (head_re * rel_re * tail_re).sum(axis=-1)
            + (head_im * rel_re * tail_im).sum(axis=-1)
            + (head_re * rel_im * tail_im).sum(axis=-1)
            - (head_im * rel_im * tail_re).sum(axis=-1)
        )

    def loss(
        self, head_index: mx.array, rel_type: mx.array, tail_index: mx.array
    ) -> mx.array:
        pos_score = self(head_index, rel_type, tail_index)
        neg_score = self(*self.random_sample(head_index, rel_type, tail_index))
        return _bce_loss(pos_score, neg_score)


class RotatE(KGEModel):
    r"""The RotatE model from the `"RotatE: Knowledge Graph Embedding by
    Relational Rotation in Complex Space" <https://arxiv.org/abs/1902.10197>`_
    paper.

    Relations are rotations in the complex plane,
    :math:`\mathbf{e}_t = \mathbf{e}_h \circ \mathbf{e}_r` with
    :math:`|\mathbf{e}_r| = 1`, which gives the score

    .. math::
        d(h, r, t) = \gamma - {\| \mathbf{e}_h \circ \mathbf{e}_r -
        \mathbf{e}_t \|}

    where :math:`\gamma` is the margin. It is trained with a binary cross
    entropy loss.

    Args:
        num_nodes: The number of nodes (entities) in the graph.
        num_relations: The number of relations in the graph.
        hidden_features_dim: The size of the embeddings (of both their real
            and imaginary parts).
        margin: The margin :math:`\gamma`. Default: ``1.0``

    References:
        [1] `Sun et al. RotatE: Knowledge Graph Embedding by Relational
        Rotation in Complex Space <https://arxiv.org/abs/1902.10197>`_
    """

    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        hidden_features_dim: int,
        margin: float = 1.0,
    ):
        super().__init__(num_nodes, num_relations, hidden_features_dim)
        self.margin = margin
        self.node_emb = _glorot_embedding(num_nodes, hidden_features_dim)
        self.node_emb_im = _glorot_embedding(num_nodes, hidden_features_dim)
        # relation phases
        self.rel_emb.weight = mx.random.uniform(
            0, 2 * math.pi, self.rel_emb.weight.shape
        )

    def __call__(
        self, head_index: mx.array, rel_type: mx.array, tail_index: mx.array
    ) -> mx.array:
        head_re, head_im = self.node_emb(head_index), self.node_emb_im(head_index)
        tail_re, tail_im = self.node_emb(tail_index), self.node_emb_im(tail_index)
        rel_theta = self.rel_emb(rel_type)
        rel_re, rel_im = mx.cos(rel_theta), mx.sin(rel_theta)
        re_score = (rel_re * head_re - rel_im * head_im) - tail_re
        im_score = (rel_re * head_im + rel_im * head_re) - tail_im
        score = mx.sqrt((re_score**2 + im_score**2).sum(axis=-1))
        return self.margin - score

    def loss(
        self, head_index: mx.array, rel_type: mx.array, tail_index: mx.array
    ) -> mx.array:
        pos_score = self(head_index, rel_type, tail_index)
        neg_score = self(*self.random_sample(head_index, rel_type, tail_index))
        return _bce_loss(pos_score, neg_score)
