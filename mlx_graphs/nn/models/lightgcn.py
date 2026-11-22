from typing import Any, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.nn.conv._norm import cached_gcn_norm
from mlx_graphs.nn.conv.simple_conv import SimpleConv


class LightGCN(nn.Module):
    r"""The LightGCN model from the `"LightGCN: Simplifying and Powering Graph
    Convolution Network for Recommendation" <https://arxiv.org/abs/2002.02126>`_
    paper.

    LightGCN learns node embeddings by linearly propagating them on the graph,
    and uses the weighted sum of the embeddings of all layers as the final
    embedding

    .. math::
        \mathbf{x}_i = \sum_{l=0}^{L} \alpha_l \mathbf{x}^{(l)}_i, \qquad
        \mathbf{x}^{(l+1)}_i = \sum_{j \in \mathcal{N}(i)}
        \frac{1}{\sqrt{\deg(i)\deg(j)}}\mathbf{x}^{(l)}_j.

    Two prediction heads and training objectives are provided: link
    prediction (:meth:`link_pred_loss` and :meth:`predict_link`) and
    recommendation (:meth:`recommendation_loss` and :meth:`recommend`).

    Embeddings are propagated along ``edge_index``, while scores are computed
    for the node pairs of ``edge_label_index``.

    Args:
        num_nodes: The number of nodes in the graph.
        embedding_dim: The size of the node embeddings.
        num_layers: The number of propagation layers.
        alpha: The weight of each layer's embedding in the final embedding: a
            scalar used for every layer, or an array of shape
            `[num_layers + 1]`. Not trained. Default: ``None``
            (``1 / (num_layers + 1)``)
        normalize: Whether to apply the symmetric normalization
            :math:`1 / \sqrt{\deg(i)\deg(j)}` when propagating. Default:
            ``True``

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import LightGCN

        # users are nodes 0..num_users-1, items the following nodes
        model = LightGCN(num_users + num_items, embedding_dim=64, num_layers=3)


        def loss_fn(model, edge_index, pos_edges, neg_edges):
            label_index = mx.concatenate([pos_edges, neg_edges], axis=1)
            rank = model(edge_index, label_index)
            pos_rank, neg_rank = mx.split(rank, 2)
            return model.recommendation_loss(pos_rank, neg_rank)


        top_items = model.recommend(edge_index, src_index=users, dst_index=items, k=10)

    References:
        [1] `He et al. LightGCN: Simplifying and Powering Graph Convolution
        Network for Recommendation <https://arxiv.org/abs/2002.02126>`_
    """

    def __init__(
        self,
        num_nodes: int,
        embedding_dim: int,
        num_layers: int,
        alpha: Optional[Union[float, mx.array, List[float]]] = None,
        normalize: bool = True,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.normalize = normalize

        if alpha is None:
            alpha = 1.0 / (num_layers + 1)
        if isinstance(alpha, (int, float)):
            alpha = mx.array([float(alpha)] * (num_layers + 1))
        alpha = mx.array(alpha)
        if alpha.shape != (num_layers + 1,):
            raise ValueError(
                f"`alpha` must have shape ({num_layers + 1},), got {alpha.shape}"
            )
        self.alpha = alpha
        self.freeze(keys=["alpha"], recurse=False)

        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        self.embedding.weight = nn.init.glorot_uniform()(self.embedding.weight)
        self.conv = SimpleConv(aggr="add")

    def get_embedding(
        self, edge_index: mx.array, edge_weights: Optional[mx.array] = None
    ) -> mx.array:
        """Returns the final embedding of every node.

        Args:
            edge_index: Graph connectivity of shape `[2, num_edges]`
            edge_weights: Optional weight of each edge. Default: ``None``

        Returns:
            An array of shape `[num_nodes, embedding_dim]`.
        """
        if self.normalize:
            edge_index, edge_weights = cached_gcn_norm(
                edge_index, self.num_nodes, edge_weights, add_self_loops=False
            )
        x = self.embedding.weight
        out = x * self.alpha[0]
        for i in range(self.num_layers):
            x = self.conv(edge_index, x, edge_weights)
            out = out + x * self.alpha[i + 1]
        return out

    def __call__(
        self,
        edge_index: mx.array,
        edge_label_index: Optional[mx.array] = None,
        edge_weights: Optional[mx.array] = None,
    ) -> mx.array:
        """Computes the scores (dot products of the final embeddings) of
        pairs of nodes.

        Args:
            edge_index: Graph connectivity of shape `[2, num_edges]`
            edge_label_index: The node pairs to score, of shape
                `[2, num_pairs]`. Default: ``None`` (the edges of
                ``edge_index``)
            edge_weights: Optional weight of each edge of ``edge_index``.
                Default: ``None``

        Returns:
            One score per pair, shape `[num_pairs]`.
        """
        if edge_label_index is None:
            edge_label_index = edge_index
        out = self.get_embedding(edge_index, edge_weights)
        return (out[edge_label_index[0]] * out[edge_label_index[1]]).sum(axis=-1)

    def predict_link(
        self,
        edge_index: mx.array,
        edge_label_index: Optional[mx.array] = None,
        edge_weights: Optional[mx.array] = None,
        prob: bool = False,
    ) -> mx.array:
        """Predicts links between the node pairs of ``edge_label_index``.

        Args:
            edge_index: Graph connectivity of shape `[2, num_edges]`
            edge_label_index: The node pairs to score. Default: ``None`` (the
                edges of ``edge_index``)
            edge_weights: Optional weight of each edge. Default: ``None``
            prob: Whether to return probabilities instead of 0/1 predictions.
                Default: ``False``
        """
        pred = mx.sigmoid(self(edge_index, edge_label_index, edge_weights))
        return pred if prob else mx.round(pred)

    def recommend(
        self,
        edge_index: mx.array,
        edge_weights: Optional[mx.array] = None,
        src_index: Optional[mx.array] = None,
        dst_index: Optional[mx.array] = None,
        k: int = 1,
        sorted: bool = True,
    ) -> mx.array:
        """Returns the top-``k`` recommendations of the nodes of
        ``src_index``.

        Args:
            edge_index: Graph connectivity of shape `[2, num_edges]`
            edge_weights: Optional weight of each edge. Default: ``None``
            src_index: The nodes to recommend for. Default: ``None`` (all
                nodes)
            dst_index: The candidate recommendations. Default: ``None`` (all
                nodes)
            k: Number of recommendations. Default: ``1``
            sorted: Whether to sort the recommendations by decreasing score.
                Default: ``True``

        Returns:
            The indices of the recommended nodes, shape `[num_src, k]`.
        """
        out_src = out_dst = self.get_embedding(edge_index, edge_weights)
        if src_index is not None:
            out_src = out_src[src_index]
        if dst_index is not None:
            out_dst = out_dst[dst_index]
        pred = out_src @ out_dst.T
        if sorted:
            top_index = mx.argsort(-pred, axis=-1)[:, :k]
        else:
            top_index = mx.argpartition(-pred, kth=k - 1, axis=-1)[:, :k]
        if dst_index is not None:  # local positions to node indices
            top_index = dst_index[top_index]
        return top_index

    def link_pred_loss(
        self, pred: mx.array, edge_label: mx.array, **kwargs: Any
    ) -> mx.array:
        """Binary cross entropy loss of link predictions.

        Args:
            pred: The predicted scores (logits)
            edge_label: The ground-truth edge labels (0 or 1)
            **kwargs: Additional arguments of
                :func:`mlx.nn.losses.binary_cross_entropy` (e.g. ``weights``)
        """
        kwargs.setdefault("reduction", "mean")
        return nn.losses.binary_cross_entropy(
            pred, edge_label.astype(pred.dtype), with_logits=True, **kwargs
        )

    def recommendation_loss(
        self,
        pos_edge_rank: mx.array,
        neg_edge_rank: mx.array,
        node_id: Optional[mx.array] = None,
        lambda_reg: float = 1e-4,
    ) -> mx.array:
        r"""Bayesian Personalized Ranking (BPR) loss.

        .. math::
            L = -\frac{1}{|P|} \sum_{(u, i, j)} \ln \sigma(\hat{y}_{ui} -
            \hat{y}_{uj}) + \frac{\lambda}{|P|} \|\mathbf{x}^{(0)}\|^2

        The ``i``-th entries of ``pos_edge_rank`` and ``neg_edge_rank`` must be
        the scores of a positive and a negative edge of the same node (e.g.
        user).

        Args:
            pos_edge_rank: Scores of the positive edges, shape `[|P|]`
            neg_edge_rank: Scores of the negative edges, shape `[|P|]`
            node_id: The nodes whose initial embeddings are regularized.
                Default: ``None`` (all nodes)
            lambda_reg: The :math:`L_2` regularization strength
                :math:`\lambda`. Default: ``1e-4``
        """
        log_prob = nn.log_sigmoid(pos_edge_rank - neg_edge_rank).mean()
        loss = -log_prob
        if lambda_reg != 0:
            emb = self.embedding.weight
            emb = emb if node_id is None else emb[node_id]
            loss = loss + lambda_reg * (emb**2).sum() / pos_edge_rank.shape[0]
        return loss
