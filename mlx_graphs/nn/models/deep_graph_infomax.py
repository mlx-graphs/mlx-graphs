import math
from typing import Any, Callable, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

EPS = 1e-15


class DeepGraphInfomax(nn.Module):
    r"""The Deep Graph Infomax model from the `"Deep Graph Infomax"
    <https://arxiv.org/abs/1809.10341>`_ paper, based on a user-defined
    encoder :math:`\mathcal{E}`, summary (readout) function
    :math:`\mathcal{R}` and corruption function :math:`\mathcal{C}`.

    The encoder is trained to maximize the mutual information between node
    embeddings and a summary of the whole graph, by telling apart the
    embeddings of the graph from those of a corrupted graph.

    Args:
        hidden_features_dim: The size of the latent node embeddings.
        encoder: The encoder module :math:`\mathcal{E}`.
        summary: The readout function :math:`\mathcal{R}`, called as
            ``summary(z, *args, **kwargs)`` with the node embeddings ``z`` and
            the inputs of the encoder.
        corruption: The corruption function :math:`\mathcal{C}`, called with
            the inputs of the encoder and returning the corrupted inputs (a
            single array or a tuple, replacing the positional then the keyword
            arguments in order).

    Example:

    .. code-block:: python

        import mlx.core as mx
        import mlx.nn as nn
        from mlx_graphs.nn import GCN, DeepGraphInfomax


        def corruption(edge_index, node_features):
            perm = mx.random.permutation(node_features.shape[0])
            return edge_index, node_features[perm]


        model = DeepGraphInfomax(
            hidden_features_dim=512,
            encoder=GCN(node_features_dim, 512, num_layers=1, act="relu"),
            summary=lambda z, *args, **kwargs: mx.sigmoid(z.mean(axis=0)),
            corruption=corruption,
        )


        def loss_fn(model, edge_index, node_features):
            pos_z, neg_z, summary = model(edge_index, node_features)
            return model.loss(pos_z, neg_z, summary)

    References:
        [1] `Veličković et al. Deep Graph Infomax
        <https://arxiv.org/abs/1809.10341>`_
    """

    def __init__(
        self,
        hidden_features_dim: int,
        encoder: nn.Module,
        summary: Callable[..., mx.array],
        corruption: Callable[..., Any],
    ):
        super().__init__()
        self.hidden_features_dim = hidden_features_dim
        self.encoder = encoder
        self.summary = summary
        self.corruption = corruption
        bound = 1.0 / math.sqrt(hidden_features_dim)
        self.weight = mx.random.uniform(
            -bound, bound, (hidden_features_dim, hidden_features_dim)
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Tuple[mx.array, ...]:
        """Encodes the inputs and their corruption, and summarizes the graph.

        Args:
            *args: Positional inputs of the encoder (e.g. ``edge_index``,
                ``node_features``)
            **kwargs: Keyword inputs of the encoder

        Returns:
            A tuple ``(pos_z, neg_z, summary)`` with the embeddings of the
            graph, those of the corrupted graph, and the graph summary.
        """
        pos_z = self.encoder(*args, **kwargs)
        cor = self.corruption(*args, **kwargs)
        cor = cor if isinstance(cor, tuple) else (cor,)
        cor_args = cor[: len(args)]
        cor_kwargs = dict(kwargs)
        for key, value in zip(kwargs.keys(), cor[len(args) :]):
            cor_kwargs[key] = value
        neg_z = self.encoder(*cor_args, **cor_kwargs)
        summary = self.summary(pos_z, *args, **kwargs)
        return pos_z, neg_z, summary

    def discriminate(
        self, z: mx.array, summary: mx.array, sigmoid: bool = True
    ) -> mx.array:
        """Scores each node embedding against the summary.

        Args:
            z: The node embeddings, shape `[num_nodes, hidden_features_dim]`
            summary: The summary vector, shape `[hidden_features_dim]`
            sigmoid: If ``False``, returns the logits. Default: ``True``

        Returns:
            One probability (or logit) per node.
        """
        summary = summary.T if summary.ndim > 1 else summary
        value = z @ (self.weight @ summary)
        return mx.sigmoid(value) if sigmoid else value

    def loss(self, pos_z: mx.array, neg_z: mx.array, summary: mx.array) -> mx.array:
        """The mutual information maximization objective (binary cross
        entropy of the discriminator)."""
        pos_loss = -mx.log(self.discriminate(pos_z, summary, sigmoid=True) + EPS).mean()
        neg_loss = -mx.log(
            1 - self.discriminate(neg_z, summary, sigmoid=True) + EPS
        ).mean()
        return pos_loss + neg_loss

    def test(
        self,
        train_z: mx.array,
        train_y: mx.array,
        test_z: mx.array,
        test_y: mx.array,
        max_iter: int = 300,
        learning_rate: float = 0.01,
        weight_decay: float = 0.0,
    ) -> float:
        """Evaluates the embeddings with a downstream logistic regression:
        trains a multinomial logistic regression on ``train_z`` and returns
        its accuracy on ``test_z``.

        The classifier is trained full-batch with Adam, in MLX.

        Args:
            train_z: Embeddings of the training nodes
            train_y: Integer labels of the training nodes
            test_z: Embeddings of the test nodes
            test_y: Integer labels of the test nodes
            max_iter: Number of optimization steps. Default: ``300``
            learning_rate: Learning rate of Adam. Default: ``0.01``
            weight_decay: :math:`L_2` penalty on the weights. Default: ``0.0``

        Returns:
            The test accuracy.
        """
        train_z = mx.stop_gradient(train_z)
        test_z = mx.stop_gradient(test_z)
        train_y = train_y.reshape(-1)
        test_y = test_y.reshape(-1)
        num_classes = int(max(train_y.max().item(), test_y.max().item())) + 1
        clf = nn.Linear(train_z.shape[-1], num_classes)
        optimizer = optim.Adam(learning_rate=learning_rate)

        def loss_fn(clf):
            logits = clf(train_z)
            loss = nn.losses.cross_entropy(logits, train_y, reduction="mean")
            if weight_decay > 0:
                loss = loss + weight_decay * (clf.weight**2).sum()
            return loss

        loss_and_grad = nn.value_and_grad(clf, loss_fn)
        for _ in range(max_iter):
            _, grads = loss_and_grad(clf)
            optimizer.update(clf, grads)
            mx.eval(clf.parameters(), optimizer.state)

        pred = mx.argmax(clf(test_z), axis=-1)
        return float(mx.mean(pred == test_y).item())
