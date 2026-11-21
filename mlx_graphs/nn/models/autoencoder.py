from typing import Any, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# The metrics of `test`, importable from here as before.
from mlx_graphs.metrics import average_precision_score, roc_auc_score  # noqa: F401
from mlx_graphs.utils.negative_sampling import negative_sampling

EPS = 1e-15
MAX_LOGSTD = 10


class InnerProductDecoder(nn.Module):
    r"""The inner product decoder from the `"Variational Graph Auto-Encoders"
    <https://arxiv.org/abs/1611.07308>`_ paper.

    .. math::
        \sigma(\mathbf{Z}\mathbf{Z}^{\top})

    where :math:`\mathbf{Z} \in \mathbb{R}^{N \times d}` denotes the latent
    space produced by the encoder.

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import InnerProductDecoder

        decoder = InnerProductDecoder()
        z = mx.random.normal((5, 16))
        edge_index = mx.array([[0, 1, 2], [1, 2, 3]])
        >>> decoder(edge_index, z).shape
        (3,)
        >>> decoder.forward_all(z).shape
        (5, 5)
    """

    def __call__(
        self, edge_index: mx.array, z: mx.array, sigmoid: bool = True
    ) -> mx.array:
        """Decodes the latent variables ``z`` into edge probabilities for the
        node pairs of ``edge_index``.

        Args:
            edge_index: Node pairs to score, of shape `[2, num_pairs]`
            z: The latent space :math:`\\mathbf{Z}`, shape `[num_nodes, dim]`
            sigmoid: If ``False``, returns the logits instead of the
                probabilities. Default: ``True``

        Returns:
            One probability (or logit) per pair, shape `[num_pairs]`.
        """
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(axis=1)
        return mx.sigmoid(value) if sigmoid else value

    def forward_all(self, z: mx.array, sigmoid: bool = True) -> mx.array:
        """Decodes the latent variables ``z`` into a dense adjacency matrix of
        probabilities.

        Args:
            z: The latent space :math:`\\mathbf{Z}`, shape `[num_nodes, dim]`
            sigmoid: If ``False``, returns the logits instead of the
                probabilities. Default: ``True``

        Returns:
            An array of shape `[num_nodes, num_nodes]`.
        """
        adj = z @ z.T
        return mx.sigmoid(adj) if sigmoid else adj


class GAE(nn.Module):
    r"""The Graph Auto-Encoder model from the `"Variational Graph
    Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_ paper, based on
    user-defined encoder and decoder models.

    The decoder is called as ``decoder(edge_index, z, sigmoid=...)``, like
    :class:`InnerProductDecoder`.

    Args:
        encoder: The encoder module, mapping its inputs (e.g. ``edge_index``
            and ``node_features``) to the latent node embeddings ``z``.
        decoder: The decoder module. Default: :class:`InnerProductDecoder`

    Example:

    .. code-block:: python

        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        from mlx_graphs.nn import GAE, GCN

        model = GAE(GCN(node_features_dim, 32, num_layers=2, out_features_dim=16))
        optimizer = optim.Adam(learning_rate=0.01)


        def loss_fn(model, edge_index, node_features):
            z = model.encode(edge_index, node_features)
            return model.recon_loss(z, edge_index)


        loss_and_grad = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grad(model, edge_index, node_features)
        optimizer.update(model, grads)

        z = model.encode(edge_index, node_features)
        auc, ap = model.test(z, test_pos_edge_index, test_neg_edge_index)

    References:
        [1] `Kipf et al. Variational Graph Auto-Encoders
        <https://arxiv.org/abs/1611.07308>`_
    """

    def __init__(self, encoder: nn.Module, decoder: Optional[nn.Module] = None):
        super().__init__()
        self.encoder = encoder
        self.decoder = InnerProductDecoder() if decoder is None else decoder

    def __call__(self, *args: Any, **kwargs: Any) -> mx.array:
        """Alias for :meth:`encode`."""
        return self.encode(*args, **kwargs)

    def encode(self, *args: Any, **kwargs: Any) -> mx.array:
        """Runs the encoder and computes the latent node embeddings."""
        return self.encoder(*args, **kwargs)

    def decode(self, *args: Any, **kwargs: Any) -> mx.array:
        """Runs the decoder and computes edge probabilities."""
        return self.decoder(*args, **kwargs)

    def recon_loss(
        self,
        z: mx.array,
        pos_edge_index: mx.array,
        neg_edge_index: Optional[mx.array] = None,
    ) -> mx.array:
        """Binary cross entropy of the reconstruction of positive edges and of
        negative edges.

        Args:
            z: The latent space :math:`\\mathbf{Z}`
            pos_edge_index: The positive edges to train against
            neg_edge_index: The negative edges to train against. If ``None``,
                as many negative edges as positive ones are sampled with
                :func:`~mlx_graphs.utils.negative_sampling`. Default: ``None``

        Returns:
            The scalar loss.
        """
        pos_loss = -mx.log(self.decoder(pos_edge_index, z, sigmoid=True) + EPS).mean()
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.shape[0])
        neg_loss = -mx.log(
            1 - self.decoder(neg_edge_index, z, sigmoid=True) + EPS
        ).mean()
        return pos_loss + neg_loss

    def test(
        self, z: mx.array, pos_edge_index: mx.array, neg_edge_index: mx.array
    ) -> Tuple[float, float]:
        """Computes the area under the ROC curve (AUC) and the average
        precision (AP) of the predictions for positive and negative edges.

        Args:
            z: The latent space :math:`\\mathbf{Z}`
            pos_edge_index: The positive edges to evaluate against
            neg_edge_index: The negative edges to evaluate against

        Returns:
            A tuple ``(auc, ap)``.
        """
        pos_pred = self.decoder(pos_edge_index, z, sigmoid=True)
        neg_pred = self.decoder(neg_edge_index, z, sigmoid=True)
        pred = np.array(mx.concatenate([pos_pred, neg_pred]))
        y = np.concatenate([np.ones(pos_pred.shape[0]), np.zeros(neg_pred.shape[0])])
        return roc_auc_score(y, pred), average_precision_score(y, pred)


class VGAE(GAE):
    r"""The Variational Graph Auto-Encoder model from the `"Variational Graph
    Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_ paper.

    The encoder returns a tuple ``(mu, logstd)``. In training mode,
    :meth:`encode` samples :math:`\mathbf{Z} = \mu + \epsilon \cdot
    \exp(\log\sigma)`; in evaluation mode it returns :math:`\mu`.

    Args:
        encoder: The encoder module computing :math:`\mu` and
            :math:`\log\sigma`.
        decoder: The decoder module. Default: :class:`InnerProductDecoder`

    Example:

    .. code-block:: python

        def loss_fn(model, edge_index, node_features):
            z = model.encode(edge_index, node_features)
            loss = model.recon_loss(z, edge_index)
            return loss + model.kl_loss() / num_nodes

    References:
        [1] `Kipf et al. Variational Graph Auto-Encoders
        <https://arxiv.org/abs/1611.07308>`_
    """

    def reparametrize(self, mu: mx.array, logstd: mx.array) -> mx.array:
        """Samples latent variables in training mode, returns ``mu`` otherwise."""
        if self.training:
            return mu + mx.random.normal(logstd.shape) * mx.exp(logstd)
        return mu

    def encode(self, *args: Any, **kwargs: Any) -> mx.array:
        """Runs the encoder and samples the latent node embeddings."""
        mu, logstd = self.encoder(*args, **kwargs)
        logstd = mx.minimum(logstd, MAX_LOGSTD)
        self._mu, self._logstd = mu, logstd
        return self.reparametrize(mu, logstd)

    def kl_loss(
        self, mu: Optional[mx.array] = None, logstd: Optional[mx.array] = None
    ) -> mx.array:
        """KL divergence between the latent distribution and a unit Gaussian.

        Args:
            mu: :math:`\\mu`. If ``None``, uses the one of the last call to
                :meth:`encode`. Default: ``None``
            logstd: :math:`\\log\\sigma`. If ``None``, uses the one of the last
                call to :meth:`encode`. Default: ``None``

        Returns:
            The scalar loss, averaged over the nodes.
        """
        mu = self._mu if mu is None else mu
        logstd = self._logstd if logstd is None else mx.minimum(logstd, MAX_LOGSTD)
        return -0.5 * mx.mean(
            mx.sum(1 + 2 * logstd - mu**2 - mx.exp(logstd) ** 2, axis=1)
        )


class ARGA(GAE):
    r"""The Adversarially Regularized Graph Auto-Encoder model from the
    `"Adversarially Regularized Graph Autoencoder for Graph Embedding"
    <https://arxiv.org/abs/1802.04407>`_ paper.

    Args:
        encoder: The encoder module.
        discriminator: The discriminator module, mapping latent embeddings to
            one logit per node.
        decoder: The decoder module. Default: :class:`InnerProductDecoder`

    References:
        [1] `Pan et al. Adversarially Regularized Graph Autoencoder for Graph
        Embedding <https://arxiv.org/abs/1802.04407>`_
    """

    def __init__(
        self,
        encoder: nn.Module,
        discriminator: nn.Module,
        decoder: Optional[nn.Module] = None,
    ):
        super().__init__(encoder, decoder)
        self.discriminator = discriminator

    def reg_loss(self, z: mx.array) -> mx.array:
        """Regularization loss of the encoder: fooling the discriminator.

        Args:
            z: The latent space :math:`\\mathbf{Z}`
        """
        real = mx.sigmoid(self.discriminator(z))
        return -mx.log(real + EPS).mean()

    def discriminator_loss(self, z: mx.array) -> mx.array:
        """Loss of the discriminator, telling Gaussian samples (real) from
        latent embeddings (fake). No gradient flows into ``z``.

        Args:
            z: The latent space :math:`\\mathbf{Z}`
        """
        real = mx.sigmoid(self.discriminator(mx.random.normal(z.shape)))
        fake = mx.sigmoid(self.discriminator(mx.stop_gradient(z)))
        real_loss = -mx.log(real + EPS).mean()
        fake_loss = -mx.log(1 - fake + EPS).mean()
        return real_loss + fake_loss


class ARGVA(ARGA):
    r"""The Adversarially Regularized Variational Graph Auto-Encoder model
    from the `"Adversarially Regularized Graph Autoencoder for Graph
    Embedding" <https://arxiv.org/abs/1802.04407>`_ paper.

    Combines :class:`ARGA` with the variational encoder of :class:`VGAE`.

    Args:
        encoder: The encoder module computing :math:`\mu` and
            :math:`\log\sigma`.
        discriminator: The discriminator module.
        decoder: The decoder module. Default: :class:`InnerProductDecoder`
    """

    reparametrize = VGAE.reparametrize
    encode = VGAE.encode
    kl_loss = VGAE.kl_loss
