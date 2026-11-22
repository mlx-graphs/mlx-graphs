from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_graphs.nn.conv.gcn_conv import GCNConv
from mlx_graphs.nn.linear import Linear
from mlx_graphs.utils import scatter


class RECT_L(nn.Module):
    r"""The RECT model, i.e. its supervised RECT-L part, from the `"Network
    Embedding with Completely-imbalanced Labels"
    <https://arxiv.org/abs/2007.03545>`_ paper.

    A one-layer GCN is trained to reconstruct semantic class knowledge (the
    class centers given by :meth:`get_semantic_labels`), and its hidden
    embeddings (:meth:`embed`) are used as node embeddings.

    Args:
        node_features_dim: Size of the input node features.
        hidden_features_dim: Size of the hidden node embeddings.
        normalize: Whether to add self-loops and apply the symmetric GCN
            normalization. Default: ``True``
        dropout: Dropout probability. Default: ``0.0``

    Example:

    .. code-block:: python

        import mlx.nn as nn
        from mlx_graphs.nn import RECT_L

        model = RECT_L(200, 200, normalize=False, dropout=0.0)
        target = model.get_semantic_labels(node_features, labels, train_mask)


        def loss_fn(model):
            out = model(edge_index, node_features)
            return nn.losses.mse_loss(out[train_idx], target)


        embeddings = model.embed(edge_index, node_features)

    References:
        [1] `Wang et al. Network Embedding with Completely-imbalanced Labels
        <https://arxiv.org/abs/2007.03545>`_
    """

    def __init__(
        self,
        node_features_dim: int,
        hidden_features_dim: int,
        normalize: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.node_features_dim = node_features_dim
        self.hidden_features_dim = hidden_features_dim
        self.normalize = normalize
        self.conv = GCNConv(
            node_features_dim, hidden_features_dim, add_self_loops=normalize
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        self.lin = Linear(hidden_features_dim, node_features_dim)

    def _conv(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_weights: Optional[mx.array],
    ) -> mx.array:
        return self.conv(
            edge_index, node_features, edge_weights, normalize=self.normalize
        )

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_weights: Optional[mx.array] = None,
    ) -> mx.array:
        """Reconstructs the semantic labels of the nodes.

        Args:
            edge_index: Input edge index of shape `[2, num_edges]`
            node_features: Input node features
            edge_weights: Optional weight of each edge. Default: ``None``

        Returns:
            An array of shape `[num_nodes, node_features_dim]`.
        """
        x = self._conv(edge_index, node_features, edge_weights)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.lin(x)

    def embed(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_weights: Optional[mx.array] = None,
    ) -> mx.array:
        """Returns the node embeddings (the GCN output), without gradient."""
        return mx.stop_gradient(self._conv(edge_index, node_features, edge_weights))

    def get_semantic_labels(
        self, node_features: mx.array, y: mx.array, mask: mx.array
    ) -> mx.array:
        """Replaces the labels of the masked nodes by their class centers, i.e.
        the mean features of the masked nodes of each class.

        Args:
            node_features: Input node features
            y: Integer labels of all the nodes, shape `[num_nodes]`
            mask: Boolean mask or index array of the labeled nodes

        Returns:
            The class center of each masked node, shape
            `[num_masked, node_features_dim]`, in the order of ``mask``.
        """
        y = y.reshape(-1)
        if mask.dtype == mx.bool_:
            mask = mx.array(np.nonzero(np.array(mask))[0])
        x, y = node_features[mask], y[mask]
        num_classes = int(y.max().item()) + 1
        mean = scatter(x, y, num_classes, "mean")
        return mx.stop_gradient(mean[y])
