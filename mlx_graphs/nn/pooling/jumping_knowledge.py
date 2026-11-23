from typing import List, Literal, Optional, get_args

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.nn.linear import Linear

JumpingKnowledgeModes = Literal["cat", "max", "lstm"]


class JumpingKnowledge(nn.Module):
    r"""Jumping Knowledge layer aggregation from `"Representation Learning on
    Graphs with Jumping Knowledge Networks" <https://arxiv.org/abs/1806.03536>`_
    paper.

    Combines the node embeddings :math:`\mathbf{x}_i^{(1)}, \ldots,
    \mathbf{x}_i^{(T)}` computed by :math:`T` successive GNN layers into a single
    embedding per node, using one of three modes.

    Concatenation (``"cat"``):

    .. math::
        \mathbf{h}_i = \mathbf{x}_i^{(1)} \, \Vert \, \ldots \, \Vert \,
        \mathbf{x}_i^{(T)}

    Feature-wise max pooling (``"max"``):

    .. math::
        \mathbf{h}_i = \max \left( \mathbf{x}_i^{(1)}, \ldots, \mathbf{x}_i^{(T)}
        \right)

    Weighted sum with attention scores :math:`\alpha_i^{(t)}` obtained from a
    bi-directional LSTM run over the layers (``"lstm"``):

    .. math::
        \mathbf{h}_i = \sum_{t=1}^T \alpha_i^{(t)} \mathbf{x}_i^{(t)}

    Args:
        mode: Aggregation mode: ``"cat"``, ``"max"`` or ``"lstm"``
        node_features_dim: Size of each layer's node embeddings. Only required
            for the ``"lstm"`` mode. Default: ``None``
        num_layers: Number of layers to aggregate. Only required for the
            ``"lstm"`` mode. Default: ``None``

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn.pooling.jumping_knowledge import JumpingKnowledge

        # embeddings of 5 nodes after each of 3 GNN layers
        xs = [mx.random.uniform(shape=(5, 16)) for _ in range(3)]

        >>> JumpingKnowledge("cat")(xs).shape
        (5, 48)
        >>> JumpingKnowledge("max")(xs).shape
        (5, 16)
        >>> JumpingKnowledge("lstm", node_features_dim=16, num_layers=3)(xs).shape
        (5, 16)
    """

    def __init__(
        self,
        mode: JumpingKnowledgeModes,
        node_features_dim: Optional[int] = None,
        num_layers: Optional[int] = None,
    ):
        super().__init__()

        mode = mode.lower()  # type: ignore[assignment]
        if mode not in get_args(JumpingKnowledgeModes):
            raise ValueError(
                f"Invalid mode {mode!r}.",
                f"Available values are {get_args(JumpingKnowledgeModes)}",
            )
        self.mode = mode

        if mode == "lstm":
            if node_features_dim is None or num_layers is None:
                raise ValueError(
                    "`node_features_dim` and `num_layers` are required for the "
                    "'lstm' mode."
                )
            self.node_features_dim = node_features_dim
            self.num_layers = num_layers

            hidden_dim = (num_layers * node_features_dim) // 2
            # Bi-directional LSTM: the second one reads the layers in reverse.
            self.lstm_forward = nn.LSTM(node_features_dim, hidden_dim)
            self.lstm_backward = nn.LSTM(node_features_dim, hidden_dim)
            self.att = Linear(2 * hidden_dim, 1)

    def __call__(self, node_features_list: List[mx.array]) -> mx.array:
        """Aggregates the layer-wise node embeddings.

        Args:
            node_features_list: Node embeddings after each layer, each of shape
                `[num_nodes, node_features_dim]` (the sizes may differ between
                layers in the ``"cat"`` mode only)

        Returns:
            mx.array: The aggregated node embeddings, of shape
            `[num_nodes, sum of the layers' sizes]` for ``"cat"`` and
            `[num_nodes, node_features_dim]` for ``"max"`` and ``"lstm"``
        """
        if self.mode == "cat":
            return mx.concatenate(node_features_list, axis=-1)

        if self.mode == "max":
            return mx.stack(node_features_list, axis=-1).max(axis=-1)

        # shape: [num_nodes, num_layers, node_features_dim]
        x = mx.stack(node_features_list, axis=1)

        hidden_forward, _ = self.lstm_forward(x)
        hidden_backward, _ = self.lstm_backward(x[:, ::-1])
        # shape: [num_nodes, num_layers, 2 * hidden_dim]
        hidden = mx.concatenate([hidden_forward, hidden_backward[:, ::-1]], axis=-1)

        alpha = self.att(hidden).squeeze(-1)  # [num_nodes, num_layers]
        alpha = mx.softmax(alpha, axis=-1)

        return (x * mx.expand_dims(alpha, -1)).sum(axis=1)
