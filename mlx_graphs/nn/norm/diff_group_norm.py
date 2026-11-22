import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.nn import Linear  # PyTorch's default initialization, as in PyG


class DiffGroupNormalization(nn.Module):
    r"""The differentiable group normalization layer from the `"Towards Deeper
    Graph Neural Networks with Differentiable Group Normalization"
    <https://arxiv.org/abs/2006.06972>`_ paper, which normalizes node features
    group-wise via a learnable soft cluster assignment.

    .. math::
        \mathbf{S} = \text{softmax} (\mathbf{X} \mathbf{W})

    where :math:`\mathbf{W} \in \mathbb{R}^{F \times G}` denotes a trainable
    weight matrix mapping each node into one of :math:`G` clusters.
    Normalization is then performed group-wise via:

    .. math::
        \mathbf{X}^{\prime} = \mathbf{X} + \lambda \sum_{i = 1}^G
        \text{BatchNorm}(\mathbf{S}[:, i] \odot \mathbf{X})

    Also available as :class:`DiffGroupNorm`, its PyG name.

    Args:
        node_features_dim: Size of the input node features :math:`F`.
        groups: The number of groups :math:`G`.
        lamda: The balancing factor :math:`\lambda` between input embeddings
            and normalized embeddings. Default: ``0.01``
        eps: A value added to the denominator for numerical stability.
            Default: ``1e-5``
        momentum: The value used for the running mean and running variance
            computation. Default: ``0.1``
        affine: If set to ``True``, the batch normalization has learnable
            affine parameters :math:`\gamma` and :math:`\beta`. Default: ``True``
        track_running_stats: If set to ``True``, the batch normalization tracks
            the running mean and variance and uses them in evaluation mode,
            otherwise it always uses batch statistics. Default: ``True``

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import DiffGroupNormalization

        norm = DiffGroupNormalization(16, groups=4)
        node_features = mx.random.normal((10, 16))

        >>> norm(node_features).shape
        (10, 16)
    """

    def __init__(
        self,
        node_features_dim: int,
        groups: int,
        lamda: float = 0.01,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        super().__init__()
        self.node_features_dim = node_features_dim
        self.groups = groups
        self.lamda = lamda

        self.lin = Linear(node_features_dim, groups, bias=False)
        self.norm = nn.BatchNorm(
            groups * node_features_dim,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def __call__(self, node_features: mx.array) -> mx.array:
        """Applies differentiable group normalization.

        Args:
            node_features: Node features of shape ``[num_nodes, node_features_dim]``.

        Returns:
            The normalized node features.
        """
        F, G = self.node_features_dim, self.groups
        s = mx.softmax(self.lin(node_features), axis=-1)  # [N, G]
        out = s[:, :, None] * node_features[:, None, :]  # [N, G, F]
        out = self.norm(out.reshape(-1, G * F)).reshape(-1, G, F).sum(axis=-2)
        return node_features + self.lamda * out

    @staticmethod
    def group_distance_ratio(
        node_features: mx.array, labels: mx.array, eps: float = 1e-5
    ) -> float:
        r"""Measures the ratio of inter-group distance over intra-group
        distance.

        .. math::
            R_{\text{Group}} = \frac{\frac{1}{(C-1)^2} \sum_{i!=j}
            \frac{1}{|\mathbf{X}_i||\mathbf{X}_j|} \sum_{\mathbf{x}_{iv}
            \in \mathbf{X}_i } \sum_{\mathbf{x}_{jv^{\prime}} \in \mathbf{X}_j}
            {\| \mathbf{x}_{iv} - \mathbf{x}_{jv^{\prime}} \|}_2 }{
            \frac{1}{C} \sum_{i} \frac{1}{{|\mathbf{X}_i|}^2}
            \sum_{\mathbf{x}_{iv}, \mathbf{x}_{iv^{\prime}} \in \mathbf{X}_i }
            {\| \mathbf{x}_{iv} - \mathbf{x}_{iv^{\prime}} \|}_2 }

        where :math:`\mathbf{X}_i` denotes the set of all nodes that belong to
        class :math:`i`, and :math:`C` denotes the total number of classes in
        ``labels``.

        Args:
            node_features: Node features of shape ``[num_nodes, num_features]``.
            labels: Class of each node, of shape ``[num_nodes]``.
            eps: A value added to the denominator for numerical stability.
                Default: ``1e-5``

        Returns:
            The group distance ratio.
        """

        def mean_distance(a: mx.array, b: mx.array) -> float:
            diff = a[:, None, :] - b[None, :, :]
            return mx.sqrt((diff * diff).sum(axis=-1)).mean().item()

        labels = np.asarray(labels).reshape(-1)
        num_classes = int(labels.max()) + 1
        numerator = 0.0
        denominator = 0.0
        for i in range(num_classes):
            inside = node_features[mx.array(np.nonzero(labels == i)[0])]
            outside = node_features[mx.array(np.nonzero(labels != i)[0])]
            numerator += mean_distance(inside, outside)
            denominator += mean_distance(inside, inside)
        numerator *= 1 / (num_classes - 1) ** 2
        denominator *= 1 / num_classes
        return numerator / (denominator + eps)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.node_features_dim}, "
            f"groups={self.groups})"
        )
