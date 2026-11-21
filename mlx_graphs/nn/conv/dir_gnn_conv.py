import copy
from typing import Any, Callable, Optional, Union

import mlx.core as mx
import mlx.nn as nn


class DirGNNConv(nn.Module):
    r"""A generic wrapper for computing graph convolution on directed graphs as
    described in the `"Edge Directionality Improves Learning on Heterophilic
    Graphs" <https://arxiv.org/abs/2305.10498>`_ paper.

    :class:`DirGNNConv` passes messages both from source nodes to destination
    nodes (along the edges) and from destination nodes to source nodes (along
    the reversed edges), with two copies of the wrapped layer, and combines
    them as a convex combination:

    .. math::
        \mathbf{x}^{\prime}_i = \alpha \, \mathrm{conv}_{\textrm{out}}
        (\mathbf{X}, \mathbf{A}^{\top})_i + (1 - \alpha) \,
        \mathrm{conv}_{\textrm{in}}(\mathbf{X}, \mathbf{A})_i +
        \mathbf{W} \mathbf{x}_i

    As in PyG, the wrapped layers have their self-loops and root weight turned
    off: the attributes ``add_self_loops``, ``_add_self_loops`` and
    ``root_weight`` are set to ``False`` on both copies when the layer has
    them. Layers whose root/skip term cannot be turned off this way keep it.

    Args:
        conv: The layer to wrap, which is copied twice. Since MLX layers have
            no generic way to re-initialize their parameters, both copies
            start with the same weights; pass instead a function that builds
            a new layer (e.g. ``lambda: GCNConv(16, 32)``) to initialize them
            independently, as PyG does.
        alpha: Weight of the reversed edges in the convex combination.
            Default: ``0.5``
        root_weight: Whether to add a linear projection of the node features
            to the output. Default: ``True``
        node_features_dim: Size of input node features, needed for the root
            projection when the wrapped layer has no ``node_features_dim``
            attribute. Default: ``None``
        out_features_dim: Size of output node embeddings, needed for the root
            projection when the wrapped layer has no ``out_features_dim``
            attribute. Default: ``None``

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import DirGNNConv, SAGEConv

        conv = DirGNNConv(lambda: SAGEConv(16, 32), alpha=0.5)
        edge_index = mx.array([[0, 1, 2, 3, 4], [0, 0, 1, 1, 3]])
        node_features = mx.random.uniform(low=0, high=1, shape=(5, 16))

        h = conv(edge_index, node_features)

        >>> h.shape
        (5, 32)

    References:
        `Rossi et al. Edge Directionality Improves Learning on Heterophilic
        Graphs <https://arxiv.org/abs/2305.10498>`_
    """

    def __init__(
        self,
        conv: Union[nn.Module, Callable[[], nn.Module]],
        alpha: float = 0.5,
        root_weight: bool = True,
        node_features_dim: Optional[int] = None,
        out_features_dim: Optional[int] = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.root_weight = root_weight

        if isinstance(conv, nn.Module):
            self.conv_in = copy.deepcopy(conv)
            self.conv_out = copy.deepcopy(conv)
        elif callable(conv):
            self.conv_in = conv()
            self.conv_out = conv()
        else:
            raise TypeError("conv must be an mlx.nn.Module or a function building one")

        for layer in (self.conv_in, self.conv_out):
            for name in ("add_self_loops", "_add_self_loops", "root_weight"):
                if hasattr(layer, name):
                    setattr(layer, name, False)

        if root_weight:
            in_dim = node_features_dim or getattr(
                self.conv_in, "node_features_dim", None
            )
            out_dim = out_features_dim or getattr(
                self.conv_in, "out_features_dim", None
            )
            if not isinstance(in_dim, int) or not isinstance(out_dim, int):
                raise ValueError(
                    "Cannot infer the input/output sizes of the wrapped layer: "
                    "pass `node_features_dim` and `out_features_dim`."
                )
            self.node_features_dim = in_dim
            self.out_features_dim = out_dim
            self.lin = nn.Linear(in_dim, out_dim)

    def __call__(
        self, edge_index: mx.array, node_features: mx.array, **kwargs: Any
    ) -> mx.array:
        """Computes the forward pass of DirGNNConv.

        Args:
            edge_index: Input edge index of shape `[2, num_edges]`
            node_features: Input node features
            **kwargs: Extra per-edge arguments passed to both wrapped layers
                (e.g. ``edge_weights``), in the order of ``edge_index``

        Returns:
            mx.array: The computed node embeddings
        """
        x_in = self.conv_in(edge_index, node_features, **kwargs)
        x_out = self.conv_out(edge_index[::-1], node_features, **kwargs)

        out = self.alpha * x_out + (1 - self.alpha) * x_in
        if self.root_weight:
            out = out + self.lin(node_features)
        return out
