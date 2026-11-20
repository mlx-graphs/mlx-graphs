from typing import Any, Callable, List, Literal, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.nn.conv.gat_conv import GATConv
from mlx_graphs.nn.conv.gatv2_conv import GATv2Conv
from mlx_graphs.nn.conv.gcn_conv import GCNConv
from mlx_graphs.nn.conv.gin_conv import GINConv
from mlx_graphs.nn.conv.sage_conv import SAGEConv
from mlx_graphs.nn.linear import Linear
from mlx_graphs.nn.pooling.jumping_knowledge import JumpingKnowledge

Activation = Union[str, Callable[[mx.array], mx.array], None]
Normalization = Optional[Literal["batch", "layer"]]

_ACTIVATIONS = {
    "relu": nn.relu,
    "elu": nn.elu,
    "gelu": nn.gelu,
    "leaky_relu": nn.leaky_relu,
    "silu": nn.silu,
    "tanh": mx.tanh,
}


def _resolve_activation(act: Activation) -> Optional[Callable[[mx.array], mx.array]]:
    if act is None or callable(act):
        return act
    if act not in _ACTIVATIONS:
        raise ValueError(
            f"Unknown activation '{act}'. Available: {sorted(_ACTIVATIONS)}, "
            "or pass a callable."
        )
    return _ACTIVATIONS[act]


def _make_norm(norm: Normalization, dim: int) -> Optional[nn.Module]:
    if norm is None:
        return None
    if norm == "batch":
        return nn.BatchNorm(dim)
    if norm == "layer":
        return nn.LayerNorm(dim)
    raise ValueError(f"Unknown normalization '{norm}'. Available: 'batch', 'layer'.")


class MLP(nn.Module):
    """Multi-layer perceptron.

    Args:
        dims: Sizes of the input, of every hidden layer and of the output,
            e.g. ``[16, 64, 64, 2]`` for two hidden layers of size 64.
        dropout: Dropout probability applied after each hidden layer.
            Default: ``0.0``
        act: Activation applied after each hidden layer: a name (``"relu"``,
            ``"elu"``, ``"gelu"``, ``"leaky_relu"``, ``"silu"``, ``"tanh"``), a
            callable, or ``None``. Default: ``"relu"``
        norm: Normalization applied after each hidden layer: ``"batch"``,
            ``"layer"`` or ``None``. Default: ``None``
        plain_last: If ``True`` (default), the last layer is a plain linear
            layer; otherwise normalization, activation and dropout are applied
            to it as well.
        bias: Whether the linear layers have a bias. Default: ``True``

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import MLP

        mlp = MLP([16, 64, 64, 2], dropout=0.5, norm="batch")
        >>> mlp(mx.random.normal((10, 16))).shape
        (10, 2)
    """

    def __init__(
        self,
        dims: List[int],
        dropout: float = 0.0,
        act: Activation = "relu",
        norm: Normalization = None,
        plain_last: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        if len(dims) < 2:
            raise ValueError("`dims` needs at least an input and an output size")

        self.dims = list(dims)
        self.plain_last = plain_last
        self.act = _resolve_activation(act)
        self.lins = [
            nn.Linear(in_dim, out_dim, bias=bias)  # PyTorch's init, as in PyG
            for in_dim, out_dim in zip(dims[:-1], dims[1:])
        ]
        num_post = len(self.lins) - 1 if plain_last else len(self.lins)
        if norm is not None:
            self.norms = [_make_norm(norm, dim) for dim in dims[1 : 1 + num_post]]
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    def __call__(self, x: mx.array) -> mx.array:
        num_layers = len(self.lins)
        for i, lin in enumerate(self.lins):
            x = lin(x)
            if self.plain_last and i == num_layers - 1:
                break
            if "norms" in self:
                x = self.norms[i](x)
            if self.act is not None:
                x = self.act(x)
            if self.dropout is not None:
                x = self.dropout(x)
        return x


class BasicGNN(nn.Module):
    """Base class of the ready-to-use GNN models.

    Stacks ``num_layers`` convolution layers with an activation, an optional
    normalization and dropout in between. Subclasses only choose the
    convolution layer by implementing :meth:`init_conv`.

    Args:
        node_features_dim: Size of the input node features.
        hidden_features_dim: Size of the hidden node embeddings.
        num_layers: Number of message passing layers.
        out_features_dim: Size of the output. If ``None``, the model outputs
            the hidden embeddings of the last layer. Default: ``None``
        dropout: Dropout probability. Default: ``0.0``
        act: Activation between layers: a name (``"relu"``, ``"elu"``,
            ``"gelu"``, ``"leaky_relu"``, ``"silu"``, ``"tanh"``), a callable
            or ``None``. Default: ``"relu"``
        norm: Normalization between layers: ``"batch"``, ``"layer"`` or
            ``None``. Default: ``None``
        jk: Jumping Knowledge mode combining the embeddings of all layers:
            ``"cat"``, ``"max"``, ``"lstm"`` or ``None`` (only the last layer
            is used). With Jumping Knowledge, the output is produced by a final
            linear layer. Default: ``None``
        **kwargs: Additional arguments of the convolution layer.
    """

    #: Whether the convolution layer accepts ``edge_weights``.
    supports_edge_weights: bool = False
    #: Whether the convolution layer accepts ``edge_features``.
    supports_edge_features: bool = False

    def __init__(
        self,
        node_features_dim: int,
        hidden_features_dim: int,
        num_layers: int,
        out_features_dim: Optional[int] = None,
        dropout: float = 0.0,
        act: Activation = "relu",
        norm: Normalization = None,
        jk: Optional[Literal["cat", "max", "lstm"]] = None,
        **kwargs: Any,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("`num_layers` must be at least 1")

        self.node_features_dim = node_features_dim
        self.hidden_features_dim = hidden_features_dim
        self.num_layers = num_layers
        self.out_features_dim = (
            out_features_dim if out_features_dim is not None else hidden_features_dim
        )
        self.act = _resolve_activation(act)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

        # Without Jumping Knowledge the last convolution maps directly to the
        # output size; with it, every convolution outputs hidden embeddings and
        # a final linear layer produces the output.
        conv_out_last = hidden_features_dim
        if out_features_dim is not None and jk is None:
            conv_out_last = out_features_dim
        self._plain_last = out_features_dim is not None and jk is None

        dims = [node_features_dim] + [hidden_features_dim] * (num_layers - 1)
        dims.append(conv_out_last)
        self.convs = [
            self.init_conv(in_dim, out_dim, is_last=(i == num_layers - 1), **kwargs)
            for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:]))
        ]

        num_post = num_layers - 1 if self._plain_last else num_layers
        if norm is not None:
            self.norms = [
                _make_norm(norm, hidden_features_dim) for _ in range(num_post)
            ]

        if jk is not None:
            self.jk = JumpingKnowledge(jk, hidden_features_dim, num_layers)
            jk_dim = hidden_features_dim * (num_layers if jk == "cat" else 1)
            if out_features_dim is not None or jk == "cat":
                self.lin = Linear(jk_dim, self.out_features_dim)

    def init_conv(
        self, node_features_dim: int, out_features_dim: int, is_last: bool, **kwargs
    ) -> nn.Module:
        """Builds one convolution layer. Implemented by subclasses."""
        raise NotImplementedError

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_weights: Optional[mx.array] = None,
        edge_features: Optional[mx.array] = None,
    ) -> mx.array:
        """Computes the node embeddings (or predictions).

        Args:
            edge_index: Input edge index of shape `[2, num_edges]`
            node_features: Input node features
            edge_weights: Optional weight of each edge, for models whose
                convolution supports it.
            edge_features: Optional features of each edge, for models whose
                convolution supports it.

        Returns:
            An array of shape ``[num_nodes, out_features_dim]``.
        """
        conv_kwargs = {}
        if edge_weights is not None:
            if not self.supports_edge_weights:
                raise ValueError(
                    f"{type(self).__name__} does not support `edge_weights`"
                )
            conv_kwargs["edge_weights"] = edge_weights
        if edge_features is not None:
            if not self.supports_edge_features:
                raise ValueError(
                    f"{type(self).__name__} does not support `edge_features`"
                )
            conv_kwargs["edge_features"] = edge_features

        x = node_features
        xs = []
        for i, conv in enumerate(self.convs):
            x = conv(edge_index, x, **conv_kwargs)
            if not (self._plain_last and i == self.num_layers - 1):
                if "norms" in self:
                    x = self.norms[i](x)
                if self.act is not None:
                    x = self.act(x)
                if self.dropout is not None:
                    x = self.dropout(x)
            xs.append(x)

        if "jk" in self:
            x = self.jk(xs)
        if "lin" in self:
            x = self.lin(x)
        return x


class GCN(BasicGNN):
    """Graph Convolutional Network from `"Semi-Supervised Classification with
    Graph Convolutional Networks" <https://arxiv.org/abs/1609.02907>`_, a stack
    of :class:`~mlx_graphs.nn.GCNConv` layers.

    Self-loops are added by default, as in the paper; pass
    ``add_self_loops=False`` to disable them.

    Args:
        node_features_dim: Size of the input node features.
        hidden_features_dim: Size of the hidden node embeddings.
        num_layers: Number of message passing layers.
        out_features_dim: Size of the output. Default: ``None``
        dropout: Dropout probability. Default: ``0.0``
        act: Activation between layers. Default: ``"relu"``
        norm: Normalization between layers. Default: ``None``
        jk: Jumping Knowledge mode. Default: ``None``
        **kwargs: Additional arguments of :class:`~mlx_graphs.nn.GCNConv`.

    Example:

    .. code-block:: python

        from mlx_graphs.nn import GCN

        model = GCN(
            dataset.num_node_features,
            64,
            num_layers=2,
            out_features_dim=dataset.num_node_classes,
            dropout=0.5,
        )
        logits = model(graph.edge_index, graph.node_features)
    """

    supports_edge_weights = True

    def init_conv(self, node_features_dim, out_features_dim, is_last, **kwargs):
        kwargs.setdefault("add_self_loops", True)
        return GCNConv(node_features_dim, out_features_dim, **kwargs)


class GraphSAGE(BasicGNN):
    """GraphSAGE from `"Inductive Representation Learning on Large Graphs"
    <https://arxiv.org/abs/1706.02216>`_, a stack of
    :class:`~mlx_graphs.nn.SAGEConv` layers.

    Args:
        node_features_dim: Size of the input node features.
        hidden_features_dim: Size of the hidden node embeddings.
        num_layers: Number of message passing layers.
        out_features_dim: Size of the output. Default: ``None``
        dropout: Dropout probability. Default: ``0.0``
        act: Activation between layers. Default: ``"relu"``
        norm: Normalization between layers. Default: ``None``
        jk: Jumping Knowledge mode. Default: ``None``
        **kwargs: Additional arguments of :class:`~mlx_graphs.nn.SAGEConv`
            (e.g. ``aggr="max"``).
    """

    supports_edge_weights = True

    def init_conv(self, node_features_dim, out_features_dim, is_last, **kwargs):
        return SAGEConv(node_features_dim, out_features_dim, **kwargs)


class GIN(BasicGNN):
    """Graph Isomorphism Network from `"How Powerful are Graph Neural
    Networks?" <https://arxiv.org/abs/1810.00826>`_, a stack of
    :class:`~mlx_graphs.nn.GINConv` layers, each with a two-layer MLP.

    Args:
        node_features_dim: Size of the input node features.
        hidden_features_dim: Size of the hidden node embeddings.
        num_layers: Number of message passing layers.
        out_features_dim: Size of the output. Default: ``None``
        dropout: Dropout probability. Default: ``0.0``
        act: Activation between layers. Default: ``"relu"``
        norm: Normalization between layers. Default: ``None``
        jk: Jumping Knowledge mode. Default: ``None``
        **kwargs: Additional arguments of :class:`~mlx_graphs.nn.GINConv`
            (e.g. ``learn_eps=True``).
    """

    supports_edge_weights = True

    def init_conv(self, node_features_dim, out_features_dim, is_last, **kwargs):
        mlp = MLP([node_features_dim, out_features_dim, out_features_dim])
        return GINConv(mlp, **kwargs)


class GAT(BasicGNN):
    """Graph Attention Network from `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ (or, with ``v2=True``, `"How Attentive
    are Graph Attention Networks?" <https://arxiv.org/abs/2105.14491>`_), a
    stack of :class:`~mlx_graphs.nn.GATConv` / :class:`~mlx_graphs.nn.GATv2Conv`
    layers.

    Hidden layers concatenate their ``heads`` attention heads, each of size
    ``hidden_features_dim // heads``. When ``out_features_dim`` is given, the
    last layer averages its heads instead.

    Args:
        node_features_dim: Size of the input node features.
        hidden_features_dim: Size of the hidden node embeddings (all heads
            together). Must be divisible by ``heads``.
        num_layers: Number of message passing layers.
        out_features_dim: Size of the output. Default: ``None``
        heads: Number of attention heads. Default: ``1``
        v2: Whether to use :class:`~mlx_graphs.nn.GATv2Conv`. Default: ``False``
        dropout: Dropout probability. Default: ``0.0``
        act: Activation between layers. Default: ``"relu"``
        norm: Normalization between layers. Default: ``None``
        jk: Jumping Knowledge mode. Default: ``None``
        **kwargs: Additional arguments of the convolution layer
            (e.g. ``edge_features_dim``).
    """

    supports_edge_features = True

    def __init__(
        self,
        node_features_dim: int,
        hidden_features_dim: int,
        num_layers: int,
        out_features_dim: Optional[int] = None,
        heads: int = 1,
        v2: bool = False,
        **kwargs: Any,
    ):
        if hidden_features_dim % heads != 0:
            raise ValueError(
                "`hidden_features_dim` must be divisible by `heads` "
                f"(got {hidden_features_dim} and {heads})"
            )
        self._heads = heads
        self._v2 = v2
        super().__init__(
            node_features_dim,
            hidden_features_dim,
            num_layers,
            out_features_dim,
            **kwargs,
        )

    def init_conv(self, node_features_dim, out_features_dim, is_last, **kwargs):
        conv = GATv2Conv if self._v2 else GATConv
        if is_last and self._plain_last:
            # output layer: average the heads
            return conv(
                node_features_dim,
                out_features_dim,
                heads=self._heads,
                concat=False,
                **kwargs,
            )
        return conv(
            node_features_dim,
            out_features_dim // self._heads,
            heads=self._heads,
            concat=True,
            **kwargs,
        )
