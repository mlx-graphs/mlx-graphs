from typing import Callable, Iterable, List, Optional, Sequence, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.nn import Linear  # PyTorch's default initialization, as in PyG

from mlx_graphs.nn.aggr.scaler import DegreeScalerAggregation
from mlx_graphs.nn.message_passing import MessagePassing


def _resolve_act(act: Union[str, Callable, None]) -> Optional[Callable]:
    if act is None or callable(act):
        return act
    fn = getattr(nn, act, None)
    if fn is None or not callable(fn):
        raise ValueError(f"Unknown activation '{act}'")
    return fn


class PNAConv(MessagePassing):
    r"""Principal Neighbourhood Aggregation graph convolution from
    `"Principal Neighbourhood Aggregation for Graph Nets"
    <https://arxiv.org/abs/2004.05718>`_ paper.

    .. math::
        \mathbf{x}_i^{\prime} = \gamma_{\mathbf{\Theta}} \left(
        \mathbf{x}_i, \underset{j \in \mathcal{N}(i)}{\bigoplus}
        h_{\mathbf{\Theta}} \left( \mathbf{x}_i, \mathbf{x}_j \right)
        \right)

    with

    .. math::
        \bigoplus = \underbrace{\begin{bmatrix}
            1 \\
            S(\mathbf{D}, \alpha=1) \\
            S(\mathbf{D}, \alpha=-1)
        \end{bmatrix} }_{\text{scalers}}
        \otimes \underbrace{\begin{bmatrix}
            \mu \\
            \sigma \\
            \max \\
            \min
        \end{bmatrix}}_{\text{aggregators}},

    where :math:`h_{\mathbf{\Theta}}` (``pre_layers`` linear layers) and
    :math:`\gamma_{\mathbf{\Theta}}` (``post_layers`` linear layers) are MLPs,
    applied separately in each of the ``towers``, followed by a final linear
    layer. The aggregation is a
    :class:`~mlx_graphs.nn.aggr.DegreeScalerAggregation`.

    Args:
        node_features_dim: Size of the input node features.
        out_features_dim: Size of the output node embeddings. Must be
            divisible by ``towers``.
        aggregators: Aggregations, among ``"sum"``, ``"mean"``, ``"min"``,
            ``"max"``, ``"var"`` and ``"std"`` (any name accepted by
            :func:`~mlx_graphs.nn.aggr.aggregation_resolver` works).
        scalers: Scalers, among ``"identity"``, ``"amplification"``,
            ``"attenuation"``, ``"linear"`` and ``"inverse_linear"``.
        deg: Histogram of the in-degrees of the training nodes, see
            :meth:`get_degree_histogram`.
        edge_features_dim: Size of the edge features, if any. Default: ``None``
        towers: Number of towers. Default: ``1``
        pre_layers: Number of layers of the message MLP. Default: ``1``
        post_layers: Number of layers of the update MLP. Default: ``1``
        divide_input: If ``True``, the input features are split between the
            towers (``node_features_dim`` must be divisible by ``towers``);
            otherwise every tower sees all of them. Default: ``False``
        act: Activation of the pre- and post-MLPs: a name of an ``mlx.nn``
            function (e.g. ``"relu"``), a callable or ``None``.
            Default: ``"relu"``
        train_norm: Whether the degree normalization constants are
            learnable. Default: ``False``
        **kwargs: Additional arguments of
            :class:`~mlx_graphs.nn.MessagePassing`.

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import PNAConv

        edge_index = mx.array([[0, 1, 2, 3, 4], [1, 0, 1, 4, 3]])
        node_features = mx.random.normal((5, 16))

        deg = PNAConv.get_degree_histogram([(edge_index, 5)])
        conv = PNAConv(
            16,
            32,
            aggregators=["mean", "min", "max", "std"],
            scalers=["identity", "amplification", "attenuation"],
            deg=deg,
            towers=4,
        )
        >>> conv(edge_index, node_features).shape
        (5, 32)
    """

    def __init__(
        self,
        node_features_dim: int,
        out_features_dim: int,
        aggregators: List[str],
        scalers: List[str],
        deg: Union[mx.array, np.ndarray, Sequence[int]],
        edge_features_dim: Optional[int] = None,
        towers: int = 1,
        pre_layers: int = 1,
        post_layers: int = 1,
        divide_input: bool = False,
        act: Union[str, Callable, None] = "relu",
        train_norm: bool = False,
        **kwargs,
    ):
        aggr = DegreeScalerAggregation(aggregators, scalers, deg, train_norm)
        super().__init__(aggr=aggr, **kwargs)

        if divide_input and node_features_dim % towers != 0:
            raise ValueError(
                f"`node_features_dim` ({node_features_dim}) must be divisible by "
                f"`towers` ({towers}) when `divide_input=True`"
            )
        if out_features_dim % towers != 0:
            raise ValueError(
                f"`out_features_dim` ({out_features_dim}) must be divisible by "
                f"`towers` ({towers})"
            )

        self.node_features_dim = node_features_dim
        self.out_features_dim = out_features_dim
        self.edge_features_dim = edge_features_dim
        self.towers = towers
        self.divide_input = divide_input

        self.F_in = node_features_dim // towers if divide_input else node_features_dim
        self.F_out = out_features_dim // towers

        if edge_features_dim is not None:
            self.edge_encoder = Linear(edge_features_dim, self.F_in)

        act_fn = _resolve_act(act)
        num_inputs = 3 if edge_features_dim is not None else 2
        post_in = (len(aggregators) * len(scalers) + 1) * self.F_in
        self.pre_nns = [
            self._mlp(num_inputs * self.F_in, self.F_in, pre_layers, act_fn)
            for _ in range(towers)
        ]
        self.post_nns = [
            self._mlp(post_in, self.F_out, post_layers, act_fn) for _ in range(towers)
        ]
        self.lin = Linear(out_features_dim, out_features_dim)

    @staticmethod
    def _mlp(
        in_dim: int, out_dim: int, num_layers: int, act: Optional[Callable]
    ) -> nn.Sequential:
        layers: list = [Linear(in_dim, out_dim)]
        for _ in range(num_layers - 1):
            if act is not None:
                layers.append(act)
            layers.append(Linear(out_dim, out_dim))
        return nn.Sequential(*layers)

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_features: Optional[mx.array] = None,
    ) -> mx.array:
        """Computes the forward pass of PNAConv.

        Args:
            edge_index: Input edge index of shape `[2, num_edges]`
            node_features: Input node features of shape
                `[num_nodes, node_features_dim]`
            edge_features: Optional edge features of shape
                `[num_edges, edge_features_dim]`

        Returns:
            The node embeddings, of shape `[num_nodes, out_features_dim]`
        """
        if edge_features is not None and self.edge_features_dim is None:
            raise ValueError(
                "PNAConv got `edge_features` but was built without "
                "`edge_features_dim`"
            )
        num_nodes = node_features.shape[0]
        # [num_nodes, towers, F_in]
        if self.divide_input:
            x = node_features.reshape(num_nodes, self.towers, self.F_in)
        else:
            x = mx.broadcast_to(
                node_features.reshape(num_nodes, 1, self.F_in),
                (num_nodes, self.towers, self.F_in),
            )

        # [num_nodes, towers, num_aggregators * num_scalers * F_in]
        out = self.propagate(
            edge_index=edge_index,
            node_features=x,
            message_kwargs={"edge_features": edge_features},
        )

        out = mx.concatenate([x, out], axis=-1)
        outs = [nn_(out[:, i]) for i, nn_ in enumerate(self.post_nns)]
        return self.lin(mx.concatenate(outs, axis=1))

    def message(
        self,
        src_features: mx.array,
        dst_features: mx.array,
        edge_features: Optional[mx.array] = None,
        **kwargs,
    ) -> mx.array:
        if edge_features is not None:
            edge = self.edge_encoder(edge_features)
            edge = mx.broadcast_to(
                edge.reshape(-1, 1, self.F_in), (edge.shape[0], self.towers, self.F_in)
            )
            h = mx.concatenate([dst_features, src_features, edge], axis=-1)
        else:
            h = mx.concatenate([dst_features, src_features], axis=-1)

        return mx.stack([nn_(h[:, i]) for i, nn_ in enumerate(self.pre_nns)], axis=1)

    @staticmethod
    def get_degree_histogram(graphs: Iterable) -> mx.array:
        """Histogram of the in-degrees of the nodes of ``graphs``, to pass as
        ``deg`` to :class:`PNAConv`, :class:`~mlx_graphs.nn.PNA` or
        :class:`~mlx_graphs.nn.aggr.DegreeScalerAggregation`.

        ``deg[i]`` is the number of nodes with in-degree ``i`` (number of
        edges ``edge_index[1] == node``). It is computed on the host with
        numpy, once, before training.

        Args:
            graphs: A :class:`~mlx_graphs.data.GraphData`, or an iterable of
                them (a dataset, a list, or a
                :class:`~mlx_graphs.loaders.Dataloader` yielding batches), or
                of ``(edge_index, num_nodes)`` tuples.

        Returns:
            An integer array with the degree histogram.

        Example:

        .. code-block:: python

            from mlx_graphs.loaders import Dataloader
            from mlx_graphs.nn import PNAConv

            deg = PNAConv.get_degree_histogram(Dataloader(train_dataset, 128))
        """
        if hasattr(graphs, "edge_index"):
            graphs = [graphs]

        hist = np.zeros(1, dtype=np.int64)
        for graph in graphs:
            if isinstance(graph, tuple):
                edge_index, num_nodes = graph
            else:
                edge_index, num_nodes = graph.edge_index, graph.num_nodes
            dst = np.asarray(edge_index[1]).astype(np.int64).reshape(-1)
            deg = np.bincount(dst, minlength=int(num_nodes))
            counts = np.bincount(deg, minlength=hist.size)
            if counts.size > hist.size:
                counts[: hist.size] += hist
                hist = counts
            else:
                hist += counts
        return mx.array(hist)
