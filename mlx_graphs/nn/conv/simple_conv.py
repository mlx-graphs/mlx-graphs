from typing import Literal, Optional, get_args

import mlx.core as mx

from mlx_graphs.nn.message_passing import MessagePassing
from mlx_graphs.utils import ScatterAggregations, add_self_loops

CombineRootFunctions = Literal["sum", "cat", "self_loop"]


class SimpleConv(MessagePassing):
    """A simple non-trainable message passing layer.

    .. math::
        \\mathbf{x}^{\\prime}_i = \\bigoplus_{j \\in \\mathcal{N(i)}} e_{j,i} \\cdot
        \\mathbf{x}_j

    where :math:`\\bigoplus` denotes an aggregation strategy (e.g. ``'add'``,\
    ``'mean'``), and :math:`e_{j,i}` denotes the edge weight between the source\
    node :math:`j` and the target node :math:`i` and :math:`\\mathcal{N(i)}` denotes\
    the neighbors of node :math:`i` and :math:`\\mathbf{x}_j` denotes the features of\
    node :math:`j`.


    Inspired by the `SimpleConv PyG layer\
    <https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.SimpleConv.html>`_.

    Args:
        aggr: Aggregation strategy used to aggregate messages,
            e.g. ``'add'``, ``'mean'``, ``'max'``. Default: ``'add'``
        combine_root_func: Strategy used to combine the features from the root nodes.\
        Available values: ``'sum'``, ``'cat'``, ``'self_loop'`` or ``None``).\
        ``'sum'``: It sums up the neighborhood's message and root node's features.\
        ``'cat'``: It concatenates neihborhood's message and root node's features.\
        ``'self_loop'``: It adds a self-loop for each root node and aggregates the\
        messages. If the graph is weighted then the edge weights of self-loops will\
        be set to :obj:`1`. Default: ``None``

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import SimpleConv

        # Sum the messages from the neighbors.
        # Use a self-loop for each root node.
        conv = SimpleConv(aggr="add", combine_root_func="self_loop")
        node_features = mx.ones((5, 3))
        edge_index = mx.array([[0, 1, 2, 3, 4], [0, 0, 1, 1, 3]])
        edge_weights = mx.array([10, 20, 5, 2, 15])
        h = conv(edge_index, node_features, edge_weights)
    """

    def __init__(
        self,
        aggr: ScatterAggregations = "add",
        combine_root_func: Optional[CombineRootFunctions] = None,
        **kwargs,
    ):
        super(SimpleConv, self).__init__(aggr, **kwargs)

        if combine_root_func is not None and combine_root_func not in get_args(
            CombineRootFunctions
        ):
            raise ValueError(
                "Invalid combine_root_func.",
                f"Available values are {get_args(CombineRootFunctions)}",
            )
        self.combine_root_func = combine_root_func

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_weights: Optional[mx.array] = None,
    ) -> mx.array:
        """Computes the forward pass of SimpleConv.

        Args:
            edge_index: Input edge index of shape `[2, num_edges]`
            node_features: Input node features
            edge_weights: Edge weights leveraged in message passing. Default: ``None``

        Returns:
            The computed node embeddings
        """
        # Add self-loops, if needed.
        if self.combine_root_func == "self_loop":
            # Edge weights exist.
            if edge_weights is not None:
                edge_index, edge_weights = add_self_loops(
                    edge_index, edge_weights.reshape(-1, 1)
                )
                edge_weights = edge_weights.reshape(-1)
            # Edge weights do not exist.
            else:
                edge_index = add_self_loops(edge_index)

        # Compute messages and aggregate them.
        output = self.propagate(
            edge_index=edge_index,
            node_features=node_features,
            message_kwargs={"edge_weights": edge_weights},
        )

        # Combine the root node features.
        if self.combine_root_func is not None:
            if self.combine_root_func == "sum":
                output = output + node_features
            elif self.combine_root_func == "cat":
                output = mx.concatenate([node_features, output], axis=-1)

        return output
