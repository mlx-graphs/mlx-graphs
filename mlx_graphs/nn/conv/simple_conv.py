from typing import Literal, Optional, get_args

import mlx.core as mx

from mlx_graphs.nn.message_passing import MessagePassing
from mlx_graphs.utils import ScatterAggregations, add_self_loops

CombineRootFunctions = Literal["sum", "cat", "self_loop"]


class SimpleConv(MessagePassing):
    """A simple non-trainable message passing layer.

    Inspired by the SimpleConv PyG layer `here \
    <https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.SimpleConv.html>`_.

    Args:
        aggr: Aggregation strategy used to aggregate messages,
            e.g. "add", "mean", "max". Default: ``add``
        combine_root_func: Strategy used to combine the features from the root nodes
        (values: "sum", "cat", "self_loop" or `None`). Default: `None`
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
            mx.array: The computed node embeddings
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
