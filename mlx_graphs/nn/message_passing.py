from typing import Any, Dict, Optional, Tuple, Union, get_args

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.utils import ScatterAggregations, get_src_dst_features, scatter


class MessagePassing(nn.Module):
    """Base class for creating Message Passing Neural Networks (MPNNs) [1].

    Inherit this class to build arbitrary GNN models based on the message
    passing paradigm. This implementation is inspired from PyTorch Geometric [2].

    Args:
        aggr: Aggregation strategy used to aggregate messages

    References:
        [1] `Gilmer et al. Neural Message Passing for Quantum Chemistry. <https://proceedings.mlr.press/v70/gilmer17a/gilmer17a.pdf>`_

        [2] `Fey et al. PyG <https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.MessagePassing.html>`_
    """

    def __init__(self, aggr: ScatterAggregations = "add"):
        super().__init__()
        if aggr not in get_args(ScatterAggregations):
            raise ValueError(
                "Invalid aggregation function.",
                f"Available values are {get_args(ScatterAggregations)}",
            )
        self.aggr: ScatterAggregations = aggr
        self.num_nodes = None

    def __call__(self, node_features: mx.array, edge_index: mx.array, **kwargs: Any):
        raise NotImplementedError

    def propagate(
        self,
        edge_index: mx.array,
        node_features: Union[mx.array, Tuple[mx.array, mx.array]],
        message_kwargs: Optional[Dict] = {},
        aggregate_kwargs: Optional[Dict] = {},
        update_kwargs: Optional[Dict] = {},
    ) -> mx.array:
        """Computes messages from neighbors, aggregates them and updates
        the final node embeddings.

        Args:
            edge_index: Graph representation of shape `[2, num_edges]`
            node_features: Input node features/embeddings.
                Can be either an array or a tuple of arrays, for distinct src and dst node features.
            message_kwargs: Arguments to pass to the `message` method
            aggregate_kwargs: Arguments to pass to the `aggregate` method
            update_kwargs: Arguments to pass to the `update_nodes` method
        """
        if not (isinstance(edge_index, mx.array) and edge_index.shape[0] == 2):
            raise ValueError("Edge index should be an array with shape (2, |E|)")
        if isinstance(node_features, tuple):
            if len(node_features) != 2 or not all(
                isinstance(array, mx.array) for array in node_features
            ):
                raise ValueError(
                    "Invalid shape for `node_features`, should be a tuple of 2 mx.array"
                )
        else:
            if not isinstance(node_features, mx.array):
                raise ValueError(
                    f"Invalid shape for `node_features`, should be an `mx.array`, found {type(node_features)}"
                )

        src_features, dst_features = get_src_dst_features(edge_index, node_features)

        self.num_nodes = (
            node_features if isinstance(node_features, mx.array) else node_features[0]
        ).shape[0]
        dst_idx = edge_index[1]

        # shapes: (|E| -> |E|)
        messages = self.message(
            src_features=src_features,
            dst_features=dst_features,
            **message_kwargs,
        )

        # shapes: (|E| -> |N|)
        aggregated = self.aggregate(
            messages=messages,
            indices=dst_idx,
            **aggregate_kwargs,
        )

        # shapes: (|N| -> |N|)
        output = self.update_nodes(
            aggregated=aggregated,
            **update_kwargs,
        )

        return output

    def message(
        self, src_features: mx.array, dst_features: mx.array, **kwargs
    ) -> mx.array:
        """Computes messages between connected nodes.

        By default, returns the features of source nodes.
        Optional ``edge_weights`` can be directly integrated in ``kwargs``

        Args:
            src_features: Source node embeddings
            dst_features: Destination node embeddings
            edge_weights: Array of scalars with shape (num_edges,) or (num_edges, 1)
                used to weigh neighbor features during aggregation. Default: ``None``
            **kwargs: Optional args to compute messages
        """
        edge_weights = kwargs.get("edge_weights", None)

        return (
            src_features
            if edge_weights is None
            else edge_weights.reshape(-1, 1) * src_features
        )

    def aggregate(self, messages: mx.array, indices: mx.array, **kwargs) -> mx.array:
        """Aggregates the messages using the `self.aggr` strategy.

        Args:
            messages: Computed messages
            indices: Indices representing the nodes that receive messages
            **kwargs: Optional args to aggregate messages
        """
        return scatter(messages, indices, self.num_nodes, self.aggr)

    def update_nodes(self, aggregated: mx.array, **kwargs) -> mx.array:
        """Updates the final embeddings given the aggregated messages.

        Args:
            aggregated: aggregated messages
            **kwargs: optional args to update messages
        """
        return aggregated
