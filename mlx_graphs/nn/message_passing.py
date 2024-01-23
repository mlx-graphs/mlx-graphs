from typing import Any, Dict, Optional, Tuple, Union, get_args

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.utils import ScatterAggregations, get_src_dst_features, scatter


class MessagePassing(nn.Module):
    r"""Base class for creating Message Passing Neural Networks (MPNNs) [1].

    Inherit this class to build arbitrary GNN models based on the message
    passing paradigm. This implementation is inspired from PyTorch Geometric [2].

    Args:
        aggr (str): Aggregation strategy used to aggregate messages

    References:
        [1] Gilmer et al. Neural Message Passing for Quantum Chemistry.
        https://proceedings.mlr.press/v70/gilmer17a/gilmer17a.pdf

        [2] Fey et al. PyG
        https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.MessagePassing.html
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
        r"""Computes messages from neighbors, aggregates them and updates
        the final node embeddings.

        Args:
            edge_index (mx.array): Graph representation of shape (2, |E|) in COO format
            node_features (Union[mx.array, Tuple[mx.array, mx.array]]): Input node features/embeddings.
                Can be either an array or a tuple of arrays, for distinct src and dst node features.
            message_kwargs (Dict, optional): Arguments to pass to the `message` method
            aggregate_kwargs (Dict, optional): Arguments to pass to the `aggregate` method
            update_kwargs (Dict, optional): Arguments to pass to the `update_nodes` method
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

        messages = self.message(
            src_features, dst_features, **message_kwargs
        )  # (|E| -> |E|)
        aggregated = self.aggregate(
            messages, dst_idx, **aggregate_kwargs
        )  # (|E| -> |N|)
        output = self.update_nodes(aggregated, **update_kwargs)  # (|N| -> |N|)

        return output

    def message(
        self, src_features: mx.array, dst_features: mx.array, **kwargs: Any
    ) -> mx.array:
        r"""Computes messages between connected nodes.

        Args:
            src_features (mx.array): Source node embeddings
            dst_features (mx.array): Destination node embeddings
            **kwargs (Any): Optional args to compute messages
        """
        return src_features

    def aggregate(
        self, messages: mx.array, indices: mx.array, **kwargs: Any
    ) -> mx.array:
        r"""Aggregates the messages using the `self.aggr` strategy.

        Args:
            messages (mx.array): Computed messages
            indices: (mx.array): Indices representing the nodes that receive messages
            **kwargs (Any): Optional args to aggregate messages
        """
        return scatter(messages, indices, self.num_nodes, self.aggr)

    def update_nodes(self, aggregated: mx.array, **kwargs: Any) -> mx.array:
        r"""Updates the final embeddings given the aggregated messages.

        Args:
            aggregated (mx.array): aggregated messages
            **kwargs (Any): optional args to update messages
        """
        return aggregated
