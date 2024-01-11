from typing import Any, Union

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.utils import scatter, gather_src_dst
from mlx_graphs.typing import ArrayTuple


class MessagePassing(nn.Module):
    r"""Base class for creating Message Passing Neural Networks (MPNNs) [1].

    Inherit this class to build arbitrary GNN models based on the message
    passing paradigm. This implementation is inspired from PyTorch Geometric [2].

    Args:
        aggr (str): the aggregation strategy used to aggregate messages

    References:
        [1] Gilmer et al. Neural Message Passing for Quantum Chemistry.
        https://proceedings.mlr.press/v70/gilmer17a/gilmer17a.pdf

        [2] Fey et al. PyG
        https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.MessagePassing.html
    """

    def __init__(self, aggr="add"):
        super().__init__()

        self.aggr = aggr
        self.node_dim = None

    def __call__(self, x: mx.array, edge_index: mx.array, **kwargs: Any):
        raise NotImplementedError

    def propagate(self, x: Union[mx.array, ArrayTuple], edge_index: mx.array, **kwargs: Any) -> mx.array:
        r"""Computes messages from neighbors, aggregates them and updates
        the final node embeddings.

        Args:
            x (Union[mx.array, ArrayTuple]): input node features/embeddings
            edge_index (mx.array): graph representation of shape (2, |E|) in COO format
            **kwargs (Any): arguments to pass to message, aggregate and update
        """
        assert isinstance(edge_index, mx.array) and edge_index.shape[0] == 2, \
            f"Edge index should be an array with shape (2, |E|)."

        if isinstance(x, tuple):
            x_i, x_j = x
            assert isinstance(x_i, mx.array) and isinstance(x_j, mx.array), \
                "MessagePassing only supports mx.array type for `x`."
        else:
            assert isinstance(x, mx.array), \
                "MessagePassing only supports mx.array type for `x`."
            x_i, x_j = gather_src_dst(x, edge_index)

        self.node_dim = x_i.shape
        dst_idx = edge_index[1]

        messages = self.message(x_i, x_j, **kwargs)
        aggregated = self.aggregate(messages, dst_idx, **kwargs)
        output = self.update_(aggregated, **kwargs)

        return output

    def message(self, x_i: mx.array, x_j: mx.array, **kwargs: Any) -> mx.array:
        r"""Computes messages between connected nodes.

        Args:
            x_i (mx.array): source node embeddings
            x_j (mx.array): destination node embeddings
            **kwargs (Any): optional args to compute messages
        """
        return x_i

    def aggregate(
        self, messages: mx.array, indices: mx.array, **kwargs: Any
    ) -> mx.array:
        r"""Aggregates the messages using the `self.aggr` strategy.

        Args:
            messages (mx.array): computed messages
            indices: (mx.array): indices representing the nodes that receive messages
            **kwargs (Any): optional args to aggregate messages
        """
        return scatter(messages, indices, self.node_dim, self.aggr)

    # NOTE: this method can't be named `update()`, or the grads will be always set to 0.
    def update_(self, aggregated: mx.array, **kwargs: Any) -> mx.array:
        r"""Updates the final embeddings given the aggregated messages.

        Args:
            aggregated (mx.array): aggregated messages
            **kwargs (Any): optional args to update messages
        """
        return aggregated
