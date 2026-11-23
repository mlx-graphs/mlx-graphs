from typing import Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_graphs.utils.scatter import scatter
from mlx_graphs.utils.transformations import _to_numpy


class PANConv(nn.Module):
    r"""The path integral based convolutional operator from the `"Path
    Integral Based Convolution and Pooling for Graph Neural Networks"
    <https://arxiv.org/abs/2006.16811>`_ paper.

    .. math::
        \mathbf{X}^{\prime} = \mathbf{M} \mathbf{X} \mathbf{W}

    where :math:`\mathbf{M}` denotes the normalized and learned maximal
    entropy transition (MET) matrix that includes neighbors up to
    ``filter_size`` hops:

    .. math::
        \mathbf{M} = \mathbf{Z}^{-1/2} \sum_{n=0}^L e^{-\frac{E(n)}{T}}
        \mathbf{A}^n \mathbf{Z}^{-1/2}

    As in PyG, the weight of :math:`\mathbf{A}^n` is the product of the
    learned weights :math:`w_0 \cdots w_n` (all initialized to 0.5), and
    :math:`\mathbf{Z}` holds the number of non-zero entries of each row.

    The layer also returns :math:`\mathbf{M}` (in COO form, row
    :math:`i` gathering from column :math:`j`), which
    :class:`~mlx_graphs.nn.PANPooling` takes as input.

    Args:
        node_features_dim: Size of the input node features.
        out_features_dim: Size of the output node features.
        filter_size: The number of hops :math:`L`.

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import PANConv

        conv = PANConv(16, 32, filter_size=2)
        edge_index = mx.array([[0, 1, 1, 2], [1, 0, 2, 1]])
        node_features = mx.random.normal((3, 16))
        h, (m_index, m_weights) = conv(edge_index, node_features)
        >>> h.shape
        (3, 32)
    """

    def __init__(self, node_features_dim: int, out_features_dim: int, filter_size: int):
        super().__init__()
        self.node_features_dim = node_features_dim
        self.out_features_dim = out_features_dim
        self.filter_size = filter_size
        # PyG's Linear default initialization, i.e. PyTorch's
        self.lin = nn.Linear(node_features_dim, out_features_dim)
        self.weight = mx.full((filter_size + 1,), 0.5)

    def panentropy(
        self, edge_index: mx.array, num_nodes: int
    ) -> Tuple[mx.array, mx.array]:
        """The un-normalized MET matrix :math:`\\sum_n (w_0 \\cdots w_n)
        \\mathbf{A}^n`, coalesced, where :math:`\\mathbf{A}_{i, j}` counts
        the edges from node ``j`` to node ``i``."""
        from mlx_graphs.nn.pooling._sparse import coalesce_sum, spspmm

        adj_t = mx.stack([edge_index[1], edge_index[0]])
        loops = np.arange(num_nodes)
        tmp_index = mx.array(np.stack([loops, loops]).astype(np.int32))
        tmp_value = mx.ones((num_nodes,)) * self.weight[0]
        indices, values = [tmp_index], [tmp_value]
        for i in range(1, self.filter_size + 1):
            tmp_index, tmp_value = spspmm(tmp_index, tmp_value, adj_t, None, num_nodes)
            tmp_value = tmp_value * self.weight[i]
            indices.append(tmp_index)
            values.append(tmp_value)
        index = np.concatenate([_to_numpy(i) for i in indices], axis=1)
        return coalesce_sum(
            index[0].astype(np.int64),
            index[1].astype(np.int64),
            mx.concatenate(values),
            num_nodes,
        )

    def __call__(
        self, edge_index: mx.array, node_features: mx.array
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        """Computes the node embeddings.

        Args:
            edge_index: Edges of shape ``[2, num_edges]``.
            node_features: Node features of shape ``[num_nodes,
                node_features_dim]``.

        Returns:
            A tuple ``(node_features, (m_index, m_weights))`` with the node
            embeddings and the MET matrix :math:`\\mathbf{M}` in COO form.
        """
        num_nodes = node_features.shape[0]
        m_index, m_raw = self.panentropy(edge_index, num_nodes)
        rows = _to_numpy(m_index[0]).astype(np.int64)
        deg = np.bincount(rows, minlength=num_nodes).astype(np.float32)
        with np.errstate(divide="ignore"):
            deg_inv_sqrt = np.where(deg > 0, deg**-0.5, 0.0).astype(np.float32)
        deg_inv_sqrt = mx.array(deg_inv_sqrt)
        row, col = m_index[0], m_index[1]
        m_weights = deg_inv_sqrt[row] * m_raw * deg_inv_sqrt[col]

        out = scatter(m_weights[:, None] * node_features[col], row, num_nodes, "add")
        return self.lin(out), (m_index, m_weights)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.node_features_dim}, "
            f"{self.out_features_dim}, filter_size={self.filter_size})"
        )
